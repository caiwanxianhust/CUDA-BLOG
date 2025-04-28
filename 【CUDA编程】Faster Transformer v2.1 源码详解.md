#! https://zhuanlan.zhihu.com/p/654368698
![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qOuFVvibRGAiaf5MwEHJtsRJLuebFyQdu9d6DqEMZR1ZdRBu6hoGwCye3kpOpjnKogBe09FEFOqvTg/640?wx_fmt=png)
# 【CUDA编程】Faster Transformer v2.1 源码详解

**写在前面**：本文将对 Faster Transformer v2.1 版本源码进行解读，重点介绍该版本基于 v1.0 和 v2.0 所做的优化内容，剖析源码作者优化意图，为了便于交流讨论，除公众号：**后来遇见AI** 以外，本文也将在知乎进行发布，欢迎各位读者阅读并给出意见。  

## 1 v2.1 版本发布背景
在 FasterTransformer v1.0 中，Nvidia 提供了一个高度优化的 BERT Transformer Encoder 模块，主要应用于序列标注推理场景，笔者针对源码的实现逻辑和优化技巧进行了深度剖析，有兴趣的读者可以移步——[【CUDA编程】Faster Transformer v1.0 源码详解](https://zhuanlan.zhihu.com/p/647012855)。  
在 FasterTransformer v2.0 中，Nvidia 添加了一个高度优化的 Decoder 模块和一套推理方案 Decoding 模型。其中，Decoder 相当于我们常说的 decoder layer；而 Decoding 则包含了整个解码的流程，包括词嵌入、位置编码、解码层和束搜索等过程，相当于我们常说的 decoder model。同样，笔者针对 v2.0 版本新增的内容进行了优化解读，有兴趣的读者可以移步——[【CUDA编程】Faster Transformer v2.0 源码详解](https://zhuanlan.zhihu.com/p/650462095)。    
在 FasterTransformer v2.1 中，官方主要添加了 3 块优化内容。第一点是考虑到 PyTorch 的用户越来越多，官方添加了对 PyTorch 的支持，这点不在本文的讨论范畴。第二个特点是支持 [Effective Transformer](https://github.com/bytedance/effective_transformer)，该优化思路来自字节跳动算法团队，计算模型中去除了 encoder 输入的无效填充，从而降低了计算开销。第三，除了使用束搜索进行解码外，还提供了基于采样的解码策略。除此之外，Nvidia 还对 Encoder、Decoder 和 beam search 等诸多模块的内核进行了优化，进一步提高了 FasterTransformer 的推理速度。因此本文的解读也主要聚焦于 3 个方面：Effective Transformer、sampling decoding、内核优化，针对其他未发生变更的内容，请读者阅读笔者的前两篇文章。  

## 2 整体架构
同前面两个版本一样，v2.1 的底层由 CUDA 和 cuBLAS 实现，提供 C++ API 和 TensorFlow/PyThorch OP。用户可以将它们集成到 TensorFlow、PyTorch 或其他在本机 C++ 中构建的推理服务代码中。此外官方还提供了一些简单的示例代码来演示如何使用 Encoder、Decoder 以及在 C++、TensorFlow 和 PyTorch 中执行 Decoding 过程。下面是整体架构图：  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qOuFVvibRGAiaf5MwEHJtsRJLuebFyQdu9d6DqEMZR1ZdRBu6hoGwCye3kpOpjnKogBe09FEFOqvTg/640?wx_fmt=png)

源码地址如下，有兴趣的读者可以前往下载：
>https://github.com/NVIDIA/FasterTransformer/tree/v2.1/

## 3 Effective Transformer
关于 Transformer Encoder 的逻辑笔者在之前的文章中有详细阐述，这里笔者不打算再重复讲解，解读重心会放在“Effective”的部分。当使用 Transformer 对一批输入序列进行编码时，我们通常将输入序列视为一个矩阵，其列数等于所有序列的最大长度。Faster Transformer 可以非常有效地处理所有序列长度大致相同的情况。然而，如果同一批中序列的长度变化很大，将它们填充到相同的长度意味着对内存和计算资源的巨大浪费。考虑下面一个例子：  
```python
bert_input = [["Hi"], ["Picking"], ["The", "seed", "of", "Job's", "tears"]]
bert_tokens = [[1], [2], [3,4,5,6,7]]
bert_tokens_padded = [[1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 4, 5, 6, 7]]
bert_tokens_mask = [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
```
对于输入的 3 个样本来说，实际有效的 word 只有 `1+1+5=7` 个，但是我们要用一个 `3 * 5` 的矩阵来计算，中间其实有一半的元素是无效的，这些无效元素既浪费了内存又占用了计算资源。所以我们在想，能不能就用 `[1,2,3,4,5,6,7]` 这 7 个元素来参与计算？  
在 Effective Transformer 中，会根据不同的计算阶段，动态删除和恢复填充值，从而减少资源占用。  
### 3.1 计算偏移量
通常上游传过来的 tensor 一般是 padded 之后的规则矩阵，假设形状为 `[batch_size, seq_len, hidden_units]` 记为 `tensor_padded`，删除填充值后的形状为 `[valid_word_num, hidden_units]` 记为 `tensor`，要想动态的删除和恢复填充值，也就是说要找到 `tensor_padded` 和 `tensor` 的对应关系，源码中用 `build_sequence_length_padding_offset_kernelLauncher` 函数解决这个问题。
```cpp
/**
 * @brief 计算偏移量
 * 
 * @param sequence_length         [batch_size,]
 * @param batch_size        
 * @param max_seq_len             
 * @param valid_word_num          [1,]
 * @param tmp_mask_offset         [valid_word_num]
 * @return __global__ 
 */
__global__ void build_sequence_length_padding_offset(const int* sequence_length, 
  const int batch_size, const int max_seq_len, int* valid_word_num, int* tmp_mask_offset)
{
  // do cumulated sum
  int total_seq_len = 0;
  int cum_offset = 0;
  int index = 0;
  for(int i = 0; i < batch_size; i++) 
  {
    const int seq_len = sequence_length[i];
    for(int j = 0; j < seq_len; j++)
    {
      tmp_mask_offset[index] = cum_offset;
      index++;
    }
    cum_offset += max_seq_len - seq_len;
    total_seq_len += seq_len;
  }
  valid_word_num[0] = total_seq_len;
}

void build_sequence_length_padding_offset_kernelLauncher(const int* sequence_length, 
  const int batch_size, const int max_seq_len, int* valid_word_num, int* tmp_mask_offset,
  cudaStream_t stream)
{
  build_sequence_length_padding_offset<<<1, 1, 0, stream>>>(sequence_length, 
    batch_size, max_seq_len, valid_word_num, tmp_mask_offset);
}
```
可以看到函数体内部执行了一个 `1*1` 的核函数，也就是说这个核函数完全没有并行，全部逻辑在一个线程里完成。可能有读者要问，既然不并行，那为啥不在主机端直接用 C++ 完成，这是因为 `tmp_mask_offset` 和 `valid_word_num` 这都是设备端的变量，如果在主机端计算还需要一次内存拷贝操作，而 host-device 内存拷贝是比较耗时的，所以干脆就在设备端开一个线程算了。  
核函数内的计算逻辑比较简单，直接看下面的图就可以了。  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rSa2fpOUQfSuyfJn1HbKFJUeRqsz2vjt8K2dicr6a2V3YvjvLMJ2lAd6Zhbe1hjcVMyib7hEicBlpcg/640?wx_fmt=png)
根据样本长度 `sequence_length` 计算了两个指标 `valid_word_num` 和 `id_offset`，用于后面动态删除和恢复矩阵，`tensor` 的行索引加上 `id_offset` 就得到了对应行在 `tensor_padded` 中的行索引。  

### 3.2 删除填充值
以形状为 `[batch_size, seq_len, hidden_units]` 的 `tensor_padded` 矩阵为例，如果计算过程只是在最后一个维度，比如右乘一个形如 `[hidden_uints, new_units]` 的矩阵，那其实完全可以去掉 `tensor_padded` 中的填充值之后再计算。这里官方提供了两个核函数进行删除填充值的操作：第一个就是单纯的删除填充值函数 `remove_sequence_length_padding_kernelLauncher`，第二个是和 transpose 操作融合后的 `transpose_rebuild_padding`，读者看起来可能会一脸懵逼，为什么删除函数命名要用 `rebuild`？没错，这里应该是源码作者笔误，导致了挂羊头卖狗肉的行为。

#### 3.2.1 remove_sequence_length_padding_kernelLauncher
单纯地删除填充值的计算逻辑很简单，就是按两步，找到行索引，按索引拷贝元素即可，直接看代码。
```cpp
template<typename T>
__global__ void remove_sequence_length_padding(const T* src, T* tgt,
                                              const int* tmp_mask_offset,
                                              int* mask_offset,
                                              const int n)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  mask_offset[bid] = tmp_mask_offset[bid];
  const int src_seq_id = bid + mask_offset[bid];
  const int tgt_seq_id = bid;
  for(int i = tid; i < n; i += blockDim.x)
  {
    tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
  }
}

template<typename T>
void remove_sequence_length_padding_kernelLauncher(const T* src, T* tgt, 
                                                  const int* tmp_mask_offset, 
                                                  int* mask_offset,
                                                  const int m, const int n, cudaStream_t stream)
{
  // src: [batch_size*max_seq_len, hidden_dim]
  // tgt: [valid_word_num, hidden_dim]
  // m: valid_word_num
  // n: hidden_dim
  remove_sequence_length_padding<<<m, 256, 0, stream>>>(src, tgt, tmp_mask_offset, mask_offset, n);
}
```
`remove_sequence_length_padding` 核函数的 `grid_size` 设置为 `valid_word_num`，也就是 `tensor` 的行数，每个 block 处理一行元素，`block_size` 取 `256`，每个线程处理对应行的一个或多个元素，步长为 `256`。可以看到，源矩阵中的行索引 `src_seq_id` 就等于目标矩阵的行索引 `tgt_seq_id` 加上 `offset`。

#### 3.2.2 transpose_rebuild_padding
关于这个函数名的问题笔者前面已经吐槽过了，到此为止，这并不影响实际应用，不然也没法通过测试上线。。。这个函数内部实现了两个操作：transpose、删除填充值，下面来看一下代码。  
```cpp
template<typename T>
__global__
void transpose_rebuild_padding(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head,
  const int* mask_offset)
{
  // TODO: optimize this kernel? 
  // do remove_sequence_length_padding
  const int tid = threadIdx.x; // batch * seq_len or valid_word_num
  const int bid = blockIdx.x; // head_num * size_per_head

  const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int src_seq_id = (bid + mask_offset[bid]) % seq_len;

  const int dst_seq_id = bid;

  const int head_id = tid / size_per_head;
  const int hidden_id = tid % size_per_head;
  dst[dst_seq_id * head_num * size_per_head + tid] = src[ src_batch_id * head_num * seq_len * size_per_head +
    head_id * seq_len * size_per_head + src_seq_id * size_per_head + hidden_id];
}

transpose_rebuild_padding<DataType_><<<param_.valid_word_num, k, 0, stream>>>(transpose_dst_, dst, 
            batch_size, seq_len, head_num, size_per_head, param_.sequence_id_offset);
```
函数体内部有两行关于 `bid` 和 `tid` 的注释，这个是源码作者的注释，应该是写混了，不用去看，请听笔者一一解释。  
首先这个函数的应用场景是在 $Softmax(\frac{QK^T}{\sqrt{d_k}})V$ 之后，也就是说这里的源矩阵是转置之后的矩阵，形状为 `[batch_size, head_num, seq_len, size_per_head]`，计算目的有两个：transpose、和删除填充值。`gird_size` 设置为 `valid_word_num`，`block_size` 设置为 `head_num * size_per_head`，一个 thread 处理一个元素。函数内部通过偏移量确定源矩阵的 `batch_id` 和 `seq_id`，有了索引后，再按照矩阵维度顺序把形如 `[batch_size, head_num, seq_len, size_per_head]` 的源矩阵，转换为形如 `[valid_word_num, head_num, size_per_head]` 的无填充矩阵。  

### 3.4 恢复填充值
我们知道，attention 操作通常包括：Q/K/V 线性变换、$Softmax(\frac{QK^T}{\sqrt{d_k}})V$、transpose 以及 attention out 线性变换等几个步骤。其中 $Softmax(\frac{QK^T}{\sqrt{d_k}})V$ 中有 2 个 Strided Batched Gemm 操作，这两个矩阵乘法是涉及 `seq_len` 维度的，因为要计算 word 与 word 间的相似度以及 scores 的加权平均值，所以在计算前要先恢复填充值矩阵。这里有一点要说明，并不是说这一步计算必须得有填充值，是因为有了填充值可以调用矩阵乘法 API，从而实现更好的并行化计算，如果舍弃矩阵乘法写一个函数根据偏移量逐个 word 计算，也是可行的，但是性能极差，所以还不如动态恢复矩阵。  
关于恢复填充值的操作，源码提供了两个 kernel，第一个就是单纯的恢复填充值函数 `rebuild_sequence_length_padding_kernelLauncher`，第二个是和 add bias 操作融合后的 `add_QKV_bias_rebuild_padding`。

#### 3.4.1 rebuild_sequence_length_padding_kernelLauncher
恢复填充值的操作和删除填充值的操作是互逆的，只需要把握一点即可：
$$
row_{padded} = row + id\_offset 
$$
```cpp
template<typename T>
__global__ void rebuild_sequence_length_padding(const T* src, T* tgt,
                                            const int* mask_offset,
                                            const int n)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int tgt_seq_id = bid + mask_offset[bid];
  const int src_seq_id = bid;

  for(int i = tid; i < n; i += blockDim.x)
  {
    tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
  }
}

template<typename T>
void rebuild_sequence_length_padding_kernelLauncher(const T* src, T* tgt, 
                                                  const int* mask_offset, const int m, 
                                                  const int n, cudaStream_t stream)
{
  // src: [valid_word_num, hidden_dim]
  // tgt: [batch_size*max_seq_len, hidden_dim]
  rebuild_sequence_length_padding<<<m, 256, 0, stream>>>(src, tgt, mask_offset, n);
}
```
核函数执行配置参数与删除填充值的核函数一致，都是一个 block 处理一行元素。可以看到，源矩阵中的行索引 `tgt_seq_id` 就等于目标矩阵的行索引 `src_seq_id` 加上 `offset`。  

#### 3.4.2 add_QKV_bias_rebuild_padding
顾名思义，核函数内部实现了两个计算逻辑：add Q/K/V bias 和恢复填充值，我们来看下源码。
```cpp
template<typename T>
__global__
void add_QKV_bias_rebuild_padding(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int* mask_offset)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int bdim = blockDim.x;

  const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
  const int tgt_head_id = tid / size_per_head;
  const int tgt_hidden_id = tid % size_per_head;

  const int src_id = bid * bdim + tid;
  const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head + \
                    tgt_head_id * seq_len * size_per_head + \
                    tgt_seq_id * size_per_head + \
                    tgt_hidden_id;
  
  q_buf_[tgt_id] = Q[src_id] + bias_Q[tid];
  k_buf_[tgt_id] = K[src_id] + bias_K[tid];
  v_buf_[tgt_id] = V[src_id] + bias_V[tid];
}
```
核函数 `grid_size` 设置为 `valid_word_num`，一个 block 内部处理 `hidden_uints = head_num*size_per_head` 个元素。由于模型整体超参数要求 `hidden_uints` 不大于 `1024`，所以这里 `block_size` 直接就设置为 `hidden_units`，一个 thread 就处理一个元素。  
核函数内部最重要的逻辑就是 `tgt_id` 的计算，有两点需要把握。首先是 `tgt_batch_id` 和 `tgt_seq_id` 的确定，通过源矩阵的索引加偏移量后计算得到。然后是 `tgt_id` 的确定，按照 `[batch_size, head_num, seq_len, size_per_head]` 的顺序计算即可。  
不熟悉 attention 计算逻辑的读者可能会问，为什么就加了一个 add bias 的操作，核函数变得如此复杂？这其实是因为这里面其实还隐藏了一个 transpose 操作，把原来形状如 `[valid_word_num, head_num, size_per_head]` 的矩阵转换成了形状如 `[batch_size, head_num, seq_len, size_per_head]` 的矩阵，笔者在前面的文章说过，这是“多头”独立计算的逻辑使然。

## 4 采样解码
### 4.1 采样解码原理
关于解码策略，笔者在上一篇文章中介绍了贪心搜索（greedy search）和束搜索（beam search）两种方法，这两种方法统称为**基于搜索的解码策略**，其解码目标都是最大化生成概率，即高概率的 word 相比于低概率 word 有压倒性优势，在解码过程中低概率的 word 绝不可能被选中。  
除了基于搜索的解码策略以外，**基于概率采样的解码策略**也被广泛应用，相比基于搜索的解码方法，通过采样生成的文本通常具有更高的多样性，同时也在一定程度上缓解了生成通用和重复文本的问题。常用的基于概率采样的解码策略分为以下四种：**随机采样**、**带温度的随机采样**、**Top-k 采样**、**Top-p 采样**。  
#### 4.1.1 随机采样  
在解码时每个 step 都从当前概率分布 $P(y|Y_t,X)$ 中按照概率随机采样一个词，即 $\hat{y}_t \sim P(y|\hat{Y}_t,X)$。  
相比于按概率“掐尖”，这样会增大所选词的范围，引入更多的随机性。这个方法是谷歌开放式聊天机器人 Meena[DialoGPT、Meena] 采用的方式。当时那篇论文的结论就是这种随机采样的方法远好于 Beam Search。但这种随机采样也是有局限性的，容易产生上下文无关前后不一致的问题。而在开放闲聊领域，生成文本的长度都比较短，这种问题就被自然的淡化了。  
#### 4.1.2 带温度的随机采样  
尽管随机采样在一定程度上能避免生成重复的文本，但是，由于从整个词表中采样可能会采到与上下文无关的词，因此，随机采样得到的文本上下文常常不连贯。为了使得模型尽可能避免采样到低概率的词，一个有效的办法是设置一个名为“温度”（temperature）的参数来控制概率分布的弥散程度，该参数用 $T$ 表示，是一个大于 $0$ 的实数。形式化地说，生成过程中需要将概率分布的计算方式修改为：  
$$
P(y|Y_t,X) = Softmax(\frac{logits_t}{T})|_y
$$
当 $T = 1$ 时，即为原始的概率分布；当 $T < 1$ 时，得到的概率分布将更加尖锐，弥散程度更小，采样的随机性降低；当 $T \rightarrow 0$时，使用随机采样解码的效果近似于贪心搜素；当 $T > 1$ 时，得到的概率分布弥散程度更小，采样的随机性升高；当 $T \rightarrow \infty$ 时，使用随机采样解码的效果则近似于从均匀分布中随机采样。因此，合理设置 $T \in (0, 1)$ 可以避免随机采到概率较小的词。

#### 4.1.3 Top-k 采样  
除了设置温度来调整概率分布的弥散程度，Top-k 采样近来也被广泛使用。具体来说，在每个 step，解码器首先选择概率最高的 k 个 word 作为候选 word 构成一个集合，然后将这个子集中 word 的概率再归一化，最后从新的概率分布中采样。这个办法据说可以获得比 Beam Search 好很多的效果，但也有一个问题，就是这个 k 值不太好选。因为实际应用中概率分布变化比较大，有时候可能很均匀，有的时候比较集中。对于集中的情况还好说，当分布均匀时，一个较小的 k 容易丢掉很多优质候选词。但如果 k 定的太大，这个方法又会退化回普通采样。

#### 4.1.4 Top-p 采样
相比于 Top-k 方法从概率最高的 k 个候选词中采样，它不再取一个固定的 k，而是固定候选集合的概率密度和在整个概率分布中的比例。也就是构造一个最小候选集，使得
$$
\sum _{y \in V_{min}^p} P(y|\hat{Y}_t,X) >= p
$$
Top-p 采样根据生成概率从高到低在词表上选择累积概率恰好超过 $p$ 的候选 word 作为采样集合，再从这些候选 word 中采样出最终的结果。选出来这个集合之后也和 Top-k 采样一样，重新归一化集合内 word 的概率，并把集合外词的概率设为 $0$。    

### 4.2 调用链
sampling decoding 模块的核心逻辑封装在 decoding_sampling.h 文件的 DecodingSampling 类中，计算逻辑都在 forward 函数里，具体调用链如下：  
```cpp
DecodingSampling->DecodingSampling()  // 构造函数
DecodingSampling->forward()
    ->init_kernelLauncher
    ->loop for step
        ->embedding_lookup_sine_position_encoding_kernel_launcher
        ->loop for decoder layer
            ->decoder->initialize
            ->decoder->forward
        ->decoder->decoder_norm1
        ->cublasGemmEx
        ->sampling_kernel_kernelLauncher
```

### 4.3 DecodingSampling 构造函数
构造函数内部首先进行了 `candidate_num_` 和 `probability_threshold_` 的判断，不能同时为 0 或同时不为 0，这两个参数分别代表 Top-k 采样的 k 和 Top-p 采样的 p，意思是源码提供了两种采样解码策略，在初始化的时候必须确定使用哪一种。  
接下来就是一些内存分配的工作，和 v2.0 版本基本一致，笔者根据源码绘制了一份内存分布图如下。
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qZW7icHTwz3cKXLxCRPFibOIlwa6iaickibR98Ub87hvqpSUR40jKFq3e642MIDqQXiaekbRIKHUxpD1Kg/640?wx_fmt=png)
首先在构造函数内部初始化了 2 个二级指针 K_cache_ 和 V_cache，这个指针是用来存储解码过程中每个 step 对应的输入 tensor 经过 Dense 变换后的 key 矩阵和 value 矩阵，用于 self-attention 的。可以看到这两个指针申请的内存大小和之前 v2.0 版本 DecodingOpenNMT 类中有所不同，DecodingOpenNMT 中有两个元素，DecodingSampling 中只有一个，这是因为 DecodingOpenNMT 的解码策略只有一个 beam search，beam search 每轮结束后取 TopK 的时候会打乱顺序，需要一个元素暂存当前每个 beam 的 Key 和 Value，等 TopK 确定后再根据 `parent_ids` 更新 Key 和 Value。而使用采样解码策略就不存在这个问题，所以一个元素足矣。  
```
K_cache_ = new DataType_ *[1];
V_cache_ = new DataType_ *[1];
```
然后就是一系列 buffer size 的计算，用于内存申请和分配的，结合笔者整理的内存分布图可以非常容易的理解。  

### 4.4 forward 函数
这个函数的计算逻辑过于复杂，不适合单列一节，大致过程见调用链，当笔者把 forward 内部调用链中子模块讲清楚的时候，forward 也就清晰了。

### 4.5 init_kernelLauncher
关于初始化函数，源码针对 Top-k 和 Top-p 两种不同的解码策略分别给了一个核函数，我们分别来研究一下。
```cpp
if(args_.candidate_num_ != 0)
{
    init_kernelLauncher(finished_buf_, decoding_params.sequence_length, word_ids_buf_, cum_log_buf_,
        args_.start_id_, args_.batch_size_, 1, decoding_params.stream);
}
else if(args_.probability_threshold_ != 0.0)
{
    topp_initialization_kernelLauncher(finished_buf_,
                                        decoding_params.sequence_length, 
                                        word_ids_buf_,
                                        topp_id_vals_buf_,
                                        topp_offset_buf_,
                                        args_,
                                        decoding_params.stream);
}
```
#### 4.5.1 Top-k 采样初始化
Top-k 采样初始化时依然调用的是 v2.0 版本的 DecodingOpenNMT 中的 init_kernel 核函数，只是把 `beam_width` 设置为 `1` 表示不使用 beam search，这个函数主要实现以下几个功能：  
- `decoding_params.sequence_length` 初始化为 `0`
- `finished_buf_` 初始化为 `false`
- `word_ids` 初始化为 `start_id`
- `cum_log_probs` 初始化为 0

#### 4.5.2 Top-p 采样初始化
Top-p 采样初始化主要做了以下工作：
- `decoding_params.sequence_length` 初始化为 `0`
- `finished_buf_` 初始化为 `false`
- `word_ids` 初始化为 `start_id`
- `topp_offset_buf` 初始化为 `[0, vocab_size, ..., batch_size * vocab_size]`
- `topp_id_val_buf` 初始化为 `[[0, 1, ..., vocab_size-1], [0, 1, ..., vocab_size-1], ..., [0, 1, ..., vocab_size-1]]`，其实就是 `batch_size` 个索引向量。  

```cpp
/**
* @brief top-p 初始化
* 
* @param finished                  [batch_size,]
* @param sequence_length           [batch_size,]
* @param word_ids                  [batch_size,]
* @param topp_id_val_buf           [batch_size, vocab_size]
* @param topp_offset_buf           [batch_size + 1 向上取 4 的倍数, ]
* @param batch_size 
* @param vocab_size 
* @param start_id 
* @return __global__ 
*/
__global__ void topp_initialization_kernel(bool* finished,
                                        int* sequence_length, 
                                        int* word_ids,
                                        int* topp_id_val_buf,
                                        int* topp_offset_buf,
                                        const int batch_size, 
                                        const int vocab_size,
                                        const int start_id)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if(bid == 0)
    {
        for(int i = tid; i < batch_size + 1; i+= blockDim.x)
        {
            topp_offset_buf[i] = i * vocab_size;
        }
        
        for(int i = tid; i < batch_size; i+= blockDim.x)
        {
            finished[i] = false;
            sequence_length[i] = 0;
            word_ids[i] = start_id; 
        }
    }

    int index = tid + bid * blockDim.x;
    while(index < batch_size * vocab_size)
    {
        topp_id_val_buf[index] = index % vocab_size;
        index += blockDim.x * gridDim.x;
    }
}

topp_initialization_kernel<<<32, 512, 0, stream>>>(finished, sequence_length, word_ids, 
                                                    topp_id_val_buf, topp_offset_buf,
                                                    args.batch_size_, args.vocab_size_,
                                                    args.start_id_);
```

### 4.6 embedding_lookup_sine_position_encoding_kernel_launcher
该核函数做了两项工作：词嵌入（embedding lookup）、位置嵌入（sine_position），在 v2.0 版本这俩功能是通过两个核函数实现的，v2.1 版本把这两个核函数进行了融合，这块内容本来笔者是计划放在第 5 节来介绍的，但是为了保证采样模块的完整性，就先在这里说了。    
```cpp
  template <typename T>
  __global__ void embedding_lookup_sine_position_encoding_kernel(T* from_tensor,
                                                                const T* embedding_table, 
                                                                const T* position_encoding,
                                                                const int* word_ids,
                                                                const int hidden_units)
  {
      const int tid = threadIdx.x;
      const int bid = blockIdx.x;
      const int write_pos = tid + bid * blockDim.x;
      // 1. lookup the table
      // 2. multiply hidden_dim**0.5
      // 3. add the position encoding
      from_tensor[write_pos] = embedding_table[word_ids[bid] * hidden_units + tid] * 
                                (T)sqrtf(float(hidden_units)) + position_encoding[tid];
  }

  template <typename T>
  void embedding_lookup_sine_position_encoding_kernel_launcher(T* from_tensor,
                                                              const T* embedding_table, 
                                                              const T* position_encoding,
                                                              const int* word_ids,
                                                              const int batch_size,
                                                              const int hidden_units, 
                                                              cudaStream_t stream)
  {
      assert(hidden_units <= 1024);
      dim3 grid(batch_size);
      dim3 block(hidden_units);
      embedding_lookup_sine_position_encoding_kernel<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                                  embedding_table,
                                                                                  position_encoding,
                                                                                  word_ids,
                                                                                  hidden_units);
  }
```
核函数的 `grid_size` 和 `block_size` 分别设置为 `batch_size` 和 `hidden_units`，在函数内部做了以下三件事：
- 根据 `embedding_table` 查表赋值，把 `word_id` 转化为词向量
- 词向量的值乘以一个修正系数 `hidden_uints ** 0.5`，达到缩放效果，这一步在 v2.0 版本中是在 `sine_position_encoder_kernel` 核函数中进行的。
- 加上位置编码向量，在 v2.0 版本的 `sine_position_encoder_kernel` 中是直接计算了一个位置编码值加上去的，但是这里为了节省计算时间，直接通过查表实现，这就要求函数入参的时候把提前计算好的 `position_encoding` 传进来，这种做法其实挺好的，因为本身位置编码就是与数据无关的，完全可以提前算好，缺点就是会增加内存占用，用空间换时间。

### 4.7 Top-k 采样解码
前面说过 Top-k 采样解码是先选取概率最大的 `k` 个 word 再进行采样，所以需要先计算概率，计算概率必然要先根据 `logits` 值计算 Softmax，但是我们知道 `Softmax` 函数是单调的，其实就相当于一个指数映射后的归一化操作。那既然是单调函数，我们完全可以直接根据 `logits` 直接选出 TopK，然后再计算 Softmax，这样可以把 Softmax 的规模从 `vocab_size` 缩减到 `k`，这是一个非常可观的缩减量。

#### 4.7.1 update_logits_without_softmax
这个核函数完成了 `logits` 的 add bias 操作，其实是 decoder out 的线性变换的内容，前面只进行了矩阵乘法，在这个核函数中把偏置项加上，另外核函数内部还加了一个停止符判断。  
```cpp
template <typename T>
__global__ void update_logits_kernel_without_softmax(T* logits, const T* bias, const int end_id, const bool* finished, const int n)
{
  int bid = blockIdx.x;
  bool finish = finished[bid];
  int offset = bid * n;

  for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
  {
    if(finish)
      logits[offset + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
    else
      logits[offset + tid] += bias[tid];
  }
}

void update_logits_without_softmax(float* logits, const float* bias, const int end_id, const bool* finished, 
  const int m, const int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
  update_logits_kernel_without_softmax<float><<<grid, block, 0, stream>>>(logits, bias, end_id, finished, n);
}
```
这里加完偏置项就可以直接用于 TopK 采样了。

#### 4.7.2 topK_sampling_kernel_kernelLauncher
根据 `topK_sampling_kernel_kernelLauncher` 函数逻辑可以看出，采样过程由两个核函数完成：`beam_topK_kernel` 和 `sampling`。函数内部首先判断了  `candidate_num` 的值，貌似目前只支持 `1、2、4` 三种情况，这里源码为什么要用宏的模式，因为编译期要对模板进行实例化，要求 `K`(`candidate_num`) 在编译期就得确定，而源码中的 `candidate_num` 显然是一个运行期才确定的参数，所以只好牺牲编译期，多实例化几个模板（如 `1、2、4`，分别对应 1 个函数），等到运行期的时候匹配真实的 `candidate_num`，去执行对应的模板函数。  
```cpp
#define CASE_K(K) \
  case K : \
    beam_topK_kernel<T, K, block_size><<<batch_size, block_size, 0, stream>>>(log_probs, \
        topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, 0.0f); \
  break; \

template <typename T>
void topK_sampling_kernel_kernelLauncher(T* log_probs,
                                        int* topk_tmp_id_buf,
                                        T* topk_tmp_val_buf,
                                        int* ids,
                                        int* sequence_length,
                                        bool* finished_buf,
                                        int random_num,
                                        DecodingSamplingArguments args,
                                        cudaStream_t stream)
{
    const int batch_size = args.batch_size_;
    const int vocab_size = args.vocab_size_;
    const int candidate_num = args.candidate_num_;
    const int end_id = args.end_id_;
    const int block_size = 256;
    switch(candidate_num)
    {
        CASE_K(1);
        CASE_K(2);
        CASE_K(4);
        default:
            printf("[ERROR] Topk kernel does not support candidate_num = %d \n", candidate_num);
            exit(0);
            break;
    }
    sampling<T> <<< batch_size, candidate_num, 0, stream>>> (topk_tmp_id_buf, topk_tmp_val_buf, 
                                                            ids, sequence_length, finished_buf,
                                                            candidate_num, random_num, end_id, vocab_size);
}
```

##### 4.7.2.1 求TopK
在 v2.0 版本的 beam search 中也有求 TopK 的操作，不过当时那个计算思路就很粗糙，简单粗暴，总共分为两个 kernel，在第一个 kernel 里面，先是用块内规约求出当前线程对应值的最大值，把最大值存起来，然后变量赋值为极小值，然后线程内部直接循环 K 次，最后获得了 `grid_size` 个 TopK，然后再第二个 kernel 中把这个范围再缩小到 TopK。可以看到这是一种 native 的求 TopK 思路，在 v2.1 版本，求 TopK 的思路有所优化。  
TopK 问题是一个经典算法问题，通常我们通过维护一个小根堆，堆里存了 K 个数据，每次新数据跟堆顶数据比较，大于堆顶元素就替换掉堆顶元素，然后重新建堆，遍历完所有元素后，堆中元素就是 TopK。这里源码中也使用了这个思路，但是并没有使用堆结构，而是定义了一个结构体 `TopK`，应该是作者嫌麻烦，因为 K 实在太小，就不折腾了，我们来看一下这个结构体。
```cpp
template<typename T, int MAX_K>
struct TopK
{
    int p[MAX_K];
    T u[MAX_K];

    __device__ __forceinline__ void insert(T elem, int elem_id)
    {
        // 把插入元素跟最后一个元素比较，如果插入元素更大，则替换掉最后一个元素
        if (elem > u[MAX_K-1] || (p[MAX_K-1] == -1) || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        //if (elem > u[MAX_K-1] || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        {
            u[MAX_K-1] = elem;
            p[MAX_K-1] = elem_id;
        }
        // 冒泡排序，把 TopK 中的元素进行排序
        for(int k = MAX_K - 2; k >= 0; --k)
        {
            if ((u[k+1] > u[k]) || (p[k] == -1) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            //if ((u[k+1] > u[k]) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            {
                T u2 = u[k];
                int p2 = p[k]; 
                u[k] = u[k+1];
                p[k] = p[k+1];
                u[k+1] = u2;
                p[k+1] = p2;
            }
        }
    }

    __device__ __forceinline__ void init()
    {
      #pragma unroll
      for(int i = 0; i < MAX_K; i++)
      {
        p[i] = -1;
        u[i] = -FLT_MAX;
      }
    }
};
```
可以看到，结构体中有两个长度为 `MAX_K` 的数组变量，`p` 用来存索引，`u` 用来存值，一一对应并按值降序排列。为啥弄两个数组？是因为这里我们还需要元素的位置，也就是 `word_id`，这两个数组同步更新。除了成员变量以外还有两个成员函数，一个是初始化函数 `init` 主要用来初始化 `p` 和 `u`，另一个是 `insert` 函数用来“插入元素”和“建堆”。`insert` 函数中首先比较最后一个元素和新插入元素，满足以下任意条件后，将用新插入的元素替换掉 `TopK` 中最后一个元素。
- 插入元素大于最后一个元素
- 最后一个元素是初始化的标识，也就是数组没有满
- 插入元素等于最后一个元素，但是插入元素的索引更小

插入元素后，还得“建堆”保证堆顶元素最小，前面说过这里直接用排序代替“建堆”，所以源码就提供了一个冒泡排序，排序完成后，数组中的元素恢复降序排列。  
`TopK` 结构介绍完之后，下面就是如何使用 `TopK` 结构完成对 `logits` 的求 TopK 操作。源码中使用 `beam_topK_kernel` 核函数来求 TopK，`grid_size` 设置为 `batch_size`，`block_size` 设置为 `256`，也就是说一个 block 内要处理 `vocab_size` 个元素，从中选出 TopK，每个线程处理 `vocab_size / 256` 个元素，步长为 `256`。
```cpp
template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__
void beam_topK_kernel(const T* log_probs, 
                        int* topk_tmp_id_buf,
                        T* topk_tmp_val_buf,
                        const int vocab_size,
                        T diversity_rate)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    TopK<T, MAX_K> partial;
    
    #pragma unroll
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -FLT_MAX;
    }

    #pragma unroll
    for(int elem_id = thread_id; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE)
    {
        int index = elem_id + block_id * vocab_size;        
        partial.insert(log_probs[index], index);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0)
    {
        int index = block_id * MAX_K;
        
        #pragma unroll
        for(int i = 0; i < MAX_K; ++i)
        {
            topk_tmp_id_buf[index + i] = total.p[i];
            topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T)i;
        }
    }
}
```
核函数内部首先使用 cub 库进行了块内规约前的准备，这个我们暂且不去看，之后内部定义了一个寄存器变量 `partial`，`partial` 存储了当前线程处理元素的 TopK，相当于当前线程下的小根堆，随后对 `partial` 进行初始化，这块其实可以直接调用成员函数 `init` 的，但是作者估计忘记还有这个函数了就又手写了一遍。然后就是对当前线程待处理的元素进行遍历，让 `partial` 来 `insert` 待处理元素，全部 `insert` 一遍后的 `partial` 其实就存储了当前线程处理的所有元素的 TopK。但是我们的目标是要获取整个 block 内的全局 TopK，所以我们还需要进行一次“大合并”，把所有的 TopK 合并成一个，这实际相当于一次块内规约操作，只是我们还需要定义一个操作函数，显然这个操作函数的输入是两个 `TopK` 类型的变量，输出是 `TopK` 类型，其计算逻辑就是把两个 `TopK` 合并成 1 个 `TopK`。源码提供了一个 `reduce_topk_op` 函数来完成这个任务。  
```cpp
template<typename T, int MAX_K>
__device__ __forceinline__ TopK<T, MAX_K> reduce_topk_op(const TopK<T, MAX_K>& a, const TopK<T, MAX_K>& b)
{
    TopK<T, MAX_K> res = a;
    for(int i = 0; i < MAX_K; ++i)
        res.insert(b.u[i], b.p[i]);
    return res;
}
```
可以看到，`reduce_topk_op` 是通过遍历一个 `TopK` 变量 `b` 的元素，不断 `insert` 到另一个 `TopK` 变量 `a` 的拷贝 `res` 中实现的合并工作。  
有了操作函数以后，直接调用 cub 库的块内规约 API 就完成了块内规约，获取了整个 block 内的全局 TopK `total`。当 `thread_id == 0` 时，把这 `k` 个元素对应的 `logit` 和 `word_id` 写入 `topk_tmp_val_buf` 和 `topk_tmp_id_buf` 中。这里还有个 `diversity_rate` 参数，这应该是一个修正系数，但是笔者发现源码中实际设置为 `0.0f` 并没有启用。  

##### 4.7.2.2 采样
前面介绍过采样原理，获取 TopK 之后，计算每个 word 的概率，然后在 TopK 中归一化，最后根据归一化后的概率采样。其实就是先 Softmax 后采样，我们来看一下源码。
```cpp
// Sampling kernels
template<typename T>
__global__ void sampling(int* topk_tmp_id_buf, 
                        T* topk_tmp_val_buf, 
                        int* ids, 
                        int* sequence_length, 
                        bool* finished_buf,
                        const int candidate_num, 
                        int random_num,
                        const int end_id,
                        const int vocab_size)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ T sum;
    __shared__ T rand_num;

    if(tid < candidate_num)
    {
        T max_val = topk_tmp_val_buf[bid * candidate_num];
        topk_tmp_val_buf[bid * candidate_num + tid] = __expf(topk_tmp_val_buf[bid * candidate_num + tid] - max_val);
    }
    
    if(tid == 0)
    {
        sum = 0.0f;
        for(int i = 0; i < candidate_num; i++)
        {
            sum = sum + topk_tmp_val_buf[bid * candidate_num + i];
        }
        
        curandState_t local_state;
        curand_init((T)random_num, bid, 0, &local_state);
        rand_num = (T)curand_uniform(&local_state) * sum;

        ids[bid] = topk_tmp_id_buf[bid * candidate_num + candidate_num - 1] % vocab_size;
        for(int i = 0; i < candidate_num; i++)
        {
            rand_num = rand_num - topk_tmp_val_buf[bid * candidate_num + i];
            if(rand_num <= 0.0f){
                ids[bid] = topk_tmp_id_buf[bid * candidate_num + i] % vocab_size;
                break;
            }
        }

        sequence_length[bid] = finished_buf[bid] ? sequence_length[bid] : sequence_length[bid] + 1;
        finished_buf[bid] = ids[bid] == end_id ? 1 : 0;
    }
}
```
核函数中 `grid_size` 和 `block_size` 分别设置为 `batch_size` 和 `candidate_num`，当前线程就只处理对应一个元素，先根据索引从 `topk_tmp_val_buf` 中获取 TopK 中的最大值，然后让当前元素减去最大值然后求指数，再存入 `topk_tmp_val_buf`。在 0 号线程内循环求规约和，得到 `sum`，这时候其实已经可以开始采样了，没有必要非得归一化。源码中调用 cuda 随机数生成库的 API 从均匀分布中随机一个 `0~1` 之间的数再乘以 `sum`，得到一个 `0~sum` 之间的数 `rand_num`，要知道 TopK 中各元素是降序排列的，我可以把他当成 k 个相互连接的组合线段记作 $S_t$（其中每个子线段记作 $S_i$），把 `rand_num` 当成一根长度为 `rand_num` 的线段记作 $S_r$，并将其与 $S_t$ 的最左侧对齐，那么 $S_r$ 的右端点落在 $S_t$ 的哪个子线段中就认为采样选中了哪个 word，笔者给出如下示意图。  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qKvGhrwW5bOI1eM5TrQ7XNZgacjbkJfrzDZC6ajPavay5PEZVmupvQWcJ2DW7ic3IcMtD3ZEd0n6A/640?wx_fmt=png)

随后根据采样选中的 `word_id` 对 `sequence_length` 和 `finished_buf` 进行更新，至此当前 step 的采样解码就完成了。  

### 4.8 Top-p 采样解码
Top-p 采样与 Top_k 采样有所不同，不再从固定 k 个候选词中采样，而是根据生成概率从高到低在词表上选择累积概率恰好超过 $p$ 的候选 word 作为采样集合，从这个集合中采样。所以采样前必须先计算每个 word 对应的概率并进行排序，即要计算 Softmax，再按概率值排序。

#### 4.8.1 update_logits_without_log
源码中使用 `update_logits_kernel_without_log` 核函数来计算 Softmax，顺带加一个上一步没进行的 add bias 操作。这个核函数比较简单是个老生常谈的 Softmax kernel，只需要注意一点，计算完 softmax 后不要取对数即可，具体计算逻辑笔者就不啰嗦了，读者有兴趣可以看笔者前面的文章。  

#### 4.8.2 topP_sampling_kernel_kernelLauncher
Softmax 后拿到词表内每个 word 的概率，在进行采样前还要进行排序。
##### 4.8.2.1 排序
这里排序是个大工程，因为 `vocab_size` 通常会很大，源码中使用了 cub 库中的 API 进行排序。  
```cpp
template<typename T>
void topP_sampling_kernel_kernelLauncher(const T* log_probs,
                                        const int* id_vals,
                                        T* sorted_log_probs,
                                        int* sorted_id_vals, 
                                        int* topp_offset_buf,
                                        void* temp_storage,
                                        bool* finished_buf,
                                        int step,
                                        DecodingSamplingArguments args,
                                        int* output_ids, 
                                        int* sequence_length, 
                                        cudaStream_t stream)
{
    // sort_kernel<<<batch_size, 256, 0, stream>>>(log_probs, 
    //                                             id_vals,
    //                                             sorted_log_probs,
    //                                             sorted_id_vals,
    //                                             vocab_size);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(temp_storage, 
                                                        args.temp_storage_size_,
                                                        log_probs, 
                                                        sorted_log_probs,
                                                        id_vals, 
                                                        sorted_id_vals, 
                                                        args.vocab_size_ * args.batch_size_,
                                                        args.batch_size_, 
                                                        topp_offset_buf, topp_offset_buf + 1);
                                                        
    top_p_sampling<<<1, args.batch_size_, 0, stream>>>(sorted_log_probs, 
                                                        sorted_id_vals,
                                                        output_ids + (step - 1) * args.batch_size_,
                                                        sequence_length,
                                                        finished_buf,
                                                        args.vocab_size_, 
                                                        step,
                                                        args.probability_threshold_,
                                                        args.end_id_);
}

```
下面我们对 `cub::DeviceSegmentedRadixSort::SortPairsDescending` 函数的主要参数进行介绍：  
- `d_temp_storage`：设备可以访问的临时内存，当设置为 `NULL` 时，所需的分配大小将写入 `temp_storage_bytes`，并且不执行任何工作。所以在真正执行函数前，我们需要先传一下 NULL 获取 `temp_storage_bytes` 然后再开始真正的执行排序
- `temp_storage_bytes`：临时内存的大小
- `d_keys_in`：指向排序过程中的比较依据，也就是说排序是根据这个指针指向的数据的来进行的，这里我们将它设置为概率值 `log_probs`
- `d_keys_out`：排序后的输出，这里我们用 `sorted_log_probs` 来接收
- `d_values_in`：与 key 一一对应，这里我们把他设置为概率值对应的索引 `id_vals`，其实就是 `word_id`
- `d_values_out`：排序后的输出，这里我们用 `sorted_id_vals` 来接收
- `num_items`：待排序的元素数目，这里应该是 `batch_size * vocab_size`
- `num_segments`：待排序的批次，也就是分为多少个组，这里是对每个样本单独排序，所以取 `batch_size`
- `d_begin_offsets`：每个分组的起始索引，为了方便 `end_offset` 的设置，这个变量对应的元素数量通常是 `num_segments + 1`，前面 `num_segments` 个元素都是分组的起始索引，最后一个元素设为 `num_items`，这里我们设置为 `topp_offset_buf`，前面已经完成初始化
- `d_end_offsets`：每个分组的结束索引，注意这里是“顾头不顾尾”的模式，所以直接可以设置为 `d_begin_offsets + 1`，这里我们设置为 `topp_offset_buf + 1`

参数意义介绍完毕后，其实函数的作用也就清晰了，就是分组降序排序，每一组对应 `batch` 内的一个样本，也就是 `vocab_size` 个元素，最终我们获取到了 batch 内每个样本下排序后的待采样 word 的概率值 `sorted_log_probs`和 `sorted_id_vals`。   

##### 4.8.2.2 采样
根据采样原理，拿到排序结果后，我们需要根据 p 值进行候选集的确定，然后在候选集的内部进行采样。  
源码中提供了核函数 `top_p_sampling` 进行采样工作，`grid_size` 设置为 `1`，`block_size` 设置为 `bacth_size`，即在一个 block 内完成计算，每个线程承担一个样本的计算任务。  
```cpp
template<typename T>
__global__ void top_p_sampling(T* sorted_log_probs, 
                                int* sorted_id_vals,
                                int* ids,
                                int* sequence_length,
                                bool* finished_buf,
                                const int vocab_size,
                                const int random_num,
                                const float prob_threshold, 
                                const int end_id)
{
    int tid = threadIdx.x;
    curandState_t local_state;
    curand_init((T)random_num, tid, 0, &local_state);
    T rand_num = (T)curand_uniform(&local_state) * prob_threshold;
    ids[tid] = sorted_id_vals[vocab_size - 1];

    for(int i = tid * vocab_size; i < tid * vocab_size + vocab_size; i++)
    {
        rand_num = rand_num - sorted_log_probs[i];
        if(rand_num <= 0)
        {
            ids[tid] = sorted_id_vals[i];
            break;
        }
    }

    sequence_length[tid] = finished_buf[tid] ? sequence_length[tid] : sequence_length[tid] + 1;
    finished_buf[tid] = ids[tid] == end_id ? 1 : 0;
}

top_p_sampling<<<1, args.batch_size_, 0, stream>>>(sorted_log_probs, 
                                                  sorted_id_vals,
                                                  output_ids + (step - 1) * args.batch_size_,
                                                  sequence_length,
                                                  finished_buf,
                                                  args.vocab_size_, 
                                                  step,
                                                  args.probability_threshold_,
                                                  args.end_id_);
```
采样过程和前面 Top-k 的过程大同小异，有一点区别就是，不用真的先确定候选集再进行采样，可以直接一步进行。先使用 cuda 随机数生成库的 API 从均匀分布中随机一个 `0~1` 之间的数再乘以 p 值（`probability_threshold_`），这其实就相当于把采样的概率点缩放到了 p 值范围内，然后遍历 `sorted_log_probs` 判断采样点落在哪个区间，就选中了哪个 word，示意图如下：  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rOU8ibicb4mGqGvrH9r0pm9fIgRJ44aurWP5bEmSedysRc2ZDsdSVoPYtEEA8c8gJX6UicBZw7gZticA/640?wx_fmt=png)

采样完成后把采样结果更新到 `ids`，然后对 `sequence_length` 和 `finished_buf` 进行更新，至此，当前 step 的 Top-p 采样解码就完成了。  

## 5 内核优化
### 5.1 批量矩阵乘法优化
首先我们来看 Encoder 部分的优化点，Encoder 的计算较为简单，主要集中在 self-attention。在介绍 `OpenMultiHeadAttention` 之前我们不妨先来看一下其内部 buffer 的内存分布情况，通过内存分布情况的变化，可以看出具体新增和删减了哪些变量，从这些变量入手可以有助于弄懂具体优化逻辑。   
不妨先来看一下 v2.0 版本的 `OpenMultiHeadAttention` 的内存分布示意图。
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5oND3ric2UY8P38xq5PppqMZPTjhD0OqNVqks2wVh4z81HTQZHgPCZPvgcEvbQLWkIR3xcmgibe9pZQ/640?wx_fmt=png)
再看一下 v2.1 版本的构造函数中的内存分配逻辑，基于代码逻辑笔者给出内存分布示意图如下。
```cpp
buf_ = (DataType_*) allocator_.malloc(sizeof(DataType_) * (buf_size * 7 + qk_buf_size) + sizeof(DataType_*) * 9);
query_buf_ = buf_;
key_buf_ = buf_ + buf_size;
value_buf_ = buf_ + 2 * buf_size;
q_buf_ = buf_ + 3 * buf_size;
k_buf_ = buf_ + 4 * buf_size;
v_buf_ = buf_ + 5 * buf_size;
qk_buf_ = buf_ + 6 * buf_size;
transpose_dst_ = qk_buf_ + qk_buf_size;
qkv_kernel_ = (DataType_**)(transpose_dst_ + buf_size);
qkv_input_ = qkv_kernel_ + 3;
qkv_buf_ = qkv_input_ + 3;
```
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5oND3ric2UY8P38xq5PppqMZibELAbVEOqsr72g6AZd7zD4oYYtt4yoacE7jeMfj9FibVN2ZKLeARrkg/640?wx_fmt=png)
从 `OpenMultiHeadAttention` 构造函数中可以发现，v2.1 版本多申请了 `sizeof(DataType_*) * 9` 大小的设备内存，也就是说多了 `9` 个二级指针。从下面内存分配可以看出，这 `9` 个二级指针分别分给了 `3` 个变量：`qkv_kernel_`、`qkv_input_`、`qkv_buf_`。这三个变量在 `initialize` 进行初始化，其中 `qkv_kernel_` 对应 self-attention 操作中对输入 tensor 进行线性变换的 3 个权重参数；`qkv_input_` 对应 3 个输入 tensor，在 self-attention 中全部都是 `from_tenosr`；`qkv_buf_` 对应的是 3 个 buffer，用于存储 Q\K\V 的中间计算结果。  

```cpp
const DataType_* hA[] {param_.self_attention.query_weight.kernel, 
                            param_.self_attention.key_weight.kernel, 
                            param_.self_attention.value_weight.kernel,
                            param_.from_tensor, param_.from_tensor, param_.from_tensor,
                            query_buf_, key_buf_, value_buf_};
cudaMemcpyAsync((void*)qkv_kernel_, hA, sizeof(DataType_*) * 9, cudaMemcpyHostToDevice, param_.stream);
```
仔细一想，这三个新增变量最终指向的数据其实都是前面已经存在的变量，为什么要单独搞这几个二级指针呢，而且这几个变量统统都和 self-attention 相关？这应该是 self-attention 中使用了某个 API，通过这个 API 可以获得加速效果，其输入参数要求是二级指针。  
果不其然，`forward` 函数中当 `is_fuse_QKV` 为 `true` 时，调用了 cuBLAS 中的 `cublasGemmBatchedEx` 函数进行矩阵乘法，该函数可以对 batch 级别的矩阵进行乘法运算，要求输入的参数为 `Array of pointers to <Btype> array` 也就是二级指针的形式，这里源码把 query、key、value 三个矩阵当成一个 batch 内的三个矩阵，使用 API 一次完成 `3` 个矩阵的乘法运算，相比与原来的先后调用三次 `cublasGemmEx` 函数计算乘法，节省了一定的运算时间。  
```cpp
if(is_fuse_QKV == true)
{
  check_cuda_error(cublasGemmBatchedEx(param_.cublas_handle, 
                      CUBLAS_OP_N, CUBLAS_OP_N, 
                      n, m, k, 
                      &alpha, 
                      (const void* const*) qkv_kernel_, AType_, n,
                      (const void* const*) qkv_input_, BType_, k,
                      &beta,
                      (void* const*)qkv_buf_, CType_, n,
                      3, 
                      computeType_,
                      static_cast<cublasGemmAlgo_t>(cublasAlgo_[3])));
}
```
关于批量矩阵乘法这个优化除了 Encoder 以外，在 Decoder 的 self-attention 中也有应用，具体各位读者可以自行阅读。

### 5.2 Decoder Attention Opt
Decoder 的两个核函数 `masked_attention_kernel_opt`、 `cross_attention_kernel_opt` 的优化是 decoder 中的主要优化内容，笔者将以 `masked_attention_kernel_opt` 为例介绍优化技巧，关于 attention 的原理等内容不再赘述。这部分代码实现过程说实话有些繁琐，会导致初读的时候一脸懵逼，总之优化思路就一句话：**向量化数据访问提升带宽**。  
#### 5.2.1 向量化数据类型
首先作者定义了一个数据类型 `Copy_t`，这个类型的定义过程也比较繁琐，其内存占用的大小是根据 `ELEMENTS_PER_WARP_LOAD` 动态调整的，具体代码如下：  
```cpp
template <int HALF_ELEMENTS_PER_WARP_LOAD>
using Copy_half_t =
    typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 32, half,
        typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 64, int,
            typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 128, int2, int4
            >::type
        >::type
    >::type;

template <typename T, int ELEMENTS_PER_WARP_LOAD>
using Copy_t = Copy_half_t<sizeof(T) / sizeof(half) * ELEMENTS_PER_WARP_LOAD>;
```
源码中使用的 `std::conditional` 是 C++11 引入的类模板，表示的是一种编译期的分支逻辑，当第一个非类型模板参数的值为 `true` 时，`type` 的类型为第一个类型模板参数的类型，为 `false` 时 `type` 的类型为第二个类型模板参数的值。那么以上代码的含义就是在一个 warp 内处理 `ELEMENTS_PER_WARP_LOAD` 个 `T` 类型的元素，`Copy_t` 类型占用的大小为 `sizeof(T) * ELEMENTS_PER_WARP_LOAD / 32`。  
这里我们假设数据类型 `T` 以 FP32 为例，`ELEMENTS_PER_WARP_LOAD` 设置为 `size_per_head` 取 `64`，这样的话 `Copy_t` 实际就是 `int2` 类型，不要去纠结为什么是 `int2`，这里写 `int2` 仅仅是因为他占了 `8` 个字节，写 `float2` 等任意占用 `8` 个字节的类型也是一样的。带着这个向量化访问的思想，我们再来看一下核函数 `masked_attention_kernel_opt` 的代码，代码太繁琐我就不完整贴了，下面只对主要内容进行介绍。  
```cpp
typedef Copy_t<T, size_per_head> copy_t;
const int elems_per_thread = size_per_head / WARP_SIZE;

union Access_t
{
  copy_t v;
  T x[elems_per_thread]; // supported size 1,2,4
};
typedef struct Float_n_t
{
  T x[elems_per_thread]; // supported size 1,2,4
} float_n_t;
```
首先提一下核函数的 `gird_size` 和 `block_size` 分别设置为 `batch_size * head_num` 和 `256`，也就是一个 block 内处理一行数据（`size_per_head`个元素）。在核函数内部定义了一个类型 `copy_t`，从模板参数可以看出，这里是想要一个 Warp 内部直接处理 `size_per_head` 个元素的，也就是说在这一个 block 内一个 warp 就完成了当前 step 的计算任务，其他 warp 在干嘛？后面将会讲到。然后定义了一个变量 `elems_per_thread` 表示每个线程处理的元素数量。接着定义了一个联合体 `Access_t` 用来存储 `elems_per_thread` 个 `T` 类型的元素，和一个结构体 `Float_n_t` 用来存储 `n` 个 `T` 类型的元素。  
#### 5.2.2 add query bias
核函数内定义了两个变量 `sq` 和 `logits`，用来存储 attention 的中间结果。  
```cpp
__shared__ float_n_t sq[block_sz];
__shared__ float logits[1024]; // only use [0 ~ step-1], the step should be smaller than 1024
```
在 add query bias 之前先计算了当前线程的各类索引以及偏移量 `qkv_id`，结合线程网格这个很好理解，然后根据偏移量更改了 Q\K\V 相关的各个变量的地址方便后续索引。计算 add query bias 的过程分为两步：从 `query_buf` 和 `self_Q_bias` 中向量化取值、结构体内循环计算。  
```cpp
// each warp will have its own copy of sq
query_buf_r.v = *((copy_t *)query_buf + lane_id);
key_buf_r.v = *((copy_t *)key_buf + lane_id);
bias_r.v = *((copy_t *)self_Q_bias + lane_id);
float qb_r[elems_per_thread];
for (int i = 0; i < elems_per_thread; ++i)
{
  qb_r[i] =  (float)query_buf_r.x[i] + (float)bias_r.x[i];
}
```
可以看到联合体 `Access_t` 中的成员 `v`，其实就起到一个方便占位的作用，当然如果没有的话，笔者认为也可以使用下面的方式取值。    
```cpp
Access_t *qbuf = reinterpret_cast<Access_t>(query_buf)
query_buf_r.v = qbuf[lane_id];
```

#### 5.2.3 add key bias & softmax
我们知道 attention 中 softmax 计算的对象是 query 和 key 的乘积，query 我们已经拿到了，存在每个 thread 的 `qb_r` 中。key 需要从 `key_cache` 中获取，对于当前 step 而言，query 是固定的，与 `from_tensor` 对应，key 与前面 step 的 `from_tensor` 也一一对应，因此这一步完全是可以并行的，所以作者在这里设计成一个 block 内总共处理 `warp_num` 个 step 的计算，这也回应了前面的疑问，明明一个 warp 内就能处理一个 step 的计算，其他 warp 干啥去了。其他 warp 处理其他 step 的计算去了。总结一下，对于所有 warp 而言 query 都是一样的，所以放在寄存器变量 `qb_r` 中，同时每个 warp 有各自的 key，通过 `ite * offset` 计算偏移量获取。   
```cpp
//offset for each step
int offset = batch_size * head_num * size_per_head;
bias_r.v = *((copy_t *) self_K_bias + lane_id);
for(int ite = warp_id; ite < step; ite += warp_num)
{
  key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);
  //for the last step, we should update K + bias_K to the cache
  if(ite == step - 1)
  {
    for (int i = 0; i < elems_per_thread; i++)
    {
      key_val_r.x[i] = (float)key_buf_r.x[i] + (float)bias_r.x[i];
    }
    *((copy_t *)&key_cache[ite * offset] + lane_id) = key_val_r.v;
  }
  float val = 0.f;
  for (int i = 0; i < elems_per_thread; i++)
  {
    val = val +  (float)key_val_r.x[i] * qb_r[i] * (float)scalar;
  }
  float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
  if (lane_id == 0)
  {
    logits[ite] = qk; 
  }
}
__syncthreads();
```
拿到 key 之后将其与 query 相乘然后调用 cub 库的束内规约 API 计算出每个 step 下 query 和 key 的向量相似度也就是 attention scores，根据 step 的索引 `ite` 将其存入 `logits` 中。  
```cpp
__shared__ float s_max_val, s_sum;

float local_i = -1e20f;
for(int i = tid; i < step; i += blockDim.x)
  local_i = max(local_i, logits[i]);

float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
if(tid == 0)
  s_max_val = max_val;
__syncthreads();


float local_o = 0.0f;
for(int i = tid; i < step; i += blockDim.x)
{
  logits[i] = __expf(logits[i] - s_max_val);
  local_o += logits[i];
}
float val = BlockReduce(block_temp_storage).Sum(local_o);

if(tid == 0)
  s_sum = val + 1e-6;
__syncthreads();

float s_sum_inverse = __fdividef(1.0f, s_sum);
for(int i = tid; i < step; i += blockDim.x)
{
  logits[i] = logits[i] * s_sum_inverse;
}
__syncthreads(); 
```
计算 Softmax 的过程比较常规，就是三个步骤：reduceMax、broadcast、reduceSum、broadcast，没什么好说的，有疑问的读者可以阅读笔者上一篇文章。  

#### 5.2.4 计算 attention
得到 attention scores 后，右乘一个 value 矩阵就得到 attention out，其实就是算加权平均值。这里计算思路也和前面计算 $QK^T$ 大致相同，都是一个 block 内计算 `warp_num` 个 step，计算完成后 `sum_r` 中存储了每个线程负责的 step 的对应元素的加权和，最终的是需要所有 step 的加权总和，所以作者把 `sum_r` 放入共享内存变量 `sq` 中暂存，然后再进行两层循环把其他线程束计算的 `sum_r` 全都加到 `warp_id == 0` 的线程对应的 `sum_r` 中，此时 `warp_id == 0` 的线程束内各 thread 中的 `sum_r` 存储的即为完成加权求和之后的 attention out，最后将 attention out 更新到 `context_buf_ptr` 中完成计算。

```cpp
// This optimization introduces discrepancy because of different order in FP32 summation
float sum_r[elems_per_thread] = {0.f};
bias_r.v = *((copy_t *) self_V_bias + lane_id);
value_buf_r.v = *((copy_t *)value_buf + lane_id);

for(int ite = warp_id; ite < step; ite += warp_num)
{
    value_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
        for (int i = 0; i < elems_per_thread; i++)
        {
            value_val_r.x[i] = (float)value_buf_r.x[i] + (float)bias_r.x[i];
        }
        *((copy_t *)&value_cache[ite * offset] + lane_id) = value_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i)
    {
        sum_r[i] += (float)value_val_r.x[i] * logits[ite]; 
    }
}
for (int i = 0; i < elems_per_thread; i++)
{
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
}
__syncthreads();
if (warp_id == 0)
{
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
        for (int i = 0; i < elems_per_thread; ++i)
        {
            sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + tid].x[i];
        }
    }
}
__syncthreads();
#pragma unroll
for (int i = 0; i < elems_per_thread; i++)
{
    value_val_r.x[i] = sum_r[i];
}
if (warp_id == 0)
{
    *((copy_t *)context_buf + lane_id) = value_val_r.v;
}
```

这里有一点需要注意，在把其他线程束计算的 `sum_r` 更新到 0 号线程束内时，源码中使用 `sq[j * WARP_SIZE + tid]` 进行取值，这里容易造成误解，虽然在 0 号线程束内 `lane_id` 和 `tid` 的值相等，但是为了便于理解这里建议还是使用 `sq[j * WARP_SIZE + lane_id]` 取值较好，以免对不熟悉的读者造成困扰。  

### 5.3 topK kernel 优化

关于 Top-k 采样解码前面已经介绍，这里说的 topK 特指 beam search 过程中的求 topK 操作，在 v2.0 版本中，固定设置 `block_size` 为 `1024`，通过第一个 topK kernel 把 `ids` 的形状缩小到 `[batch_size, grid_size_1st, beam_width]`，再经过第二个 topK kernel 求出最终每个样本的 topK。其中关于 `batch_size` 维度的每个样本的计算过程是通过循环实现的，并行程度不高。在 v2.1 版本，作者更新了 topK kernel，依然通过两个 kernel （`topk_stage_1_opt3` 和 `topk_stage_2_opt3`）完成 topK 计算。

在 `topk_stage_1_opt3` 中把 `gird_size` 设置为 `batch_size * K * BLOCKS_PER_BEAM_`，也就是说对于每一行 `vocab_size` 个元素，要使用 `BLOCKS_PER_BEAM_` 个 block 参与计算。

```cpp
/**
 * @brief 
 * // grid_size = batch_size * K * BLOCKS_PER_BEAM_
 * @tparam T 
 * @tparam BLOCK_SIZE_ 
 * @tparam BLOCKS_PER_BEAM_ 
 * @param log_probs                 [batch_size, beam_width, vocab_size]
 * @param tmp_log_probs             [batch_size, beam_width, vocab_size]
 * @param topk_tmp_id_buf           [batch_size, beam_width, BLOCKS_PER_BEAM_, K]
 * @param topk_tmp_val_buf          [batch_size, beam_width, BLOCKS_PER_BEAM_, K]
 * @param k                          beam_width
 * @param vocab_size 
 * @return __global__ 
 */
template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_1_opt3(
    const T* __restrict log_probs,
    T* tmp_log_probs,
    int* topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    const int k,    // beam_width
    const int vocab_size
)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row_id = bid / BLOCKS_PER_BEAM_; // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM_; // block id for a beam 
    const int tmp_log_buf_index = row_id * vocab_size; 
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
    TopK_2<T> partial;

    for(int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        int index = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index]; 
    }


    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
        {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            const int index = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index] = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p] = -FLT_MAX;
        }
        __syncthreads();
    }
}
```

核函数内引入了一个新的数据结构 `TopK_2`，这个数据结构中有两个成员属性，`p` 和 `u`，分别代表存储了概率值和其对应的 `word_id`；有两个成员方法，`init` 和 `insert`，分别进行初始化和更新，`insert` 方法非常简单，就是单纯的把更大的值和索引更新到对象中。接下来，我们看一下 `topk_stage_1_opt3`  函数。

首先计算了当前线程对应 `log_probs` 的各种索引，这个根据线程网格不难理解，根据索引将 `log_probs` 的值更新到 `tmp_log_probs` 中，注意这里每个线程处理元素的步长为 `BLOCK_SIZE_ * BLOCKS_PER_BEAM_`。

随后对 K 进行循环，在循环中首先对该线程处理的所有元素进行遍历，不断将数据 `insert` 到 `partial` 中，这样就得到了每个线程处理元素的最大值，然后再对 `partial` 进行块内规约，得到每个线程块内的最大值 `total`。在 `tid == 0` 的线程内把 `total` 更新到 `topk_tmp_id_buf`，再把 `tmp_log_probs` 中的值置为极小值，循环 K 次上述过程就得到每个线程块内的 topK，最终一行元素被处理成了 `BLOCKS_PER_BEAM_` 个 topK，`topk_tmp_val_buf` 的形状为 `[batch_size, beam_width, BLOCKS_PER_BEAM_, K]`。在第二个 kernel 中我们需要将其缩小到 `[batch_size, K]`，下面来看一下代码。

```cpp
/**
 * @brief 
 * // grid_size = batch_size
 * @tparam T 
 * @tparam BLOCK_SIZE_ 
 * @tparam BLOCKS_PER_BEAM_ 
 * @param topk_tmp_id_buf               [batch_size, beam_width, BLOCKS_PER_BEAM_, K]
 * @param topk_tmp_val_buf 
 * @param ids 
 * @param k 
 * @return __global__ 
 */
template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_2_opt3(
    const int* __restrict topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    int* ids,
    const int k)
{
    const int size = k * k * BLOCKS_PER_BEAM_; 
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;

    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T *s_val = topk_tmp_val_buf + batch_id * size;
    int *s_id = (int*)(array);
    
    TopK_2<T> partial;

    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int i = tid; i < size; i+= BLOCK_SIZE_)
        {
            partial.insert(s_val[i], i);
        }
    
        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);
    
        if(tid == 0) 
        {
            s_id[ite] = total.p;
            s_val[total.p] = -FLT_MAX;
        }
        __syncthreads();
    }
    if(tid < k) ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
}
```

`topk_stage_2_opt3` 的 `grid_size` 直接设置为 `batch_size`，`block_size` 设置为 `BLOCK_SIZE_`，也就是说，我们在一个 block 内求出 topK 就可以完成计算任务。计算思路和 `topk_stage_2_opt3` 大致相同，都是每次求最大值后把原值置小，循环 K 次即可。定义了一个共享内存变量 `s_id` 用来存储 topK，最终在 `tid < k` 的线程分别把 `s_id` 更新到 `ids` 完成计算。  
总的来说，更新后的 topK kernel 计算思路更加清晰，便于理解，是一个较好的思路，但是笔者还是更推荐使用 Top-k 采样解码中的思路来计算 topK 问题，猜测这两种解码方式的代码不是同一个作者编写的，否则完全可以复用代码。  

## 6 小结
总的来说，v2.1 版本的 Faster Transformer 相比与 v2.0 版本细节改动还是比较多的，但是整体大框架没有改变，仍然还是 3 个主要模块：Encoder、Decoder、Decoding，新增了 Effective Transoformer 和 sample Decoding 两个子模块。现对本文总结如下：  
- 新增了 Effective Transoformer，通过动态删除和恢复填充值的方式一定程度上可以节约 Encoder 部分的计算资源，当一个 batch 内样本长度变化越大，性能提升越明显。在实际应用中，训练模型阶段，其实在处理数据时一般会刻意地将一个 batch 的文本长度控制在一个较小的变化范围，但在推理阶段通常不会这么干，所以这个 Effective Transoformer 就有用武之地了。  
- 在 Top-k 采样中，源码给出了一个很好的求 topK 的思路，值得学习借鉴。
- 在 Top-p 采样中，源码示范了如何调用 cub 库 API 进行分组排序，值得学习借鉴。
- 在计算 self-attention 过程中，源码示范了 cuBLAS 库的 `cublasGemmBatchedEx` 函数调用方法，将三次串行调用矩阵乘法 API 缩减为 1 次。
- 在 Decoder 中，源码首次引入向量化数据类型，提升访存效率。

