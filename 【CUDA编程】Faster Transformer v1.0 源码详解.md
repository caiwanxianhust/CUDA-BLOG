#! https://zhuanlan.zhihu.com/p/647012855

# 【CUDA编程】Faster Transformer v1.0 源码详解
**写在前面**：本文将对 Nvidia BERT 推理解决方案 Faster Transformer 源码进行深度剖析，详细分析作者的优化意图，并对源码中的加速技巧进行介绍，希望对读者有所帮助。本文源码解读的内容仅限 Faster Transformer v1.0 版本，更高版本的源码将在后续文章中继续解读。

## 1 Faster Transformer
Transformer 模型最早在 2017 年由谷歌在论文中提出，抛弃了以往深度学习任务里面使用到的 CNN 和 RNN，取而代之的是一种 self-Attention 的结构，将 Attention 思想发挥到了极致，一定程度上解决了 RNN 的长序列信息丢失问题，基本取代了 RNN 的地位。  
Transformer 一经面世便在 NLP 领域脱颖而出，近两年一些文章开创性地将 Transformer 模型跨领域地引用到了 CV 任务中，并取得了不错地成果。这也被许多学者认为是开创了 CV 领域的新时代，甚至可能完全取代传统的卷积操作。   
虽然 Transformer在多种场景下都有优秀的表现，但是在推理部署阶段，其计算性能却受到了巨大的挑战：以 BERT 为原型的多层 Transformer 模型，其性能常常难以满足在线业务对于低延迟和高吞吐的要求。以 BERT-BASE 为例，超过 90% 的计算时间消耗在 12 层 Transformer 的前向计算上。因此，一个高效的 Transformer 前向计算方案，既可以为在线业务带来降本增效的作用，也有利于以 Transformer 结构为核心的各类网络在更多实际工业场景中落地。  
基于上述背景，NVIDIA GPU 计算专家团队针对 Transformer 推理提出了的性能优化方案：Faster Transformer。  
Faster Transformer 是一个 BERT Transformer 单层前向计算的高效实现，其代码简洁明了，后续可以通过简单修改支持多种 Transformer 结构。目前优化集中在编码器（encoder）的前向计算。底层由 CUDA 和 cuBLAS 实现，支持 FP16 和 FP32 两种计算模式，其中 FP16 可以充分利用 Volta 和 Turing 架构 GPU 上的 Tensor Core 计算单元。

## 2 优化原理
在深入了解 Faster Transformer 的优化原理之前，我们先来了解一下主流深度学习框架 Tensorflow 中 Transformer 的实现情况，仅仅以一个基本的激活函数 `gelu` 为例，这个函数在框架中是通过 8 个类似 Pow、Add、和 Tanh 等基本 OP 来实现的。也就是说每进行一次 `gelu` 运算要调用 8 次基本 OP，同时底层也对应 8 次 GPU kernel 的调用，每一次调用也意味着一次显存读写，先不说 kernel 计算耗时，光是显存读写就已经是大量的开销。如何降低这部分开销？最直观的方法就是减少调用，让数据一直留在显存甚至寄存器里被访问，即 OP 融合，一次调用就实现整个计算逻辑。  
出于性能最大化的考虑，在 Faster Transformer 内部，Nividia 将除矩阵乘法以外的所有 kernel 都进行了尽可能的融合，单层 Transformer 的计算流程如下图所示:  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5pZzkl4Iq3sJsMK3pNophMX8HoV8CHBwClsjvz9ps0haC9esptu8eOia6ZOujTnYjAYsicrdx9KhkicA/640?wx_fmt=png")

如图所示，基于 OP 融合的思想，Faster Transformer 只用了 14 个 kernel 就完成了原来将近 60 个 kernel 的计算逻辑。这其中，8 个 kernel 是通过调用 cuBLAS 接口计算矩阵乘法（黄色框），其余 6 个是自定义 kernel（蓝色框）。  
接下来笔者将沿调用链逐步介绍每个 kernel 的优化逻辑。

## 3 调用链
Faster Transformer v1.0 版本源码地址如下，有兴趣的读者可以前往阅读。  

>https://github.com/NVIDIA/FasterTransformer/tree/v1.0/fastertransformer  

通读源码后笔者对调用关系梳理如下。
```cpp
BertEncoderTransformer->forward()
    ->OpenMultiHeadAttention->forward()
        ->cublasGemmEx
        ->cublasGemmEx
        ->cublasGemmEx
        ->multiHeadAttr_nofuse_kernelLauncher
            ->add_QKV_bias  (kernel)
            ->cublasGemmStridedBatchedEx
            ->softmax_kernel    (kernel)
            ->cublasGemmStridedBatchedEx
            ->transpose (kernel)
    ->cublasGemmEx
    ->add_bias_input_layernorm_kernelLauncher   (kernel)
    ->cublasGemmEx
    ->add_bias_act_kernelLauncher   (kernel)
    ->cublasGemmEx
    ->add_bias_input_layernorm_kernelLauncher   (kernel)
```
从调用链也可以看出，总共 14 个步骤，与示意图一一对应。核心逻辑都在两个类中实现：`BertEncoderTransformer` 和 `OpenMultiHeadAttention`。  

## 4 OpenMultiHeadAttention
`OpenMultiHeadAttention` 类中有两个重要的成员方法：构造函数、`forward` 方法。其中构造函数内主要进行一些参数初始化功能，设备内存的申请和初始化也在该函数内进行。`forward` 方法内主要是多头注意力机制核心逻辑的具体实现。
### 4.1 cublasGemmEx for Q、K、V
`forward` 方法中首先就是对输入的 3 个 tensor 进行线性变换，其实就是对 3 个 tensor 分别进行 Dense 层变换，我们知道 Dense 是包含一个矩阵乘法和一个 add_bias 操作，这里只进行矩阵乘法，add_bias 操作放在后面的 kernel 进行。这里使用了 cuBLAS 接口计算矩阵乘法，具体代码如下：
```cuda
check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.attr_kernel_Q, AType_, n, 
    param_.from_tensor, BType_, k, 
    &beta, 
    query_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k, 
    &alpha, 
    param_.attr_kernel_K, AType_, n, 
    param_.to_tensor, BType_, k, 
    &beta, 
    key_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k,
    &alpha,
    param_.attr_kernel_V, AType_, n, 
    param_.to_tensor, BType_, k, 
    &beta, 
    value_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
```
这里仅仅是矩阵乘法 API 的调用，按文档传参即可，这里不展开介绍，笔者计划另开一篇文章专门介绍这个 API 的调用方法。

### 4.2 multiHeadAttr_nofuse_kernelLauncher
#### 4.2.1 add_QKV_bias
上面说过 Dense 层包含矩阵乘法和 add_bias 操作，其中 add_bias 操作在核函数 `add_QKV_bias` 中完成，源码针对两种数据类型 fp16 和 fp32 分别提供了一个 kernel，只是网络结构有所差异。  
针对 fp32，每个 block 处理一个 word，总共有 `batch_size * seq_len * 3` 个 block，对于 Q、K、V 3 个 tensor 而言，前 `batch_size * seq_len` 个 block 处理 Q，中间 `batch_size * seq_len` 个 block 处理 K，后 `batch_size * seq_len` 个 block 处理 V。示意图如下：  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rolgkM9MC5kXrobk84ykeUxxRmwD9Ed9M5TXvdRy6ox7q0yicywySiaDp0NL1RfoKJm8hEavlCkEgw/640?wx_fmt=png)  
```cuda
/**
 * @brief 
 * 
 * @tparam T                          OperationType
 * @param Q                           [batch_size, seq_len, head_num, size_per_head], query
 * @param bias_Q                      [head_num * size_per_head, ] length is the same as word's embedding dim
 * @param K                           [batch_size, seq_len, head_num, size_per_head], key
 * @param bias_K                      [head_num * size_per_head, ] length is the same as word's embedding dim
 * @param V                           [batch_size, seq_len, head_num, size_per_head], value
 * @param bias_V                      [head_num * size_per_head, ] length is the same as word's embedding dim
 * @param q_buf_                      [batch_size, head_num, seq_len, size_per_head], transpose query & add bias
 * @param k_buf_                      [batch_size, head_num, seq_len, size_per_head], transpose key & add bias
 * @param v_buf_                      [batch_size, head_num, seq_len, size_per_head], transpose value & add bias
 * @param batch_size
 * @param seq_len                     
 * @param head_num 
 * @param size_per_head 
 * @param word_per_block              1
 * @return __global__ 
 */
template<typename T>
__global__
void add_QKV_bias(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block)
{

  T* data_ptr;
  T* buf_ptr;
  const T* bias_ptr;
  // word counts per batch
  int m = batch_size * seq_len;
  // word embedding dim
  int n = head_num * size_per_head;
  // 总共有3m个block，第一部分处理q，第二部分处理k，第三部分处理v，这里使用qkv_id区分处理哪个矩阵
  int qkv_id = blockIdx.x * word_per_block / m;
  // 矩阵偏移量
  int row_offset = (blockIdx.x * word_per_block % m) * n;

  if(qkv_id == 0)
  {
    data_ptr = Q + row_offset;
    buf_ptr = q_buf_;
    bias_ptr = bias_Q;
  }
  else if(qkv_id == 1)
  {
    data_ptr = K + row_offset;
    buf_ptr = k_buf_;
    bias_ptr = bias_K;
  }
  else
  {
    data_ptr = V + row_offset;
    buf_ptr = v_buf_;
    bias_ptr = bias_V;
  }

  int batch_id = (blockIdx.x * word_per_block % m) / seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  // word_id in seq, not data_index
  int word_start_id = (blockIdx.x * word_per_block) % seq_len;

  T bias = __ldg(&bias_ptr[threadIdx.x]);

  for(int i = word_start_id; i < word_start_id + word_per_block; ++i)
  {
    // add bias
    T tmp = data_ptr[threadIdx.x] + bias;
    // buf's shape: [bacth_size, head_num, seq_len, size_per_head]
    int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head + 
      i * size_per_head + id_in_head;

    buf_ptr[target_id] = tmp;
    data_ptr += n;
  }
}

```
核函数第一部分先根据 `block_id` 确定当前处理的 tensor 具体是 Q、K、V 中的哪一个，从而拿到输入输出变量的内存地址。  
第二部分求出 tensor 中对应元素的索引，首先我们知道输入输出 tensor 是一个四维的 array，所以应该有四个索引，按维度顺序依次是 `batch_id`、`word_start_id`、`head_id`、`id_in_head`，有读者看到这里可能会有疑问：为什么要计算这些索引，前面计算了矩阵偏移量 `row_offset`，完全可以在 block 内按 `thread_id` 索引就可以拿到对应元素。原因是在 `add_QKV_bias` 核函数中计算逻辑不仅仅是 `add`，还有 `transpose`，熟悉 `multiHeadAttention` 的读者都知道对 Q、K、V 线性映射之后，紧接着就是一个 `transpose` 操作，目的是把 `embedding_dim` 这个维度划分成多个独立的 `head`，每个 `head` 后面单独进行 `attention`，所以要把 `head` 维度移到 `seq_len` 维度前面。换句话说这里的 `transpose` 解决的是“多头”的问题，和 attention 无关。  
理解了前面的逻辑，第三部分就比较简单了，先进行 `add` 操作，然后将结果按照 ` [bacth_size, head_num, seq_len, size_per_head]` 的维度顺序写在输出 tensor 中，这里隐含了一个 `transpose`，需要注意的是这个 `transpose` 操作是输出的 tensor 中元素存储顺序相对于输入 tensor 而言的，并不是对输入 tensor 做了变换。

针对 fp16，每个 block 同时处理 Q、K、V 上的同一个 word，同一个线程先后处理 3 个 word 上对应元素的计算逻辑，实际计算中把 `half` 都转成了 `half2`，使用标准库中的函数 `__hadd2` 运算。网络结构如下：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rolgkM9MC5kXrobk84ykeUBuVZlNDYTMzzRe1JaYc0xukicKzN1rlZDpD7ch92AZ2CE9Tp8ZeED6w/640?wx_fmt=png)

从图中可以看出，block_size 为 `embedding_dim` 的一半，这是因为用了 `half2` 这个数据结构，实际上每个线程处理了 2 个元素，所以线程数量缩减一半。核函数内部逻辑分为 2 个部分：1、求出 tensor 中对应元素的索引。2、一次对 Q、K、V 进行 `add` 和 `transpose` 操作。
```cuda
template <>
__global__
void add_QKV_bias(__half* Q, const __half* bias_Q, __half* K, const __half* bias_K, __half* V, const __half* bias_V, 
  __half* q_buf_, __half* k_buf_, __half* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = tid / (head_num * seq_len * size_per_head);
  int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
  int head_id = (tid % (head_num * size_per_head)) / size_per_head;
  // id in head
  int id = tid % size_per_head;
  // from [batch_size, seq_len, head_num, size_per_head] tanspose to [bacth_size, head_num, seq_len, size_per_head]
  int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);

  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)Q;
  half2* dst_ptr = (half2*)q_buf_;
  const half2* bias_ptr = (const half2*)bias_Q;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

  src_ptr = (half2*)K;
  dst_ptr = (half2*)k_buf_;
  bias_ptr = (const half2*)bias_K;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

  src_ptr = (half2*)V;
  dst_ptr = (half2*)v_buf_;
  bias_ptr = (const half2*)bias_V;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
}
```

#### 4.2.2 计算 attention scores
先来看一下 attention 的计算公式，定义如下：
$$
Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
其中，$Attention \, scores = QK^T$，也就是说这一步要解决的是一个矩阵计算，用 tensorflow 代码表示如下：
```cuda
scores = tf.matmul(query, key, transpose_b=True)
```
针对矩阵运算，源码中直接调用了 cuBLAS API，具体代码如下：
```cuda
DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
// 计算 q * k^T
check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    seq_len, seq_len, size_per_head,
    &alpha,
    k_buf_, AType_, size_per_head, seq_len * size_per_head,
    q_buf_, BType_, size_per_head, seq_len * size_per_head,
    &beta,
    qk_buf_, CType_, seq_len, seq_len * seq_len,
    batch_size * head_num,
    computeType_,
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));
```
不熟悉 attention 的读者可能会问 attention scores 的具体含义是什么，笔者在早期的文章中有过介绍，其实就是两个矩阵的词向量两两相乘，向量相乘有什么含义？相似度，这个分数就代表 Q、K 的相似度。

#### 4.2.3 softmax_kernel
拿到 Q、K 的相似度之后，直观上只要右乘一个 V 就可以得到 `attention out`，其含义就是一个加权平均的概念，既然要加权平均，必然要对权值进行归一化处理，这里的 softmax 就是这个作用。关于 softmax 核函数的实现方法笔者在前两篇文章也有介绍，OneFlow 官方给出了更为高效的实现方式，其高效的原因主要在访存带宽处理上，有兴趣的读者可以移步。[【CUDA编程】OneFlow Softmax 算子源码解读之WarpSoftmax](https://zhuanlan.zhihu.com/p/646994689)，[【CUDA编程】OneFlow Softmax算子源码解读之BlockSoftmax](https://zhuanlan.zhihu.com/p/646998408)

```cuda
// 计算softmax(qk)
if(seq_len <= 32)
    block.x = 32;
else if(seq_len > 32 && seq_len <= 64)
    block.x = 64;
else if(seq_len > 64 && seq_len <= 128)
    block.x = 128;
else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
else
    block.x = 1024;

if(batch_size * head_num <= 120)
{
    grid.x = batch_size * head_num * seq_len;
    softmax_kernel_v2<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scaler); 
}
else
{
    grid.x = batch_size * head_num;
    softmax_kernel<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scaler); 
}
```

源码中核函数的 `block_size` 是根据 `seq_len` 确定的，取大于 `seq_len` 且为 `32` 的 $2^n$ 的最小值。  
另外在调用 softmax kernel 之前，会根据 `batch_size * head_num` 选择不同的 softmax kernel，主要是为了保证在大 batch 的情况下的计算效率，这里以 `120` 为阈值，应该是作者的经验数值。这里作者给出了 2 个 softmax kernel 的实现。  
当 `batch_size * head_num > 120` 时，此时 batch 内元素较多，`grid_size` 取 `batch_size * head_num`，这时一个线程内处理一个 `seq_len` 的数据。  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rolgkM9MC5kXrobk84ykeUDyacVGmQdwBnEAdy9JuyqkG0viaJkMTzUoItdOBWc2PM8T6SGV33qRw/640?wx_fmt=png)
```cuda
/**
 * @brief 
 * 
 * @tparam T 
 * @param qk_buf_                 [batch_size, head_num, seq_len, seq_len]
 * @param attr_mask               [batch_size, seq_len, seq_len]
 * @param batch_size 
 * @param head_num 
 * @param seq_len 
 * @param scaler                  缩放因子
 * @return __global__ 
 */
template <typename T>
__global__
void softmax_kernel(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, 
  const T scaler)
{
    // grid_size = batch_size * head_num
    int batch_id = blockIdx.x / head_num;
    // batch偏移量
    int qk_offset = blockIdx.x * seq_len * seq_len;
    int mask_offset = batch_id * seq_len * seq_len;

    __shared__ float s_sum, s_max;

    // 每次处理一个seq_len的数据
    for(int i = 0; i < seq_len; ++i)
    {
      float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
      float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      // 对于某些padded的word，给一个很小的值使其近似达到不参与运算的目的
      mask_val = (1.0f - mask_val) * -10000.0f;

      float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val): -1e20f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val + 1e-6f;
      }
      __syncthreads();

      if(threadIdx.x < seq_len)
        qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

      qk_offset += seq_len;
      mask_offset += seq_len;
    }
}
```

核函数内首先计算了每个元素偏移量，对于输入 tensor 而言，每个 block 处理 `seq_len * seq_len` 个数据，所以 block 内元素偏移量为 `blockIdx.x * seq_len * seq_len`，而对于 mask 矩阵而言，其维度为 `[batch_size, seq_len, seq_len]`，跟 `head_num` 无关，所以其偏移量为 `batch_id * seq_len * seq_len`。  
接下来是一层循环，对于 `seq_len * seq_len` 矩阵而言，每个线程处理当前 `thread_id` 列的元素，每轮循环结束，处理该列下一行的元素。在每一轮循环中，所有的线程一起处理一行数据，首先拿到数据 `qk` 以及 `mask_val`。如果 `mask_val` 为 `0`，则给 `mask_val` 赋一个很小的值最后加在 `qk` 上使 `qk` 值很小，以致最终这个 softmax 分量趋于 `0`；如果 `mask_val` 为 `1`，则 mask 不干预后续计算。每个线程拿到处理后的 `qk` 值即 `tmp` 后，进行一次块内规约，即可求出当前行的最大值 `max_val`，然后为了避免指数运算导致数值溢出，让 `tmp` 减去 `max_val` 并求其指数值赋给 `qk` ，然后对 `qk` 再一次块内规约求出当前行的和 `s_sum`，最后让 `qk` 除以 `s_sum` 即可得到 softmax 值。核函数内要注意在两次块内规约后一定要进行一次块内同步，否则可能计算错误。  

当 `batch_size * head_num <= 120` 时，此时 batch 较小，`grid_size` 取 `batch_size * head_num * seq_len`，这时一个线程块内处理一行数据，每个线程内只处理一个的数据。  
```cuda
template <typename T>
__global__
void softmax_kernel_v2(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, 
  const int seq_len, const float scaler)
{
    int batch_id = blockIdx.x / head_num / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int qk_offset = blockIdx.x * seq_len;
    int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
    mask_val = (1.0f - mask_val) * -10000.0f;

    float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val) : -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}
```
这种情况下不涉及循环处理，计算逻辑与前面 `softmax_kernel` 循环体内部计算逻辑相同，不再赘述。

#### 4.2.4 计算多头 attention out
这一步的意思就是使用 softmax 后的相似度矩阵右乘一个 V，得到多头注意力输出，注意这时候输出 tensor 的维度为 `[batch_size, head_num, seq_len, size_per_head]`。源码中直接调用了 cuBLAS API，具体代码如下：
```cuda
// 计算qk * v
check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    size_per_head, seq_len, seq_len,
    &alpha,
    v_buf_, AType_, size_per_head, seq_len * size_per_head,
    qk_buf_, BType_, seq_len, seq_len * seq_len,
    &beta,
    transpose_dst_, CType_, size_per_head, seq_len * size_per_head,
    batch_size * head_num,
    computeType_,
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));
```
#### 4.2.5 transpose
前面说过，多头 `attention out` 的维度为 `[batch_size, head_num, seq_len, size_per_head]`，此时这些 head 已经完成使命了，通过独立的 `head_num` 组 attention 参数计算出了 `attention out`，最后需要做的就是把这 `head_num` 组 `attention out` 拼接起来，体现在 tensor 上就是做一次 `transpose`，将维度变为 `[batch_size, seq_len, head_num,  size_per_head]`。源码针对 fp16 和 fp32 分别提供了一个核函数 `transpose`，计算逻辑和 `add_QKV_bias` 中 transpose 计算逻辑相同，索引按顺序乘即可。具体代码如下：
```cuda
template<typename T>
__global__
void transpose(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len))/ seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
    + head_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template<>
  __global__
void transpose(__half* src, __half* dst,
    const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int batch_id = tid / (head_num * seq_len * size_per_head);
  int head_id = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
  int seq_id = (tid % (seq_len * size_per_head)) / size_per_head;
  int id = tid % size_per_head;

  int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);
  half2* src_ptr = (half2*)src;
  half2* dst_ptr = (half2*)dst;

  dst_ptr[target_id] = src_ptr[tid];
}
```

## 5 BertEncoderTransformer
`BertEncoderTransformer` 类中有两个重要的成员方法：构造函数、`forward` 方法。其中构造函数内主要进行一些参数初始化功能，设备内存的申请和初始化也在该函数内进行。`forward` 方法内主要是核心逻辑的实现。  
### 5.1 attention forward
根据调用链可知，`BertEncoderTransformer->forward()` 中第一步就是 `attention_->forward()`，其中 `attention_` 对象在构造函数中被定义，`attention_->forward()` 执行的就是第 4 节的内容。

### 5.2 对 attention out 做线性变换
根据流程图和调用链可知，这一步是对多头注意力的输出 tensor 做一层线性变换，右乘一个参数矩阵，其实就是一个不加激活函数的 Dense 层，分为矩阵乘法和 add bias 两个操作步骤，这里调用了 cuBLAS API 实现矩阵乘法。
```cuda
DataType_ alpha = (DataType_)1.0f;
DataType_ beta = (DataType_)0.0f;
int m = batch_size_ * from_seq_len_;
int k = head_num_ * size_per_head_;
int n = k;

check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k, 
    &alpha, 
    param_.attr_output_kernel, AType_, n, 
    attr_out_buf_, BType_, k, 
    &beta, 
    attr_matmul_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
```

### 5.3 add_bias_input_layernorm_kernel
从核函数名字可以看出，这个核函数实现了 3 个操作：add bias、add input、layernomalization。其中 add bias 是完成上一步线性变换未完成的加偏置工作，add input 是 transformer 模型中的残差结构，layernomalization 则是层归一化操作。综合起来这个核函数的作用是：对线性变换后的 attention out 加偏置，然后加上原始输入 tensor 组成一个残差结构，最后进行一次层归一化变换。源码中针对 fp16 和 fp32 分别提供了一个核函数实现，计算逻辑都一样，这里只以 fp32 为例介绍。
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qmVJow3YFH6UXcu3SEfVOmQhITiaYbvaEP6XMZFJH2KKSpvTicgnq0A9FOE1Cb8gtICAkkpRa7fsXw/640?wx_fmt=png)
```cuda
/**
 * @brief                       grid_size = m, block_size = n
 * 
 * @tparam T 
 * @param out                   [batch_size, sql_len, latent_dim]
 * @param input                 [batch_size, sql_len, latent_dim]
 * @param bias                  [latent_dim,]
 * @param gamma 
 * @param beta 
 * @param m                     batch_size * seq_len
 * @param n                     latent_dim
 * @return __global__ 
 */
template <typename T>
__global__ 
void add_bias_input_layernorm(T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  // add，一个block处理一行
  for(int i = tid; i < n; i += blockDim.x)
    local_out += (float)(out[blockIdx.x * n + i] + input[blockIdx.x * n + i] + __ldg(&bias[i]));
  // mean_i = sum(x_i[j] for j in range(k)) / k
  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();
  // var_i = sum((x_i[j] - mean_i) ** 2 for j in range(k)) / k + epsilon
  variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-6f;
  __syncthreads();
  // x_i_normalized = (x_i - mean_i) / sqrt(var_i)
  // output_i = x_i_normalized * gamma + beta
  for(int i = tid; i < n; i += blockDim.x)
    out[blockIdx.x * n + i] = 
	    (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}
```
如示意图所示，核函数中每个 block 处理一行数据，共 `latent_dim = head_num * size_per_head` 个元素，核函数中首先计算了 add bias、add input 两个操作，并将计算结果存储在寄存器变量 `local_out` 中。  
接下来就是标准的 layerNormalization 操作，我们先来看一下 layerNormalization 的操作步骤，可以参照一下 tensorflow 框架 API 文档。
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5odFyeXeADib92AGEt85n1xSnGBKy0n96D9DL6icJRib6Q5UXSttSzWNVuZxnj5uwADD20Pd5F6UwGqg/640?wx_fmt=png)
具体地，第一步计算均值和方差，核函数中使用块内规约计算出均值 `s_mean` 存储在共享内存中，所有块内线程都可以访问。然后根据 `s_mean` 和线程内的 `local_out` 以及 `epsilon` 系数再进行一次块内规约计算出方差 `s_variance` 存储在共享内存中。  
第二步进行归一化和线性变换，对应 tensorflow API 的二、三步，直接计算即可，没有其他技巧，公式如下：
$$
out_i = \frac{local\_out - s\_mean}{\sqrt{s\_variance}} \cdot \gamma + \beta
$$

### 5.4 FeedForward 结构
根据 Transformer 模型结构，多头注意力之后为了增强表达能力，加了一个 FeedForward 层，该结构内部就是两个 Dense 层，第一层 Dense 中使用了激活函数，第二层没有激活函数。所以 FeedForward 层中包含了 5 个操作：矩阵乘法、add bias、activation、矩阵乘法、add bias。

#### 5.4.1 attention out * inter kernel
FeedForward 层第一次线性变换会扩展 tensor 的最后一个维度的长度，源码中将 `latent_dim`（也就是 `n`）扩展为原来的 4 倍，所以这里的 `inter kernel` 的形状为 `[latent_dim, 4 * latent_dim]`，矩阵运算后的输出 tensor 形状为 `[batch_size, seq_len, 4 * latent_dim]`。
```cuda
n *= 4;
check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k, 
    &alpha, 
    param_.inter_kernel, AType_, n, 
    attr_matmul_buf_, BType_, k, 
    &beta, 
    inter_matmul_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));
```

#### 5.4.2 add_bias_act_kernel
顾名思义，`add_bias_act_kernel` 核函数包含了 add bias 和 activation 两个操作。源码中 `block_size = n / 4` 实际就是 `latent_dim`，为什么不直接取上一次运算后的矩阵宽度 `n = 4 * latent_dim` 呢？这里是希望一行元素（`4 * latent_dim`）能在一个 block 内处理，如果 `block_size` 直接取 `n = 4 * latent_dim`，可能超过 1024，因此还取 `latent_dim`，线程内循环 4 次处理即可。同样，源码中针对 `grid_size` 也取了 `m / 4`，在线程中通过循环每次跨 `m / 4` 步长处理 4 行数据。
```cuda
template <typename T>
__global__ 
void add_bias_act(T* out, const T* bias, int m, int n)
{
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m){
      val = out[tid + i * blockDim.x + row_id * n]+ reg_bias;
      out[tid + i * blockDim.x + row_id * n] = gelu<T>(val);
      row_id += gridDim.x;
    }
  }
}
```
核函数中先对列进行循环，`ite = 4`，从全局内存读出当前列的 `bias`，然后针对行进行循环，步长为 `m / 4`，循环体内部对当前行当前列的元素进行 add bias 和 gelu 操作，这里gelu 操作是一个简单的 element-wise 操作，比较简单不再介绍。  
**笔者点评**：这里笔者私以为没有必要 `grid_size` 也取 `m / 4`，cuda 本身对线程块的数量没有限制，完全可以直接取 `m`，每次每个线程只处理一行数据，一方面可以增加并行程度，另一方面代码可阅读性也更好。笔者给出代码如下，亲测可用。  
```cuda
dim3 grid(m);
dim3 block(n / 4);
assert(block.x <= 1024);
add_bias_act_v2<T><<<grid, block, 0, stream>>>(out, bias, m, n);

template <typename T>
__global__ 
void add_bias_act_v2(T* out, const T* bias, int m, int n) {
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i) {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    val = out[tid + i * blockDim.x + row_id * n]+ reg_bias;
    out[tid + i * blockDim.x + row_id * n] = gelu<T>(val);
  }
}
```

#### 5.4.3 inter out * out kernel
FeedForward 层第二次线性变换将 tensor 的最后一个维度的长度转换为原始大小，源码中将 `n` 重新赋值为 `latent_dim`，所以这里的 `out kernel` 的形状为 `[4 * latent_dim, latent_dim]`，矩阵运算后的输出 tensor 形状为 `[batch_size, seq_len, latent_dim]`。
```cuda
n = k;
k *= 4;
check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k, 
    &alpha, 
    param_.output_kernel, AType_, n, 
    inter_matmul_buf_, BType_, k, 
    &beta, 
    param_.transformer_out, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));
```

### 5.5 add_bias_input_layernorm_kernel
这个核函数的计算逻辑在 5.3 中已经介绍过了，包含加偏置项、残差结构、层归一化三个操作，不再赘述。  

## 6 小结
至此，Transformer encoder 前向计算的 14 个操作优化技巧已介绍完毕。总结如下：
- 针对半精度 fp16 的优化方面。首先，在 kernel 的实现中，将输入的 half 指针转成 half2 类型，并使用了 half2 相关的数学函数。这样不仅仅可以达到 2 倍于 half 的访存带宽和计算吞吐，还可以极大地减少指令的发射数量。其次，在 softmax 以及 layerNormalization 的操作中，为防止求和溢出，将数据以 half2 的形式读入后，会转成 float2 类型，来做求和计算，这里就非常细致了，尽可能地保障了较高精度，值得学习借鉴。
- 针对访存带宽方面，笔者以为除 fp16 以外其它数据类型也可以进一步优化，比如可以自定义 pack 类型进行合并读写，尽量把带宽打满。
- 针对线程网络结构方面，源码中基本使用一个 block 处理一行数据的模式进行设计，这里笔者私以为针对 `seq_len` 和 `latent_dim` 已知比较小的情况下（不超过1024），完全可以一个线程束处理一行数据，束内同步的开销远小于块内同步。当然，这个要求确实有些苛刻了。
- 源码中提供了一个块内规约的代码，思路非常好，值得初学 cuda 的读者品读。