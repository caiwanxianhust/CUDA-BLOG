#! https://zhuanlan.zhihu.com/p/650462095
# 【CUDA编程】Faster Transformer v2.0 源码详解
**写在前面**：笔者之前对 Nvidia BERT 推理解决方案 Faster Transformer v1.0 版本源码进行了深度剖析，详细介绍了源码中的各种优化技巧，文章受到不少读者的肯定。应读者之邀，本文将对 Faster Transformer v2.0 版本源码进行解读，为了便于交流讨论，除公众号：**后来遇见AI** 以外，本文也将在知乎进行发布，欢迎各位读者阅读并给出意见。  

## 1 v2.0 版本发布背景
2019 年 7 月，Nvidia 官方开源了 Nvidia BERT 推理解决方案 Faster Transformer v1.0，针对 BERT 中的 Transformer Encoder 进行优化和加速，以满足在线业务的低延迟要求。  
在解决了 Transformer Encoder 的性能问题之后，Nvidia 将重点放到了同样重要的 Transformer Decoder 推理上。调查研究发现，在众多基于 Encoding-Decoding 的 NLP 应用推理过程中，有 90% 以上的时间是消耗在 Decoder 上面。因此，官方在 FasterTransformer v1.0 版本的基础上，推出了2.0 的版本，增加了针对 Decoder 的优化。其优越的性能将助力于翻译，对话机器人，文字补全修正等多种生成式的场景。  
简单来说，v1.0 解决了推理过程中 Encoder 部分的性能问题，极大地提升了序列标注等应用场景的推理速度。v2.0 在此基础上加入了 Decoder 部分的优化，使得 Seq2Seq 场景的推理速度也得到显著改善。

## 2 整体架构
同 v1.0 版本一样，v2.0 的底层由 CUDA 和 cuBLAS 实现，支持 FP16 和 FP32 两种计算模式。为了兼顾灵活性与效率，官方提供了两个不同大小和效果的模块。其中，较小的 Decoder 模块主要优化了 Transformer layer 的部分，能够提供 2~4 倍左右的加速，相当于我们常说的 decoder layer；而较大的 Decoding 模块则包含了整个解码的流程，灵活性上较 Decoder 模块稍差，但最高能够达到 10 倍以上的加速，相当于我们常说的 decoder model。  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qGSLlF2uibuwVnqbW7TZN1xbiabALenxvibAHEddzpGdtzQynbXhauYUJyOYApOEiaTROic7p0CiaEgdVg/640?wx_fmt=png)

上图展示了 Decoder 和 Decoding 的差别。黄色区块是 Decoder，它包含两个 attention 和一个 feed forward network。而蓝色的大区块则是 Decoding，它包含了整个 Decoding 的流程，除了 Decoder 外还包括 embedding lookup、sine position encoding、beam search 等等。  

总结来看，相比 v1.0来说 v2.0 做了以下的更新:
- 加入了 Transformer Decoder 优化实现。可以支持 Encoder-Decoder 的业务场景。目前 Decoder 可以支持的超参数范围如下:
  - I. Headnumber: 8, 12  
  - II. Sizeper head: 64  
  - III. Sequencelength: 128 以内
  - IV. Batch size * beam width: 1024 以内
  - V. 字典大小: 64 到 30000
需要注意的是，超参数超出支持范围的情况，我们未验证其性能和正确性。
- 修复 FasterTransformer 1.0 当中的 bug，并且支持 dynamic batch size。
- 代码进行部分部分重构。
  

另外再明确一点，v2.0 优化的仍然是推理过程的性能，并不是用来加速训练过程的。  
从更新说明可以发现，本次版本更新主要是添加了 Decoder 和 Decoding 两个模块，接下来笔者将分别对这两个模块的源码进行解读。源码地址如下，有兴趣的读者可以前往下载：
>https://github.com/NVIDIA/FasterTransformer/tree/v2.0/

## 3 Decoder模块
### 3.1 计算框架
前面说过 Decoder 模块实际就是一个单层的解码 layer 的实现，宏观上包含两个 multi-head attention layer 和 一个 ffn layer，细节上 layer 与 layer 之间还加入了残差结构和层归一化等操作，具体计算逻辑如下图：  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qGm2nSuuUsyGqO2np6BquupK3xKFO4HNXiaicPr4zyBqZFib0FBFXpPzPzpNsbnE1v4hgbhcznJXIbw/640?wx_fmt=png)

整体结构可以分成 7 个步骤，其中，橘色外框的 mask multi-head attention 可以在拆成 5 个核函数，cross multi-head attention 可以在拆成 3 个核函数，FFN 也可以再拆成 3 个核函数。

### 3.2 调用链
Decoder 模块的核心逻辑封装在 open_decoder.h 文件的 `OpenDecoder` 类中，计算逻辑都在 `forward` 函数里，具体调用链如下：
```cpp
OpenDecoder->initialize()
OpenDecoder->forward()
   ->decoder_norm1
   ->masked_multi_head_attention
   ->decoder_norm2
   ->cross_multi_head_attention
   ->decoder_norm2
   ->ffn
   ->add_bias_input
```
7 个函数对应结构图中 7 个步骤，下一面将对 7 个函数源码进行一一解读。
```cpp
/**
   * @brief 
   * 
   * @param from_tensor                     [batch_size_, hidden_units_]
   * @param memory_tensor 
   * @param key_cache_                      [cache_size, ] = [batch_size_, seq_len, hidden_units_]
   * @param value_cache_                    [cache_size, ] = [batch_size_, seq_len, hidden_units_]
   * @param key_mem_cache_                  [cache_size, ] = [batch_size_, seq_len, hidden_units_]
   * @param value_mem_cache_                [cache_size, ] = [batch_size_, seq_len, hidden_units_]
   * @param memory_sequence_length 
   * @param decoder_output 
   * @param step 
   */
  void forward(const DataType_ *from_tensor, const DataType_ *memory_tensor,
               DataType_ *key_cache_, DataType_ *value_cache_,
               DataType_ *key_mem_cache_, DataType_ *value_mem_cache_,
               const int *memory_sequence_length, DataType_ *decoder_output, const int step)
  {
    int m = batch_size_;
    int n = hidden_units_;
    try
    {
      /* masked multi-head attention */
      /* layernorm(from_tensor) -> norm_from_tensor_buf_ */
      decoder_norm1(from_tensor, param_.self_layernorm.gamma,
                    param_.self_layernorm.beta, norm_from_tensor_buf_, m, n);

      masked_multi_head_attention(norm_from_tensor_buf_, key_cache_, value_cache_, masked_output_buf_, step);

      /* add bias to masked_output_buf_
           masked_output_buf_ + from_tensor -> masked_output_buf_
           norm(masked_output_buf_) -> norm_masked_output_buf_ 
        */
      decoder_norm2(from_tensor, 
                    param_.cross_layernorm.gamma, 
                    param_.cross_layernorm.beta,
                    param_.self_attention.attention_output_weight.bias, 
                    masked_output_buf_, 
                    norm_masked_output_buf_, m, n);

      /* cross attention with memory */
      cross_multi_head_attention(norm_masked_output_buf_, memory_tensor,
                                 key_mem_cache_, value_mem_cache_, cross_output_buf_,
                                 memory_sequence_length, max_seq_len_, step);

      /* cross_output_buf_ + bias + masked_output_buf_ -> cross_output_buf_
           norm(cross_otuput_buf) -> normed_last_context (input for ffn)
       */
      decoder_norm2(masked_output_buf_, 
                    param_.ffn_layernorm.gamma, 
                    param_.ffn_layernorm.beta,
                    param_.cross_attention.attention_output_weight.bias,
                    cross_output_buf_, 
                    norm_cross_output_buf_, m, n);

      ffn(norm_cross_output_buf_, ffn_inner_buf_, decoder_output, m, 4 * n, n);

      add_bias_input(decoder_output, cross_output_buf_, m, n);
    }
    catch (std::runtime_error &error)
    {
      throw error;
    }
  }
```

### 3.3 initialize
在执行 forward 之前，需要先调用 `initialize` 函数进行初始化处理，该函数接收两个参数 `param` 和 `buf`。
```cpp
/**
  * @brief decoder layer 初始化
  * 
  * @param param 当前 layer 的所有权重参数，包含layerNorm层、attention层、ffn层等
  * @param buf decoder_buf_
  */
void initialize(DecoderInitParam<DataType_> param, DataType_ *buf)
{
#ifndef NDEBUG
  PRINT_FUNC_NAME_();
#endif
  param_ = param;
  int buf_size = batch_size_ * hidden_units_;
  norm_from_tensor_buf_ = buf;
  query_buf_ = buf + buf_size;       //store the query values (from_tensor * Q) in both masked and multi-head attention
  context_buf_ = buf + 2 * buf_size; //store the context result (softmax(qk)v) in both masked and multi-head attention

  masked_output_buf_ = buf + 3 * buf_size;      //masked_attention_output
  norm_masked_output_buf_ = buf + 4 * buf_size; //norm(masked_attention_output)

  cross_output_buf_ = buf + 5 * buf_size;      //mutli-head attention_output
  norm_cross_output_buf_ = buf + 6 * buf_size; //norm(multi-head attention_output)
  ffn_inner_buf_ = buf + 7 * buf_size;         //4 buf size to store inner product
}
```
笔者根据源码绘制了一份内存分布图如下：  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5p5XEtkDKpgcX0VHMoRncOPR91l0TpqpWd2XoIu7bpc9EkhrSa062ONI8d65EWXicNZGa9qbYLoZ3Q/640?wx_fmt=png)

`param` 内包含了 decoder layer 中的权重参数，包含 layerNorm 层、attention 层、ffn 层等，可以看一下源码。
```cpp
template <typename T>
class DecoderInitParam
{
public:
  /* weights for masked_multi_head_attention */
  LayerNormWeight<T> self_layernorm;
  AttentionWeight<T> self_attention;

  LayerNormWeight<T> cross_layernorm;
  AttentionWeight<T> cross_attention;

  LayerNormWeight<T> ffn_layernorm;
  FFNWeight<T> ffn;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};
```
`buf` 是从外部传进来的一个设备内存上的指针，表示 decoder layer 数据的起始位置，在函数内部通过偏移量来确定各个变量的起始地址，内存分布如上图。另外说一下，关于 decoder layer 的 `batch_size_` 参数，实际调用中是通过构造函数初始化的，在 `DecodingOpenNMT` 的构造函数中 new 了一个 `decoder_`，使用 `batch_size * beam_width` 初始化了 decoder layer 中的 `batch_size_` 参数。也就是说，Decoder 模块中的 `batch_size_` 实际是 `batch_size * beam_width`，不过我们现在只讨论 Decoder 的计算逻辑，可以就把他当 `batch_size` 来看，这并不影响计算。  
```cpp
// 注意这里是用batch_size * beam_width当做batch_size初始化decoder
decoder_ = new OpenDecoder<OpType_>(allocator, batch_size * beam_width, memory_max_seq_len, 
                                    head_num, size_per_head, memory_hidden_units);
```

### 3.4 decoder_norm1 
这个函数的作用是对输入 tensor 进行层归一化操作，内部调用了一个核函数 `decoder_norm1_kernel` 完成计算，代码如下：
```cpp
/**
 * @brief 对输入 tensor 进行层归一化操作
 * 
 * @tparam OpType_ 
 * @param input                 [batch_size_, hidden_units_]
 * @param gamma 
 * @param beta 
 * @param output                [batch_size_, hidden_units_]
 * @param m                     batch_size_
 * @param n                     hidden_units_
 */
template<OperationType OpType_>
void OpenDecoder<OpType_>::decoder_norm1(
  const DataType_* input,
  const DataType_* gamma,
  const DataType_* beta,
  DataType_* output,
  int m, int n)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
     Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */
  if(n % 32 != 0)
    block.x = 1024;

  assert(n <= 1024);

/* should pay attention to the rsqrt precision*/
  decoder_norm1_kernel<DataType_><<<grid, block, 0, param_.stream>>>(input, gamma, beta, output, m, n);
}
```
有读者看了注释可能要问，为什么这里的输入输出 tensor 的形状是 `[batch_size_, hidden_units_]`，而不是 `[batch_size_, seq_len, hidden_units_]` ？这是因为推理场景下解码的过程是一个 step 一个 step 进行的，一次只能解码一个 token，所以没有 `seq_len` 这个维度了。另外补充说一下，这里的 `hidden_units_ = size_per_head * head_num`。  
通过源码可以发现，这里 `grid_size` 直接设置为 `batch_size_`，一个 block 处理一个 token。当 `hidden_units_` 小于 `1024` 且可以被 `32` 整除时，把 `block_size` 设置为 `hidden_units_`，除此之外 `block_size` 都直接取 `1024`。这里可以预想到计算规模不会太大，所以两个核函数的超参数不需要考虑太多，直接根据业务需要设置即可。具体 layerNormalization 的计算公式原理，可以参照一下笔者上一篇文章[【CUDA编程】Faster Transformer v1.0 源码详解](https://zhuanlan.zhihu.com/p/647012855)，这里不再赘述。  
核函数内先定义了两个共享内存变量 `s_mean` 和 `s_variance`，以及两个寄存器变量 `mean` 和 `variance`。先从全局内存 `input` 中取出当前线程对应的变量的值 `local_out`，执行一次块内规约得到 block 内元素的和，存入 `mean`，在 0 号线程内求出均值 `mean`，然后存入 `s_mean`，同样的方法求出 `s_variance`，然后根据公式算出结果即可。
```cpp
template <typename T>
__global__
void decoder_norm1_kernel(const T* input, const T* gamma, const T* beta, T* output, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = tid < n ? (float)(__ldg(&input[blockIdx.x * n + tid])) : 0.0f;

  mean = blockReduceSum<float>(local_out);

  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>(tid < n ? (local_out - s_mean) * (local_out - s_mean) : 0.0f);

  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6);

  __syncthreads();

  if(tid < n)
    output[blockIdx.x * n + tid] = 
      (T)(((local_out - s_mean) * s_variance) * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}
```
### 3.5 masked_multi_head_attention
这一步的作用是对 tensor 进行 self-attention 操作，总共拆分了 5 个步骤，如下图所示。  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5p5XEtkDKpgcX0VHMoRncOPYRltPVjVaETpymNoicJZU9u06wIYhhTykNzjYY5wwqNUVzdJUgsg7CA/640?wx_fmt=png)
```cpp
/**
 * @brief decoder layer masked_multi_head_attention
 * 
 * @tparam OpType_ 
 * @param from_tensor               [batch_size_, hidden_units_]
 * @param key_cache_                [cache_size,] = [seq_len, batch_size_, hidden_units_]
 * @param value_cache_              [cache_size,] = [seq_len, batch_size_, hidden_units_]
 * @param decoder_output            [batch_size_, hidden_units_]
 * @param step 
 */
template<OperationType OpType_>
void OpenDecoder<OpType_>::masked_multi_head_attention(
  const DataType_* from_tensor,
  DataType_* key_cache_,
  DataType_* value_cache_,
  DataType_* decoder_output,
  const int step)
{
  int m = batch_size_;
  int n = hidden_units_;
  int k = hidden_units_;

  DataType_* key_buf_ = key_cache_ + (step - 1) * m * n;
  DataType_* value_buf_ = value_cache_ + (step - 1) * m * n;

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.query_weight.kernel , AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    query_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.key_weight.kernel, AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    key_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.value_weight.kernel, AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    value_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  dim3 grid(batch_size_ * head_num_);
  dim3 block(128);

  //suppose size_per_head <= 128
  if(step <= 64)
    block.x = 64;
  else if(step <= 128 && step > size_per_head_)
    block.x = 128;
  else if(step > 128 && step <= 256)
    block.x = 256;
  else if(step > 256 && step <= 512)
    block.x = 512;
  else
    block.x = 1024;

  if(block.x < size_per_head_)
    block.x = size_per_head_;
  
  assert(block.x <= 1024);

  DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);

  int shared_size = sizeof(DataType_) * (size_per_head_ + step);

  masked_attention_kernel<DataType_><<<grid, block, shared_size, param_.stream>>>(
    query_buf_, param_.self_attention.query_weight.bias, 
    key_cache_, param_.self_attention.key_weight.bias,
    value_cache_, param_.self_attention.value_weight.bias,
    context_buf_, batch_size_,
    head_num_, size_per_head_, step, scalar);

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.attention_output_weight.kernel, AType_, n, 
    context_buf_, BType_, k, 
    &beta, 
    decoder_output, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
} 
```
我们先来看一下该函数的几个重要参数。
- `from_tensor`：上一步经过 layerNorm 后的 `from_tensor`，形状为 `[batch_size_, hidden_units_]`
- `key_cache_`：存储的是所有 step 的经过 Dense 变换（`from_tensor_ * weight_K + bias`）后的 `from_tensor`，形状为 `[seq_len, batch_size_, hidden_units_]`
- `value_cache_`：存储的是所有 step 的经过 Dense 变换（`from_tensor_ * weight_V + bias`）后的 `from_tensor`，形状为 `[seq_len, batch_size_, hidden_units_]`
- `decoder_output`：最终输出 tensor，形状为 `[batch_size_, hidden_units_]`

函数体内首先定义了两个变量 `key_buf_` 和 `value_buf_` 用来标记 key 和 value 在 `key_cache_` 和 `value_cache_` 中的存储位置，通过代码可以发现是分 step 存储的。看到这里，有些读者可能会问，为什么要搞 `key_cache_` 和 `value_cache_` 这么麻烦，存它有什么用？答案是计算结果复用。对于每一个 step 而言，只有 query 是新的，key 和 value 实际就是前面 step 的 `from_tensor` 经过 Dense 变换的，换句话说前面都计算过，重复计算没有意义，所以申请了两块内存存起来。

#### 3.5.1 Dense 变换
这里直接使用 cuBLAS API 进行矩阵乘法，add bias 操作放在后面的 kernel 中进行。对 `from_tensor` 使用 3 个 矩阵乘法，计算结果分别存在 `query_buf_`、`key_buf_` 和 `value_buf_` 中。

#### 3.5.2 masked_attention_kernel
核函数内部主要完成两个操作：add bias 和 计算 attention。其中 add bias 是上一步 Dense 变换的补充。`masked_attention_kernel` 是目前为止 Decoder 里面逻辑最复杂的核函数，下面笔者将尽量讲解明白，笔者水平有限，如果理解有误，欢迎各位多提意见。  
##### 3.5.2.1 核函数执行配置参数的确定  
源码中 `grid_size` 直接取 `batch_size_ * head_num_`，也就是说一个 block 处理一个 head 的元素。而 `block_size` 的确认逻辑就比较复杂了，感觉这块代码有些乱，笔者用自己的理解总结一下：
- `block_size` 满足基本规范，如 2 的 n 次幂，最大不超过 `1024`，最小取 `64`
- `block_size` 不小于 `step`
- `block_size` 不小于 `size_per_head`

读到这里不禁产生了一个疑问，为什么 `block_size` 同时跟 `step` 和 `size_per_head` 有关？原因很直观，一个 block 内做了两件事，分别完成了 `size_per_head` 个元素的计算以及 `step` 轮计算。作者把两个功能放在了一个 kernel 里，通常情况下为了代码的可维护性我们不会这么干，这里这么实现我想是为了尽可能地融合 kernel 减小 kernel 调动的开销，代价就是代码可维护性降低，不容易理解。  
除此之外，代码中提前定义了一个 `shared_size`，作为核函数内部动态共享内存的大小参数，在启动核函数时传入，可以看到，总的内存大小是 `size_per_head_ + step` 个元素。  

##### 3.5.2.2 add query bias
核函数内部首先定义了一个共享内存变量数组 `s_buf`，数组大小对应前面传入的 `size_per_head_ + step`，然后根据偏移量分别定义了两个变量 `sq` 和 `logits`，分别用来存储 add bias 和 attention 的中间结果。这种写法提供了一个思路，就是如果我们核函数内部不止一个地方需要使用动态大小的共享内存时，由于核函数执行参数里面只让传一个表示共享内存大小的参数，可以传一个总的内存大小，在核函数内部再通过偏移量自行取用，注意规划好内存大小，不要越界访问。
```cpp
extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
T* sq = reinterpret_cast<T *>(s_buf);
T* logits = reinterpret_cast<T *>(&sq[size_per_head]);
```
然后定义了一个 `qkv_id` 变量，用来描述当前线程处理的元素在 `query_buf` 中的位置。定义了一个 `qkv_bias_id` 用来描述对应元素在 bias 中的位置，可以看到 bias 的形状为 `[head_num, size_per_head]`。随后计算 query 的 add bias 并将其存入 `sq` 中。这里笔者有个疑问，就是有没有必要将 add bias 的结果存入共享内存？似乎这里并没有需要块内通信的场景，笔者认为直接存在寄存器中可能更好。  
```cpp
int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
int qkv_bias_id = head_id * size_per_head + tid;

if(tid < size_per_head)
  sq[tid] = query_buf[qkv_id] + self_Q_bias[qkv_bias_id];
__syncthreads();
```
至此，核函数内部完成了 query 的 add bias 操作，下一步该进行 softmax 的计算了。  

##### 3.5.2.3 add key bias & softmax
我们知道 attention 中 softmax 计算的对象是 query 和 key 的乘积，query 我们已经拿到了，就是当前解码 step 的输入 tensor 变换后的结果，分别存在每个 block 的 `sq` 中。key 是什么？对于当前 step 的 query 来说这里的 key 应该是前面 step 的 token 对应的 tensor 变换后的结果，前面我们讲过，由于 Dense 变换的权重是固定的且 token 也是确定的，所以 key 也是固定的，那么我们每轮 step 的时候就可以计算好当前 step 的 key 存入 `key_cache` 中供后面的 step 计算时使用，同时在当前 step 也可以从 `key_cache` 中取前面 step 的 key 用于计算。
```cpp
//offset for each step
int offset = batch_size * head_num * size_per_head;
for(int ite = 0; ite < step; ++ite)
{
  T key = tid < size_per_head ? key_cache[ite * offset + qkv_id] : (T)0.0f;
  //for the last step, we should update K + bias_K to the cache
  if(ite == step - 1 && tid < size_per_head)
  {
    key += self_K_bias[qkv_bias_id];
    key_cache[ite * offset + qkv_id] = key;
  }

  T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
  T qk = blockReduceSum(val);
  if(threadIdx.x == 0)
    logits[ite] = qk;
  __syncthreads(); //try to remove
}
__syncthreads(); //try to remove
```
首先对 step 进行循环，通过偏移量 offset 获取前面 step 的 key，如果循环到了当前 step，还需要在 key 上 add bias 然后存进 `key_cache`。拿到 key 之后和 `sq[tid]` 相乘相当于两个向量的相应位置元素相乘，再乘以一个缩放因子 `scalar` 之后进行块内规约得到 `qk`，相当于两个向量的内积。这个 `qk` 就是当前 step 的输入 token 对应的 tensor 和第 `ite+1` step 的 token 对应的 tensor 在某个 head 中的 attention scores，每一轮计算获取一个 `qk`，将其存入共享内存变量 `logits` 中，最终 `logits` 中将存储 `step` 个元素。  
```cpp
__shared__ float s_max_val, s_sum;
float local_i = tid < step ? (float)logits[tid] : -1e20f; 
float max_val = blockReduceMax<float>(local_i);
if(tid == 0)
  s_max_val = max_val;
__syncthreads();

local_i -= s_max_val;
float local_o = tid < step ? __expf(local_i) : 0.0f;
float val = blockReduceSum<float>(local_o);

if(tid == 0)
  s_sum = val + 1e-6;
__syncthreads();

if(tid < step)
  logits[tid] = local_o / s_sum;
__syncthreads();

```
接下来要对 `logits` 中的元素求 softmax，为了避免数值溢出，我们要让每个元素减去 `logits` 中的最大值。调整后的 softmax 公式如下：  
$$
softmax(x_i) = \frac{e^{x_{i} - x_{max}}}{\sum _{j=0}^{n} {e^{x_{j} - x_{max}}}}
$$
定义了两个共享内存变量 `s_max_val` 和 `s_sum` 分别存储 `logits` 中的最大值以及元素的指数和，最大值和指数和都使用块内规约获得。这块逻辑比较清晰，读者可以看代码理解。  

##### 3.5.2.4 计算 attention
根据 attention 的计算逻辑，得到 attention scores 后，右乘一个 value 矩阵就得到 attention out。具体含义就是，attention scores 代表 query 和 key 的相似度，用相似度当做权重系数，把所有 value 向量加权平均就得到 attention out。带着这个思路我们来看一下代码：
```cpp
if(tid < size_per_head)
{
  T sum = (T)0.0f;
  for(int ite = 0; ite < step; ++ite)
  {
    T value = value_cache[ite * offset + qkv_id];
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      value += self_V_bias[qkv_bias_id];
      value_cache[ite * offset + qkv_id] = value;
    }
    sum += value * logits[ite];
  }
  context_buf[qkv_id] = sum;
}
```
首先是对 step 的循环，有多少个 step 就有多少个 value 向量，具体就是从 `value_cache` 中拿到 value 对应的元素，如果是当前 step 的 value，还需要先进行 add bias 操作并存入缓存变量，然后通过 value 值和 `logits` 中的 attention score 计算加权平均值 `sum`。循环结束后把 `sum` 存进 `context` 中的对应位置。可以用下面的公式表示。
$$
Attention \, out(a_i) = \sum _{ite = 0}^{step-1} {logits[ite] * v_{ite}^{i}}
$$

#### 3.5.3 Dense 变换
这里直接使用 cuBLAS API 进行矩阵乘法，对 `context_buf` 右乘一个 `output kernel` 矩阵。add bias 操作放到后面的 kernel 中进行。

### 3.6 decoder_norm2
该函数包含了 add bias、残差结构、layerNorm 等 3 个主要操作。函数内部直接调用一个核函数 `decoder_norm2_kernel`，核函数的的 `grid_size` 取 `batch_size_`，`block_size` 取 `hidden_units_` 和 `1024` 中能被 `32` 整除的较小值。  
相比 `decoder_norm1`，只是加了两行代码，分别实现 add input 和 add bias 操作，逻辑较为简单，建议读者直接看代码。
```cpp
/**
 * @brief 该函数包含了 add bias、残差结构、layerNorm 等 3 个主要操作。
 * 
 * @tparam T 
 * @param input               from_tensor      [batch_size_, hidden_units_]
 * @param gamma 
 * @param beta 
 * @param bias                                 [hidden_units,]
 * @param output              masked_attn_out  [batch_size_, hidden_units_]
 * @param norm_output         norm_out         [batch_size_, hidden_units_]
 * @param m                   batch_size_
 * @param n                   hidden_units_
 * @return __global__ 
 */
template <typename T>
__global__
void decoder_norm2_kernel(const T* input, const T* gamma, const T* beta, const T* bias, T* output, T* norm_output, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  if(tid < n)
  {
    local_out = (float)(__ldg(&input[blockIdx.x * n + tid]));
    local_out += (float)(output[blockIdx.x * n + tid]);
    local_out += (float)(__ldg(&bias[tid]));
    output[blockIdx.x * n + tid] = (T)local_out;
  }

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>(tid < n ? (local_out - s_mean) * (local_out - s_mean) : 0.0f);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  if(tid < n)
    norm_output[blockIdx.x * n + tid] = 
      (T)((local_out - s_mean) * s_variance * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}
```

### 3.7 cross_multi_head_attention
cross attention 计算的是 decoder 的 tensor 与 encoder out 之间的 attention，具体地，decoder 的每个 step 下 token 对应的 from_tensor 作为 query，encoder out 作为 key，由于 encoder out 是不变的，权重参数也是不变的，所以 query_K 和 query_V 只在 first step 即可，把结果存到 `key_mem_cache` 和 `value_mem_cache` 中。  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5oJGnXzeUO5Nq44QRkrbKyaImXZCibnWEajXreB7czx9y5f6I5wjkWTLH1uu9fAK4Bp8UbdpTDyicLA/640?wx_fmt=png)

```cpp
/**
 * @brief attention with source sentence
 * 
 * @tparam OpType_ 
 * @param from_tensor                   [batch_size_, hidden_units_]
 * @param memory_tensor                 [batch_size_, memory_sequence_length, memory_hidden_units_]
 * @param key_mem_cache                 [batch_size_, memory_sequence_length, hidden_units_]
 * @param value_mem_cache               [batch_size_, memory_sequence_length, hidden_units_]
 * @param decoder_output                [batch_size_, hidden_units_]
 * @param length                        memory_sequence_length
 * @param seq_len                       mem_max_seq_len_
 * @param step 
 */
template<OperationType OpType_>
void OpenDecoder<OpType_>::cross_multi_head_attention(
  const DataType_* from_tensor,
  const DataType_* memory_tensor,
  DataType_* key_mem_cache,
  DataType_* value_mem_cache,
  DataType_* decoder_output,
  const int* length,
  const int seq_len,
  const int step)
{
  int m = batch_size_;
  int n = hidden_units_;
  int k = hidden_units_;

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

  //reuse the query_buf 
  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.cross_attention.query_weight.kernel, AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    query_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  if(step == 1)
  {
    m *= seq_len;
    k = memory_hidden_units_;
    check_cuda_error(cublasGemmEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      param_.cross_attention.key_weight.kernel, AType_, n, 
      memory_tensor, BType_, k, 
      &beta, 
      key_mem_cache, CType_, n, 
      computeType_, 
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

    check_cuda_error(cublasGemmEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      param_.cross_attention.value_weight.kernel, AType_, n, 
      memory_tensor, BType_, k, 
      &beta, 
      value_mem_cache, CType_, n, 
      computeType_, 
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));
    k = hidden_units_;
  }

  dim3 grid(batch_size_ * head_num_);
  dim3 block(128);

  if(seq_len <= 64)
    block.x = 64;
  else if(seq_len <= 128 && seq_len > size_per_head_)
    block.x = 128;
  else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
  else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
  else
    block.x = 1024;

  if(block.x < size_per_head_)
    block.x = size_per_head_;

  assert(block.x <= 1024);
  
  DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);

  int shared_size = sizeof(DataType_) * (size_per_head_ + seq_len);
  cross_attention_kernel<DataType_><<<grid, block, shared_size, param_.stream>>>(
    query_buf_, param_.cross_attention.query_weight.bias, 
    key_mem_cache, param_.cross_attention.key_weight.bias,
    value_mem_cache, param_.cross_attention.value_weight.bias,
    length, context_buf_,  
    batch_size_,
    head_num_, size_per_head_, step, seq_len, scalar);

  m = batch_size_;
  n = head_num_ * size_per_head_;
  k = n;

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.cross_attention.attention_output_weight.kernel, AType_, n, 
    context_buf_, BType_, k, 
    &beta, 
    decoder_output, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
}
```

#### 3.7.1 Dense 变换
使用 cuBLAS API 进行矩阵乘法，主要注意一点，`memory_tensor` 的形状是 `[batch_size_, memory_max_seq_len, memory_hidden_units_]`，其只在 first step 进行计算，计算后形状为 `[batch_size_, memory_max_seq_len, hidden_units_]`。add bias 操作放在后面的 kernel 中进行。

#### 3.7.2 cross_attention_kernel

核函数的执行配置参数确定方法和 `masked_attention_kernel`，具体见 3.5.2.1 节，这里不再重复介绍。

有所不同的是，这里提前定义的核函数共享内存大小是 `size_per_head_ + seq_len`，这个和之前有所不同，具体要看下源码。

```cpp
/**
 * @brief 
 * 
 * @tparam T 
 * @param query_buf                   [batch_size_, hidden_units_]
 * @param Q_bias                      [hidden_units_,] = [head_num, size_per_head]
 * @param key_cache                   [batch_size_, mem_max_seq_len, hidden_units_]
 * @param K_bias                      [hidden_units_,] = [head_num, size_per_head]
 * @param value_cache                 [batch_size_, mem_max_seq_len, hidden_units_]
 * @param V_bias                      [hidden_units_,] = [head_num, size_per_head]
 * @param length_per_sample           [batch_size_,]
 * @param context_buf                 
 * @param batch_size 
 * @param head_num 
 * @param size_per_head 
 * @param step 
 * @param seq_len                     mem_max_seq_len
 * @param scalar 
 * @return __global__ 
 */
template<typename T>
__global__
void cross_attention_kernel(
  T* query_buf, const T* Q_bias,
  T* key_cache, const T* K_bias,
  T* value_cache, const T* V_bias,
  const int* length_per_sample, T* context_buf, 
  int batch_size, int head_num, int size_per_head, int step, const int seq_len, const T scalar)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int length = __ldg(&length_per_sample[bid]);

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + Q_bias[qkv_bias_id];
  __syncthreads();

  for(int ite = 0; ite < length; ++ite)
  {
    int key_id = bid * (seq_len * head_num * size_per_head) + ite * (head_num * size_per_head)
     + head_id * size_per_head + tid;

    T key = tid < size_per_head ? key_cache[key_id] : (T)(0.0f);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if(step == 1 && tid < size_per_head)
    {
      key += K_bias[head_id * size_per_head + tid];
      key_cache[key_id] = key;
    }

    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
      logits[ite] = qk;
    __syncthreads(); //try to remove
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = tid < length ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < length ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();
  if(tid < length)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < length; ++ite)
    {
      int value_id = bid * seq_len * head_num * size_per_head + ite * head_num * size_per_head 
        + head_id * size_per_head + tid;

      T value = value_cache[value_id];

      //for the first step, we should add bias to key memory cache
      if(step == 1)
      {
        value += V_bias[head_id * size_per_head + tid];
        value_cache[value_id] = value;
      }  
      sum += value * logits[ite];
    }
    context_buf[bid * head_num * size_per_head + head_id * size_per_head + tid] = sum;
  }
}
```

阅读源码可以发现，`cross_attention_kernel` 的实现逻辑完全就是照搬 `masked_attention_kernel`，只不过之前是当前 step 的 token 对应的 tensor 逐个和前面 step 的 token 对应的 tensor 进行 attention，现在是当前 step 的 token 对应的 tensor 逐个和 encoder out 中每个 token 对应的 tensor 进行 attention，把 `step` 换成了 `length`，最终计算结果存入 `context_buf`，形状为 `[bacth_size_, hidden_units_]`。另外就是注意这次的 key 和 value 在 first step 时就可以完成 add bias 计算并存入缓存变量。

**笔者点评**：在 encoder out 已知且当前 step 的 `from_tensor` 确定的情况下，笔者私以为完全可以考虑更大尺度的并行计算 cross attention，而不是在核函数内部使用循环一次计算一个 encoder token 和 decoder token 间的 attention score。为此笔者给出以下代码实现，因为实现思路不同所以一些参数和变量不能和上下文完全衔接，读者理解思路即可。
```cpp
/**
 * @brief add QKV bias and transpose kv
 * 
 * @tparam T 
 * @param Q                         [batch_size, head_num, size_per_head]
 * @param K                         [batch_size, seq_len, head_num, size_per_head]
 * @param V                         [batch_size, seq_len, head_num, size_per_head]
 * @param query_buf                 [batch_size, head_num, size_per_head]
 * @param key_buf                   [batch_size, head_num, seq_len, size_per_head]
 * @param value_buf                 [batch_size, head_num, seq_len, size_per_head]
 * @param Q_bias                    [head_num, size_per_head]
 * @param K_bias                    [head_num, size_per_head]
 * @param V_bias                    [head_num, size_per_head]
 * @param seq_len 
 * @param head_num 
 * @param size_per_head 
 * @return __global__ 
 */
template<typename T>
__global__ void add_QKV_bias_kernel_transpose(T* Q, T* K, T* V, T* query_buf, T* key_buf, T* value_buf, const T* Q_bias,
    const T* K_bias, const T* V_bias, const int seq_len, const int head_num, const int size_per_head) {
    // grid_size = batch_size * seq_len, block_size = hidden_units
    int tid = threadIdx.x;
    int batch_id = blockIdx.x / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int head_id = tid / size_per_head;
    int id_in_head = tid % size_per_head;
    int hidden_units = head_num * size_per_head;
    T bias_tmp;
    if (seq_id == 0) {
        bias_tmp = tid < hidden_units ? Q_bias[tid] : 0.0f;
        query_buf[batch_id * hidden_units + tid] = Q[batch_id * hidden_units + tid] + bias_tmp;
    }
    __syncthreads();
    int target_index = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + 
        seq_id * size_per_head + id_in_head;
    bias_tmp = tid < hidden_units ? K_bias[tid] : 0.0f;
    key_buf[target_index] = K[blockIdx.x * hidden_units + tid] + bias_tmp;
    bias_tmp = tid < hidden_units ? V_bias[tid] : 0.0f;
    value_buf[target_index] = V[blockIdx.x * hidden_units + tid] + bias_tmp;
}

template<typename T>
__global__ void softmax_kernel(T* qk_buf, int* length_per_sample, const int seq_len, const int head_num, const T scalar) {
    // grid_size = batch_size * head_num
    int tid = threadIdx.x;
    int batch_id = blockIdx.x / head_num;
    int offset = blockIdx.x * seq_len;
    __shared__ T s_sum, s_max;
    int length = length_per_sample[batch_id];
    T qk = tid < length ? qk_buf[offset + tid] * scalar: -1000000.0f;
    T max_val = blockReduceMax<T>(qk);
    if (tid == 0) {
        s_max = max_val;
    }
    __syncthreads();
    T qk_tmp = tid < seq_len ? __expf(qk - s_max) : 0.0f;
    T sum_val = blockReduceSum<T>(qk_tmp);
    if (tid == 0) {
        s_sum = sum_val + 1e-6f;
    }
    if (tid < seq_len) {
        qk_buf[offset + tid] = qk_tmp / s_sum;
    }
}

/**
 * @brief 
 * 
 * @tparam T                        
 * @param from_tensor                [batch_size_, hidden_units_]        
 * @param K                          [batch_size_, mem_max_seq_len, hidden_units_]
 * @param V                          [batch_size_, mem_max_seq_len, hidden_units_]
 * @param query_buf                  [batch_size_, hidden_units_]
 * @param key_buf                    [batch_size_, mem_max_seq_len, hidden_units_]
 * @param value_buf                  [batch_size_, mem_max_seq_len, hidden_units_]
 * @param Q_bias                     [hidden_units_,]
 * @param K_bias                     [hidden_units_,]
 * @param V_bias                     [hidden_units_,]
 * @param context_buf                [batch_size_, hidden_units_]
 * @param batch_size 
 * @param head_num 
 * @param length                     [bacth_size,]
 * @param size_per_head 
 * @param step 
 * @param seq_len                    mem_max_seq_len
 * @param scalar 
 * @return 
 */
template<typename T>
void customer_cross_attention(T* from_tensor, T* K, T* V, T* query_buf, T* key_buf, T* value_buf, T*qk_buf, const T* Q_bias,
    const T* K_bias, const T* V_bias, T* context_buf, int batch_size, int head_num, int* length,
    int size_per_head, const int seq_len, const T scalar) {
    
    dim3 grid(batch_size * seq_len);
    dim3 block(1024);
    add_QKV_bias_kernel<T><<<grid, block, 0, steam>>>(from_tensor, K, V, query_buf, key_buf, value_buf, 
        Q_bias, K_bias, V_bias, seq_len, head_num, size_per_head);
    
    int m = 1;
    int n = seq_len;
    int k = size_per_head;
    check_cuda_error(cublasGemmStridedBatchedEx(param_.cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        key_buf, AType_, k, n * k,
        query_buf, BType_, k, m * k,
        &beta,
        qk_buf, CType_, n, m * n,
        batch_size * head_num,
        computeType_,
        static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

    grid(batch_size * head_num);
    block(1024);
    softmax_kernel<T><<<grid, block, 0, stream>>>(qk_buf, length, seq_len, head_num, scalar);
    k = seq_len;
    n = size_per_head;
    check_cuda_error(cublasGemmStridedBatchedEx(param_.cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        value_buf, AType_, n, n * k,
        qk_buf, BType_, k, m * k,
        &beta,
        context_buf, CType_, n, m * n,
        batch_size * head_num,
        computeType_,
        static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));
}
```
总的来说，笔者把 cross attention 分为 4 步。第一步调用核函数 `add_QKV_bias_kernel` 对 `from_tensor` 和 key、query 进行 add bias 操作，`grid_size` 设置为 `batch_size * seq_len`，由于 `from_tensor` 没有 `seq_len` 这个维度，所以只有当 `seq_id == 0` 的 block 需要进行计算，另外结束后再对 key 和 value 进行 transpose 将其形状变为 `[batch_size, head_num, seq_len, size_per_head]`。第二步，调用 cuBLAS API 进行矩阵乘法，实现 $QK^T$，得到 attention scores `qk_buf`，形状为 `[batch_size_, head_num, 1, seq_len]`。第三步调用核函数 `softmax_kernel` 对 `qk_buf` 进行 softmax，实现 $softmax(QK^T)$。第四步调用 cuBLAS API 进行矩阵乘法，实现 $softmax(QK^T)V$，得到 `context_buf`，形状为 `[batch_size_, head_num, 1, size_per_head]` 相当于 `[batch_size_, hidden_units_]`。  
写完之后回顾，发现似乎也并不见得真的提升了计算效率，毕竟笔者的计算逻辑虽然容易理解，且由于没有在核函数内部使用循环从而并行程度更高，但总的来说需要四步，核函数调用开销要大于源码的实现方式，姑且也算一种思路吧。  

#### 3.7.3 Dense 变换
与 3.5.3 节一样，这里直接使用 cuBLAS API 进行矩阵乘法，对 `context_buf` 右乘一个 `output kernel` 矩阵得到 `decoder_output`。add bias 操作放到后面的 kernel 中进行。

### 3.8 decoder_norm2
与 3.6 节一样，实现了 add bias、残差结构、layerNorm 等 3 个主要操作，只不过入参变了而已。

### 3.9 ffn 层
关于 FeedForward 层笔者在上一篇介绍 Faster Transformer v1.0 的文章中讲过，该结构内部就是两个 Dense 层，第一层 Dense 中使用了激活函数，第二层没有激活函数，另外加入了残差结构。所以 FeedForward 层中包含了 6 个操作：矩阵乘法、add bias、activation、矩阵乘法、add bias、add input，源码中通过 `ffn` 和 `add_bias_input` 两个函数实现。
#### 3.9.1 from_tensor * inter kernel
FeedForward 层第一次线性变换会扩展 from_tensor 的最后一个维度的长度，源码中将 `hudden_units_` 扩展为原来的 4 倍，所以这里的 inter kernel 的形状为 `[hudden_units_, 4 * hudden_units_]`，矩阵运算后的输出 `ffn_inner` 形状为 `[batch_size_, 4 * hudden_units_]`。
```cpp
int m1 = m, k1 = n, n1 = inner_size;
DataType_ alpha = (DataType_)1.0f;
DataType_ beta = (DataType_)0.0f;

check_cuda_error(cublasGemmEx(param_.cublas_handle, 
  CUBLAS_OP_N, CUBLAS_OP_N, 
  n1, m1, k1, 
  &alpha, 
  param_.ffn.intermediate_weight.kernel, AType_, n1, 
  input, BType_, k1, 
  &beta, 
  ffn_inner, CType_, n1, 
  computeType_, 
  static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));
```
#### 3.9.2 add_bias_relu
顾名思义，`add_bias_relu` 核函数包含了 add bias 和 relu 两个操作。源码中 `block_size = n1 / 4` 实际就是 `hudden_units_`，为什么不直接取上一次运算后的矩阵宽度 `n1 = 4 * hudden_units_` 呢？这里是希望一行元素（4 * hudden_units_）能在一个 block 内处理，如果 `block_size` 直接取 `n1`，可能超过 `1024`，因此还取 `hudden_units_`，线程内循环 `4` 次处理即可。核函数逻辑非常简单，注意按步长 `blockDim.x` 取数即可。
```cpp
template <typename T>
__global__ 
void add_bias_relu(T* out, const T* bias, int m, int n)
{
  // grid_size = batch_size_, block_size = 4 * hidden_units_
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m)
    {
      val = out[tid + i * blockDim.x + row_id * n] + reg_bias;
      out[tid + i * blockDim.x + row_id * n] = (T)(val > 0.0f ? val : 0.0f);
      row_id += gridDim.x;
     }
  }
}
```
#### 3.9.3 inter out * out kernel
FeedForward 层第二次线性变换将 tensor 的最后一个维度的长度转换为原始大小，源码中将 `n2` 赋值为 `hudden_units_`，所以这里的 out kernel 的形状为 `[4 * hudden_units_, hudden_units_]`，矩阵运算后的输出 tensor 形状为 `[batch_size, hudden_units_]`。  
```cpp
int m2 = m, n2 = n, k2 = inner_size;
check_cuda_error(cublasGemmEx(param_.cublas_handle, 
  CUBLAS_OP_N, CUBLAS_OP_N, 
  n2, m2, k2, 
  &alpha, 
  param_.ffn.output_weight.kernel, AType_, n2, 
  ffn_inner, BType_, k2, 
  &beta, 
  output, CType_, n2, 
  computeType_, 
  static_cast<cublasGemmAlgo_t>(cublasAlgo_[3])));
```
#### 3.9.4 add_bias_input
顾名思义，`add_bias_input` 函数包含了 add bias 和 add input 两个操作，前者是 Dense 变换的一部分，后者是残差结构。朴实无华的计算逻辑，没什么好说的，直接上代码。
```cpp
template <typename T>
__global__ 
void add_bias_input_kernel(T* output, const T* input, const T* bias, const int m, const int n)
{
  int id = blockIdx.x * n + threadIdx.x;
  output[id] = output[id] + input[id] + __ldg(&bias[threadIdx.x]);
}

template<OperationType OpType_>
void OpenDecoder<OpType_>::add_bias_input(DataType_* output, const DataType_* input, const int m, const int n)
{
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  add_bias_input_kernel<<<grid, block, 0, param_.stream>>>(output, input, param_.ffn.output_weight.bias, m, n);
}
```
至此 Decoder 模块的计算逻辑已经介绍完毕。下面将对 Decoding 模块的源码进行解读。

## 4 Decoding 模块
根据第 2 节的结构图可以看到，Decoding 模块除了包含 Decoder 模块以外，还有 embedding lookup、position encoding、compute log probability 以及 beam search 等模块，包含了整个解码环节。
### 4.1 调用链
Decoding 模块的核心逻辑封装在 decoding_opennmt.h 文件的 `DecodingOpenNMT` 类中，主要的计算逻辑都在 `forward` 函数里，具体调用链如下：
```cpp
DecodingOpenNMT->DecodingOpenNMT()  // 构造函数
DecodingOpenNMT->forward()
    ->init()
    ->loop for step
        ->embedding_lookup()
        ->sine_position_encoder()
        ->loop for decoder layer
            ->decoder->initialize()
            ->decoder->forward()
        ->decoder->decoder_norm1()
        ->cublasGemmEx
        ->update_logits()
        ->BeamSearch_OpenNMT()
```

### 4.2 DecodingOpenNMT 构造函数
DecodingOpenNMT 构造函数中重要做的是内存分配的工作，要充分理解整体源码，必须要读懂内存分配的逻辑，笔者根据源码绘制了一份内存分布图如下。
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5p5XEtkDKpgcX0VHMoRncOP8sUwwOhqgiaT2eickoaQY3C70sibRq6sEqYtSbB8hXP9A8OFXic6LibnG1A/640?wx_fmt=png)

首先在构造函数内部初始化了 2 个二级指针 `K_cache_` 和 `V_cache`，这个指针是用来存储解码过程中每个 step 对应的输入 tensor 经过 Dense 变换后的 key 矩阵和 value 矩阵，用于 self-attention 的。为什么有两个元素？这是一种**双缓存机制**，在逐 step 解码过程中如果变量有更新，第一个元素作为输入 buffer，第二个元素作为输出 buffer，当前 step 中不在输入 buffer 中直接更新变量，当进入下一个 step 时再把第二个元素作为输入 buffer，第一个元素作为输出 buffer。
```cpp
// 双缓存机制，存储两个指针
K_cache_ = new DataType_ *[2];
V_cache_ = new DataType_ *[2];
```
然后又初始化了 2 个二级指针 `K_mem_cache_` 和 `V_mem_cache_`，这个指针是用来存储 `memory_tensor` 经过 Dense 变换后的 key 和 value。这里虽然 `memory_tensor` 不会变但是每一层 Decoder layer 的权重 kernel 不一样，所以会产生 `decoder_layers` 个 key 和 value。
```cpp
// 存储每一层的指针
K_mem_cache_ = new DataType_ *[decoder_layers_];
V_mem_cache_ = new DataType_ *[decoder_layers_];
```
随后创建了一个 Decoder layer 对象，注意这里是用 `batch_size * beam_width` 当做 `batch_size` 初始化的，具体原因我们后面讲解。
```cpp
hidden_units_ = head_num_ * size_per_head_;
// 注意这里是用batch_size * beam_width当做batch_size初始化decoder
decoder_ = new OpenDecoder<OpType_>(allocator, batch_size * beam_width, memory_max_seq_len, 
                                    head_num, size_per_head, memory_hidden_units);
```
然后就是一系列 buffer size 的计算，用于内存申请和分配的，结合笔者整理的内存分布图可以非常容易的理解。当然这里有一个小 bug，`cache_size` 的计算用的是 `seq_len` 这个变量，这是一个解码长度的变量，因此在申请 `K_cache_` 和 `V_cache_` 这种存储解码端逐个 step 的数据的变量内存时可以用 `cache_size`，但是在申请 `K_mem_cache_` 和 `V_mem_cache_` 这种存储编码端的数据的变量内存的时候就不适用了，源码中 `datatype_buf_size` 直接简单粗暴使用 `cache_size * 6` 把 `K_cache_`、`V_cache_`、`K_mem_cache_` 和 `V_mem_cache_` 的内存一块申请了，当 `memory_seq_len` 大于 `seq_len` 的时候有内存不足导致访问越界的风险。笔者为此阅读了 v2.1 版本的源码，发现作者已经意识到这个 bug 并且已经修复。做法是专门定义一个 `mem_cache_size = batch_size_ * beam_width_ * memory_max_seq_len * hidden_units_`，申请 `K_mem_cache_` 和 `V_mem_cache_` 的内存的时候使用 `mem_cache_size`。  

### 4.3 forward 函数
这个函数的计算逻辑过于复杂，不适合单列一节，大致过程见调用链，当笔者把 forward 内部调用链中子模块讲清楚的时候，forward 也就清晰了。

### 4.4 init 函数
这个函数主要实现以下几个功能：
- `decoding_params.sequence_length` 初始化为 0
- `finished_buf_` 初始化为 false
- `word_ids` 初始化为 start_id
- `cum_log_probs` 将 `beam_id` 为 0 的位置初始化为 0，其他位置初始化为 -lnf
  
```cpp
template <typename T>
__global__ void init_kernel(bool* finished, int* sequence_length, int* word_ids, T* cum_log_probs, const int sentence_id, const int n, const int beam_width)
{
  int tid = threadIdx.x;
  finished[tid] = false;
  sequence_length[tid] = 0;
  word_ids[tid] = sentence_id;
  cum_log_probs[tid] = (T)(tid % beam_width == 0 ? 0.0f: -1e20f);
}

void init(bool* finished, int* sequence_length, int* word_ids, float* cum_log_probs, const int sentence_id, const int batch_size, 
  const int beam_width, cudaStream_t stream)
{
  dim3 grid(1);
  dim3 block(min(1024, batch_size * beam_width));

  assert(batch_size * beam_width <= 1024);
  
  init_kernel<float><<<grid, block, 0, stream>>>(finished, sequence_length, word_ids, cum_log_probs, sentence_id, batch_size * beam_width, beam_width);
}
```
初始化完成之后，就开始逐 step 解码了，下面的函数均在 loop for step 中执行。

### 4.5 embedding_lookup
顾名思义，`embedding_lookup` 函数的功能就是把输入 token 从 `word_id` 映射为词向量，其实现逻辑就是根据 `word_id` 去 `decoding_params.embedding_table` 中查表，把向量存进 `from_tensor[0]` 中。
```cpp
/**
 * @brief 读 word_ids[blockIdx.x] 获取 word_id，embedding_table 的 word_id 行就是词向量
 * 
 * @tparam T 
 * @param embedding_table           [vocab_size, hidden_size]
 * @param word_ids                  [batch_size, beam_width]
 * @param hidden_units 
 * @param from_tensor               [batch_size, beam_width, hidden_size]
 * @return __global__ 
 */
template <typename T>
__global__ void embedding_lookup_kernel(const T* embedding_table, const int* word_ids,
    const int hidden_units, T* from_tensor)
{
  int write_pos = threadIdx.x + blockIdx.x * hidden_units;
  from_tensor[write_pos] = embedding_table[word_ids[blockIdx.x] * hidden_units + threadIdx.x];
}

/**
 * @brief embedding_lookup
 * 
 * @tparam T 
 * @param embedding_table               [vocab_size, hidden_size]
 * @param word_ids                      [batch_size, beam_width]
 * @param from_tensor   from_tensor[0]: [batch_size, beam_width, hidden_size]
 * @param batch_size 
 * @param beam_width 
 * @param hidden_units 
 * @param stream 
 */
template <typename T>
void embedding_lookup(const T* embedding_table, const int* word_ids, T* from_tensor,
  const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream)
{
   dim3 grid(batch_size * beam_width);
   dim3 block(hidden_units);
   assert(hidden_units <= 1024);
   embedding_lookup_kernel<<<grid, block, 0, stream>>>(embedding_table, word_ids, hidden_units, from_tensor);
}
```
通过核函数逻辑可以发现，就是把 `word_ids_buf_` lookup 成了 `from_tensor_[0]`，形状为 `[batch_size, beam_width, hidden_size]`。

### 4.6 sine_position_encoder
我们知道 attention 本身是无序的，每个 token 在计算过程中地位都是等同的，为了表达这种 token 之间的顺序效果，transformer 加入了 Position Encoding layer 给输入 tensor 添加一种位置信息。原始论文中使用下面的公式来实现位置嵌入：  
$$\Large{PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{model} })}$$
$$\Large{PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{model} })}$$
可以看到 position encoding 和 token 位置和 hidden 维度奇偶有关，实际计算过程中我们不考虑 hidden 元素的奇偶，这个维度的顺序本身也没什么意义，所以我们直接前半 `hidden_units` 使用 `sin` 后半部分使用 `cos`，其实 tensorflow api 也是这么简化的，具体如下：
```python
def positional_encoding(length, depth):
  depth = depth/2
  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)
  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 
  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
```
位置编码示意图如下：  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qvlYq7ThmxcyplVPfQ72nxCGN9rsQMFE55rrJLjgkcA5unwCPg6uONqeOia97Dg3j3BBbSg4icEa5w/640?wx_fmt=png)

下面我们来看一下源码，首先给每个元素乘以 $\sqrt{n}$ 达到缩放效果，然后来计算 $1/{10000^{2i / d_{model}}}$ 的部分，源码可能是为了保证精度和数值溢出考虑，使用先取对数再计算指数的策略将其分解为两部分。`log_timescale_increment` 计算的是 $\frac {\ln {10000}} {{d_{model} / 2}}$，这里源码中对 `half_n` 做了略微修正。`inv_timescales` 计算的是 $\exp {(-i  \frac {\ln {10000}} {{d_{model} / 2}})}$，随后再乘以 `step`（也就是 $pos$），得到三角函数里面的内容。最后根据 `tid` 判断应该使用正弦还是余弦然后将计算结果加在 tensor 上即可。
```cpp
template<typename T>
__global__
void sine_position_encoder_kernel(T* output, int step, int n){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  float half_n = (float)n / 2.;

  // input = input * hidden_dim**0.5
  output[bid * n + tid] = output[bid * n + tid] * (T)sqrtf(float(n));

  float log_timescale_increment = __logf(10000) / (half_n - 1.f);
  float inv_timescales = __expf( (tid % (int)half_n) * -1 * log_timescale_increment );
  float scaled_time = inv_timescales * step;
  
  T encoding_val = (tid < half_n) ? (T) __sinf(scaled_time) : (T) __cosf(scaled_time);
  output[bid * n + tid] = output[bid * n + tid]  + encoding_val;
}

/**
 * @brief position encoding
 * 
 * @tparam T 
 * @param output              [m, hidden_units]
 * @param step 
 * @param m                   batch_size * beam_width
 * @param n                   hidden_units
 * @param stream 
 */
template<typename T>
void sine_position_encoder(
  T* output,
  int step,
  int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  sine_position_encoder_kernel<T><<<grid, block, 0, stream>>>(output, step, n);
}
```

### 4.7 执行 decoder layer
循环执行 decoder layer 计算，前一次的计算输出 tensor 作为后一次的输入 tensor，这里源码使用双缓存机制，定义了两个变量 `from_id` 和 `out_id` 根据循环次数的奇偶情况调整 tensor 入参出参。有一点需要注意 decoder layer 在 `forward` 前必须 `initialize` 把对应层的权重参数传进去，否则会导致 decoder layer 的权重未初始化计算错误。

### 4.8 decoder 计算结果归一化
调用 `decoder_norm1` 对 decoder layer 的计算结果进行归一化处理，函数逻辑前面介绍过这里不再赘述。

### 4.9 计算 logits
在深度学习中，logits 的计算一般是通过一个 Dense 层加一个 softmax 激活（二分类中使用 sigmoid，属于 softmax 的特化版本）来实现，源码中通过一个 cuBLAS API 调用和一个 `update_logits` 实现。cuBLAS API 实现矩阵乘法，`add bias 和 softmax` 放在 `update_logits` 中实现，下面重点介绍 `update_logits` 的逻辑。
```cpp
/**
 * @brief 
 * 
 * @tparam T 
 * @param logits                          [batch_size, beam_width, vocab_size]
 * @param bias                            [vocab_size,]
 * @param end_id 
 * @param finished                        [batch_size, beam_width]
 * @param n                               vocab_size
 * @return __global__ 
 */
template <typename T>
__global__ void update_logits_kernel(T* logits, const T* bias, const int end_id, const bool* finished, const int n)
{
  // grid_size = batch_size * beam_width
  int bid = blockIdx.x;
  bool finish = finished[bid];
  int offset = bid * n;

  float max_val = -1 * FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;
  // tid 对应的其实就是 word_id
  for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
  {
    if(finish)
      logits[offset + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
    else
      logits[offset + tid] += bias[tid];
    max_val = max(max_val, logits[offset + tid]);
  }

  max_val = blockReduceMax<float>((float)max_val);
  if(threadIdx.x == 0)
    s_max_val = max_val;
  __syncthreads();

  float sum_val = 0.0f;
  for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
  {
    logits[offset + tid] = __expf((float)logits[offset + tid] - s_max_val);
    sum_val += (float)logits[offset + tid];
  }

  sum_val = blockReduceSum<float>(sum_val);
  if(threadIdx.x == 0)
    s_sum_val = sum_val;
  __syncthreads();

  for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
  {
    logits[offset + tid] = logf((float)logits[offset + tid] / s_sum_val);
  }
}

void update_logits(float* logits, const float* bias, const int end_id, const bool* finished, 
  const int m, const int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
  update_logits_kernel<float><<<grid, block, 0, stream>>>(logits, bias, end_id, finished, n);
}
```
通常我们的词典大小 `vocab_size` 远大于 `1024`，所以 `block_size` 绝大多数都取 `1024`，一个 block 处理一行元素的计算，一个线程会处理多个元素，步长为 `blockDim.x`。第一个循环体内包含 3 个计算任务：首先判断当前 step 的 finish flag，若为 true 就把 `end_id` 的那个分量的 logit 直接设置为最大值，否则就正常 add bias，最后计算当前线程处理的 logit 的最大值 `max_val`。然后通过块内规约求出整个 block 内的 `max_val` 的最大值，这也是整行 `vocab_size` 个元素的最大值，存进共享内存 `s_max_val` 中。第二、三个循环体分别完成了求指数和以及除以指数和的任务，**最终计算结果取对数就是 LogSoftmax 结果**。

### 4.10 BeamSearch_OpenNMT
在 Decoding 过程中，模型的输出是一个 step 一个 step 依次获得的，而且前面 step 的结果还会影响后面 step 的结果。也就是说，每一个 step，模型给出的都是基于历史生成结果的条件概率。为了生成完整的序列，需要一个额外动作来融合模型多个 step 的输出，而且使得最终得到的序列的每一步条件概率连乘起来最大。  
在生成任务中，每一个 step 可能的输出种类称为字典大小（`vocab_size`），进行 T 步随机的生成可能获得的结果总共有 $vocab\_size ^ T$ 种。拿中文文本生成来说，`vocab_size` 的值大约是 `5000-6000`，即常用汉字的个数。在如此大的基数下，遍历整个生成空间寻找最佳序列是不现实的。  
基于上述背景，我们首先想到的策略是逐帧取最大值，也就是**贪心搜索**，即每一个时间步都取出一个条件概率最大的输出，再将从开始到当前步的结果作为输入去获得下一个时间步的输出，直到模型给出生成结束的标志。优点很明显，这样做将原来指数级别的求解空间直接压缩到了与长度线性相关的大小。缺点同样很明显，丢弃了绝大多数的可能解，这种关注当下的策略无法保证最终得到的序列概率是最优的。  
beam search（**集束搜索**）是对贪心搜索一个改进。思路也很简单，在每一个 step，不再只保留当前分数最高的 `1` 个输出，而是保留 `beam_width` 个。当 `beam_width = 1` 时集束搜索就退化成了贪心搜索。  
下图是一个实际的例子，每个时间步有 ABCDE 共 5 种可能的输出，即，图中的 `beam_width = 2`，也就是说每个 step 都会保留到当前步为止条件概率最优的 2 个序列。
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5ooOUuVaG3PcLGwd2mKVw7JzLN4YbWTP5Q4HGco6GF9jCNX0at7Vbar5DaKNc0rDIRNed77IibaRBA/640?wx_fmt=png)

可以发现，beam search 在每一步需要考察的候选人数量是贪心搜索的 `beam_width` 倍，因此是一种牺牲时间换效果的折中方法。源码中 `BeamSearch_OpenNMT` 函数内部分别调用了 4 个函数，下面进行逐一讲解。  
```cpp
/**
 * @brief beam search
 * 
 * @tparam T 
 * @param log_probs           logits: [batch_size, beam_width, vocab_size]
 * @param cum_log_probs               [batch_size, beam_width]
 * @param finished                    [batch_size, beam_width]
 * @param key_cache               2 * [batch_size, beam_width, seq_len, hidden_units]
 * @param value_cache 
 * @param parent_ids 
 * @param sequence_length 
 * @param word_ids                    [batch_size, beam_width]
 * @param ids 
 * @param output_ids 
 * @param batch_size 
 * @param beam_width 
 * @param vocab_size 
 * @param hidden_dim 
 * @param step 
 * @param cache_size 
 * @param decoder_layers 
 * @param stream 
 * @param end_id 
 * @param finished_count 
 */
template <typename T>
void BeamSearch_OpenNMT(
    float *log_probs, float *cum_log_probs, bool *finished,
    T **key_cache, T **value_cache,
    int *parent_ids,
    int *sequence_length,
    int *word_ids,
    int *ids,
    int *output_ids,
    const int batch_size, const int beam_width,
    const int vocab_size, const int hidden_dim, const int step,
    const int cache_size, const int decoder_layers, cudaStream_t stream,
    const int end_id, 
    int *finished_count)
{
  /* adding cum_log_probs to log_probs */
  broadcast_kernelLauncher(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, stream);

  /*Use two round kernels to pick the topK values for each batch */
  topK(log_probs, ids, batch_size, beam_width, vocab_size, stream);

  update(log_probs, cum_log_probs, ids, finished, 
        parent_ids, sequence_length, word_ids, output_ids,
        batch_size, beam_width, vocab_size, stream, 
        end_id, finished_count);

  update_KV_cache<T>(key_cache, value_cache, parent_ids, batch_size, 
                    beam_width, hidden_dim, step, cache_size, 
                    decoder_layers, stream);
}
```
#### 4.10.1 broadcast_kernelLauncher
beam search 的计算依据就是整个序列的条件概率，也就是说要把每个 step 的概率连乘起来，所以对于当前 step 来说各 beam 下每个 word 的概率首先应该乘以前面所有 step 组成序列的累计概率，由于我们这里的概率值在 `update` 函数中计算的是 log 值，所以把这里把连乘换成累加。  
`cum_log_probs` 的形状是 `[batch_size, beam_width]`，表示每个 beam 下的累计概率，这里给 `log_probs` 加上累计概率之后，就表示一个 batch 中第 `batch_id` 个样本第 `beam_id` 个 beam 的第 `word_id` 个 word 作为当前 step 下输出 token 的概率。核函数计算逻辑非常简单，可以直接看代码。

#### 4.10.2 topK
对于一个样本而言拿到 `log_probs` 后我们就得到了 `beam_width` 个分支下共 `beam_width * vocab_size` 条路径的概率，按照 beam search 的计算思想，我们需要找到这 `beam_width * vocab_size` 个路径中概率最大的 `beam_width` 个路径，这是一个求 topK 的问题。由于 `vocab_size` 数值比较大为了保障效率，源码通过两轮 topK 操作求出 topK。
```cpp
template <typename T>
__global__
void topK_kernel(const T* log_probs, int* ids, const int batch_size, const int N, const int K)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val, max_val;
  __shared__ float s_max_val;
  for(int ite = 0; ite < batch_size; ++ite)
  {
    bool choosed = false;
    val = (tid < N ) ? (float)log_probs[ite * N + tid] : -1e20f;
    
    for(int kids = 0; kids < K; ++kids)
    {
      max_val = blockReduceMax<float>(val);
      
      if(threadIdx.x == 0)
        s_max_val = max_val;
      __syncthreads();

      if(s_max_val == val && !choosed && tid < N) 
      {
        ids[ite * gridDim.x * K + blockIdx.x * K + kids] = tid + ite * N;
        val = -1e20f;
        choosed = true;
      }
    }
  }
}

/**
 * @brief for each batch, get the final TopK values out from grid.x * K values
 * 
 * @tparam T 
 * @param log_probs             [batch_size, beam_width, vocab_size]
 * @param ids                   [batch_size, N]
 * @param batch_size 
 * @param N                     gridDim.x(1st) * beam_width
 * @param K                     beam_width
 * @param id_offset             beam_width * vocab_size
 * @return __global__ 
 */
template <typename T>
__global__
void topK_kernel_2nd(const T* log_probs, int* ids, const int batch_size, const int N, const int K, const int id_offset)
{
  int tid = threadIdx.x;
  float val, max_val;
  __shared__ float s_max_val;
  __shared__ int beam_index;
  __shared__ int ids_before_sort[16];

  for(int ite = 0; ite < batch_size; ++ite)
  {
    bool choosed = false;
    const int id = (tid < N) ? ids[ite * N + tid] : -1;
    val = (tid < N) ? (float)log_probs[id] : -1e20f;

    __syncthreads();

    if(tid == 0) beam_index = 0;
    if(tid < 16) ids_before_sort[tid] = -1;
    
    __syncthreads();
    while(beam_index < K){
      int begin_beam_index = beam_index;
      max_val = blockReduceMax<float>(val);
      if(threadIdx.x == 0){
        s_max_val = max_val;
      }
      __syncthreads();
      if(s_max_val == val && !choosed && id != -1)
      {
        int id_offset_ = atomicAdd(&beam_index, 1);
        ids_before_sort[id_offset_] = id;
        val = -1e20f;
        choosed = true;
      }
      __syncthreads();
     
      // simply sort the ids
      if(threadIdx.x == 0 && beam_index - begin_beam_index > 1){
        for(int i = begin_beam_index; i < beam_index; i++){
          for(int j = i; j < beam_index; j++){
            if(ids_before_sort[j] < ids_before_sort[i]){
              int tmpid = ids_before_sort[j];
              ids_before_sort[j] = ids_before_sort[i];
              ids_before_sort[i] = tmpid;
            }
          }
        }
      }
    }
    __syncthreads();
    if(tid < K) ids[ite * K + tid] = ids_before_sort[tid];
    __syncthreads();
  }
}

void topK(const float* log_probs, int* ids, const int batch_size, const int beam_width, const int vocab_size,
  cudaStream_t stream)
{
  int N = beam_width * vocab_size;
  dim3 block(1024);
  dim3 grid((N - 1) / block.x + 1);
  /* First round topK, for each batch, get grid.x * K values */
  topK_kernel<float><<<grid, block, 0, stream>>>(log_probs, ids, batch_size, N, beam_width);
  /*Second round, for each batch, get the final TopK values out from grid.x * K values. */
  topK_kernel_2nd<float><<<1, block, 0, stream>>>(log_probs, ids, batch_size, beam_width * grid.x, beam_width, N);
}
```

第一轮 topK 操作将 `beam_width * vocab_size` 个路径划分到每个 block  中计算，取 block_size 为 `1024`，对于一个样本来说，每个线程只处理一种可能路径，每个 block 内部求出一个 topK，所以最终计算完成后共有 `grid_size_1st` 个 topK。  
核函数中首先是一轮针对 `batch_size` 的循环，表示一个线程内部会处理多个样本，然后取出当前线程对应的路径的概率 `val`，然后开始求 topK。源码中求 topK 的思路非常之朴实无华，循环 K 次，每次块内规约取最大值，然后给当前最大值赋一个极小值防止干扰。取到最大值后，就把最大值所在的位置存入 `ids` 中，注意这里的位置包含 3 个信息：`batch_id`、`beam_id`、`word_id`，分别对应 `log_probs` 的三个维度，`ids` 的形状为 `[batch_size, grid_size_1st, beam_width]`。
第二轮 topK 操作将 `grid_size_1st * beam_width` 个路径进一步缩小到 `beam_width` 个路径，本轮计算全部在一个 block 内完成，取 block_size 为 `1024`，对于一个样本来说，每个线程只处理一种可能路径。说实话这个核函数实现过程过于复杂，笔者看了几遍都不能完全理解，笔者的想法是完全可以复用第一轮 topK 的核函数进行计算，所以以下的解读仅代表笔者本人揣测，如有错误请读者评论或私信指出。  
核函数内部定义了一个共享内存数组 `ids_before_sort` 用来临时存储 topK 的位置，至于数组大小为什么是 `16`，笔者猜可能是目前 `beam_width` 最大支持取 `16`，但是笔者没有在任何官方声明里面看到这个信息。然后是针对 `batch_size` 的循环，表示一个线程内部会处理多个样本，然后取出当前线程对应的路径的概率 `val` 和 word 的位置 `id`。再来一轮循环，每次块内规约取最大值，然后给最大值赋一个极小值，把最大值的位置信息 `id` 存入 `ids_before_sort`，这里还使用了原子操作 `atomicAdd`，猜测是为了防止有两个 word 的概率一样大的情况下可能由于内存读写竞争导致计算错误。每轮选出最大值后还根据 word 的位置给排了个序，咱也不知道啥意图。。。最后把 topK 的位置信息存入 `ids` 中，注意这里的位置信息仍然是 `batch_id`、`beam_id`、`word_id`，不过这时候 `ids` 里面只有 `[batch_size, beam_width]` 范围内的元素是有效的。


#### 4.10.3 update
`update` 函数主要是对累计概率、序列长度、解码 token 等信息根据本次 beam search 的结果进行更新。更新过程全部在 1 个 block 内完成，`block_size` 取 `batch_size * beam_width`，每个线程内部处理一个样本的一条路径。 
```cpp
template <typename T>
__global__
void update_kernel(T* log_probs, T* cum_log_probs, 
                  int* ids, bool* finished, 
                  int* parent_ids, int* sequence_length, 
                  int* word_ids, int* output_ids, 
                  const int batch_size, const int beam_width, 
                  const int vocab_size, const int end_id, 
                  int* finished_count)
{
  // grid_size = 1, block_size = batch_size * beam_width
  int tid = threadIdx.x;
  sequence_length[tid] = finished[tid] ? sequence_length[tid] : sequence_length[tid] + 1;

  int beam_id = ids[tid];
  beam_id /= vocab_size;
  int word_id = ids[tid];
  word_id %= vocab_size;

  cum_log_probs[tid] = log_probs[ids[tid]];
  sequence_length[tid] = sequence_length[beam_id];
  finished[tid] = word_id == end_id ? 1 : 0;
  parent_ids[tid] = beam_id;
  word_ids[tid] = word_id;
  output_ids[tid] = word_id;

  // TODO use reduce sum to compute how many sentence are finished
  // int fi = finished[tid]
  // int total_finish = reduceSum(fi);
}

void update(float* log_probs, float* cum_log_probs, 
            int* ids, bool* finished, 
            int* parent_ids, int* sequence_length,
            int* word_ids, int* output_ids, 
            const int batch_size, const int beam_width, 
            const int vocab_size, cudaStream_t stream, 
            const int end_id, int* finished_count)
{ 
  dim3 grid(1);
  dim3 block(batch_size * beam_width);
  assert(block.x <= 1024);
  update_kernel<float><<<grid, block, 0, stream>>>(log_probs, cum_log_probs, ids, 
                                                  finished, parent_ids, sequence_length,
                                                  word_ids, output_ids, batch_size, 
                                                  beam_width, vocab_size, end_id, 
                                                  finished_count);
}
```

核函数内首先对 `sequence_length` 进行更新，如果 `finished` 标识为 `false`，`sequence_length` 长度增加 `1`，否则认为当前 step 当前 beam 已经终止了。然后计算了几个索引的值，`beam_id` 表示当前线程处理的 topK 路径是属于哪一个 beam。`word_id` 表示当前线程处理的路径对应的 word 在词表中的位置。首先将 `cum_log_probs` 更新到最新，也就是取之前计算的 topK 条路径对应的概率。更新 `sequence_length` 为 topK 路径对应的 beam 的长度。根据 topK 对应的 word_id 是否为 `end_id` 更新 `finished` 标识，这其实就标识下一轮 step 的输入 token 是否是 end_token。更新 `parent_ids` 为 `beam_id`，标识当前这个 topK 路径是从哪个 beam 经过的，说白了这个变量存储着下一轮 beam 中每个路径是从上一轮哪个 beam 经过的。`word_ids` 和 `output_ids` 存储着本轮输出的 topK 个 word 在词表中的位置，这里的 `word_id` 在下一轮 step 经过 embedding lookup 之后就变成了了 `from_tensor`。

#### 4.10.4 update_KV_cache

```cpp
template <typename T>
__global__ void update_KV_cache_kernel(
  T* key_src_cache, T* key_tgt_cache,
  T* value_src_cache, T* value_tgt_cache,
  const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim, const int cache_size, const int step, const int decoder_layers)
{
  // grid_size = decoder_layers * batch_size * beam_width * step,  block(min(1024, hidden_dim))
  int layer_id = blockIdx.x / batch_size / beam_width / step;
  int batch_id = (blockIdx.x % (batch_size * beam_width * step)) / (beam_width * step);
  int beam_id = (blockIdx.x % (beam_width * step)) / step;
  int step_id = blockIdx.x % step;

  int hidden_id = step_id * batch_size * beam_width * hidden_dim + 
    beam_ids[batch_id * beam_width + beam_id] * hidden_dim;

  int tgt_hidden_id = step_id * batch_size * beam_width * hidden_dim + 
    batch_id * beam_width * hidden_dim + beam_id * hidden_dim;

  T* key_src_ptr = key_src_cache + layer_id * cache_size;
  T* key_tgt_ptr = key_tgt_cache + layer_id * cache_size;
  T* value_src_ptr = value_src_cache + layer_id * cache_size;
  T* value_tgt_ptr = value_tgt_cache + layer_id * cache_size;

  for(int tid = threadIdx.x; tid < hidden_dim; tid += blockDim.x)
  {
    key_tgt_ptr[tgt_hidden_id + tid] = key_src_ptr[hidden_id + tid];
    value_tgt_ptr[tgt_hidden_id + tid] = value_src_ptr[hidden_id + tid];
  }
}

template <typename T>
void update_KV_cache(T** key_cache, T** value_cache, const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim,
  const int step, const int cache_size, const int decoder_layers, cudaStream_t stream)
{
  dim3 grid(decoder_layers * batch_size * beam_width * step);
  dim3 block(min(1024, hidden_dim));

  int src_id = step & 0x1;
  int tgt_id = 1 - src_id;

  update_KV_cache_kernel<<<grid, block, 0, stream>>>(
    key_cache[src_id], key_cache[tgt_id],
    value_cache[src_id], value_cache[tgt_id],
    beam_ids, batch_size, beam_width, hidden_dim, cache_size, step, decoder_layers);
}
```
前面 3.5.2.3 节中讲过，对于下一轮 step 的 query 来说 self-attention 的 key 是上一轮 step 的输入 token 对应的 tensor 变换后的结果。由于我们每一轮 beam search 取 topK 会打乱顺序，直观上咱们并不知道下一轮 topK 分别来源于上一轮哪个 beam，这时候我们就要用上 `parent_ids` 了，根据 `parent_ids` 获取上一轮 step 经过的 beam。前面讲过，在 Decoder layer 计算的过程中会把当前 beam 的 token 对应的 key 和 value 计算好存入 `K_cache_` 和 `V_cache_` 中，现在我们直接根据 `beam_id` 去取值就可以了，基于双缓存机制，从 `key_cache[src_id]` 中取值更新 `key_cache[tgt_id]`。这块稍微复杂的地方就是 `hidden_id` 的计算逻辑，不理解的读者可以结合笔者的这段话多读几遍。

### 4.11 提前终止判定
每一轮 step 结束后判断 `finished_buf` 中如果全部都是 `true`，就提前终止。这块逻辑比较简单，源码通过一次内存交换后在主机端完成。
```cpp
// Find a better method to check the is_finished
      cudaMemcpy(h_finished_buf_, finished_buf_, sizeof(bool) * batch_size_ * beam_width_, cudaMemcpyDeviceToHost);
      int sum = 0;
      for(int i = 0; i < batch_size_ * beam_width_; i++){
        sum += (int)h_finished_buf_[i];
      }
      if(sum == batch_size_ * beam_width_) break;
```

## 5 小结
至此，整个 Faster Transformer v2.0 源码逻辑均已解读完毕，总结如下：
- 访存带宽方面，建议加入向量化访问的机制，可以有效提升性能。
- 发现一个小问题，源码作者经常会使用一些不必要的共享内存，比如经常存一些 `s_max`、`s_sum` 这种变量，其实在规约之后线程内部已经获取到块内规约值了，又不涉及线程之间的通信，没有必要舍弃更快的寄存器用共享内存，这块算是官方源码作者的一个编程小习惯。
- 感觉 topK 的求解过程略微繁琐，应该可以优化。而且这块作者的意图笔者理解不够透彻，直观上感觉部分代码有些冗余，期待熟悉的读者给与指导和解惑。
- 源码 beam search 的部分，需要细读才能理解，也是整个源码中较为难懂的部分。
- 申请 `K_mem_cache_` 和 `V_mem_cache_` 内存的时候有个小 bug，官方已在 v2.1 版本修复。