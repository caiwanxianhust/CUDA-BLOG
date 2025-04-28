# 【CUDA编程】online softmax 的 CUDA 实现

**写在前面**：关于 softmax 算子的 CUDA 实现，笔者之前有两篇文章对 OneFlow 框架中的 softmax 源码进行了解析，本文将提供一种 softmax 算子的新的实现方式——online softmax，由于笔者水平有限，文中如有错漏，欢迎各位读者指出。

## 1 softmax

Softmax 操作是深度学习模型中最常用的操作之一，作为一个被广泛使用的算子，其 CUDA Kernel 的实现会影响模型的训练、推理性能。那么如何实现一个高效的 Softmax CUDA Kernel？在这之前我们先来看一下 Softmax 的计算公式。  
定义 `x` 是一个 `n` 维向量，其 Softmax 输出 `y` 也是一个 `n` 维向量，那么有如下计算公式：
$$
y_i = softmax(x_i) = \frac{e^{x_i}}{\sum _{j=0}^{n-1} e^j}，其中i,j=0,1,2...,n-1
$$
从上面的公式可以发现一个问题，当 $x^i$ 为一个较大的正数时，指数运算后 $e^{x_i}$ 将会非常大，从而导致数值溢出，如何解决这个问题呢？  
一般的处理方法是，让每个分量去减掉向量的最大值，这样可以保证取指数后的结果必然在 `0~1` 之间，可以有效避免数值溢出。处理后的公式如下：
$$
y_i = softmax(x_i) = \frac{e^{x_i - x_{max}}}{\sum _{j=0}^{n-1} e^{x_j - x_{max}}}，其中i,j=0,1,2...,n-1
$$

修正后的公式我们称之为 safe softmax，在传统公式的基础上多了一个规约求最大值 $x_{max}$ 的操作，但有效解决了数值溢出问题。

在 safe softmax 的计算过程中，由于存在**规约求最大值 $x_{max}$**、**规约求指数和 $\sum _{j=0}^{n-1} e^{x_j - x_{max}}$**以及指数归一化等 3 步操作，且每一步都依赖前一步的计算结果，因此在大规模矩阵或向量场景下，单行元素由于无法完全放入 shared memory 而需要连续访问 3 次 global memory，这将产生显著的性能瓶颈。

下面我们先来看一下标准 3-pass safe softmax 的 kernel 实现。
```cpp
namespace
{
    constexpr int BLOCK_SIZE = 256;
}

__global__ void softmaxKernel(const float *__restrict__ mat, float *__restrict__ output, const int ncol)
{
    float val;
    float vmax = -FLT_MAX;
    float exp_sum = 1e-10f;

#pragma unroll
    for (int i = threadIdx.x; i < ncol; i += blockDim.x)
    {
        vmax = max(mat[blockIdx.x * ncol + i], vmax);
    }
    __syncthreads();

    vmax = blockAllReduceMax<float>(vmax);

#pragma unroll
    for (int i = threadIdx.x; i < ncol; i += blockDim.x)
    {
        exp_sum += __expf(mat[blockIdx.x * ncol + i] - vmax);
    }
    __syncthreads();

    exp_sum = blockAllReduceSum<float>(exp_sum);

#pragma unroll
    for (int i = threadIdx.x; i < ncol; i += blockDim.x)
    {
        val = __expf(mat[blockIdx.x * ncol + i] - vmax) / exp_sum;
        output[blockIdx.x * ncol + i] = val;
    }
}

void launchSoftmaxKernel(const float *__restrict__ mat, float *__restrict__ output, const int ncol, const int nrow, cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(nrow);
    softmaxKernel<<<grid, block, 0, stream>>>(mat, output, ncol);
}
```

代码结构非常简洁直观，3 个 for 循环分别完成 3 个主要操作：
  
  - 第一个循环，遍历一行元素，规约求出 `vmax`，即 $x_{max}$；
  - 第二个循环，遍历一行元素，规约求出 `exp_sum`，即 $\sum e^{x_j - x_{max}}$；
  - 第三个循环，遍历一行元素，归一化求出 softmax 结果；

## 2 online softmax
从 safe softmax 的 kernel 代码可以发现，在 softmax 计算过程中，访存开销要比计算开销占比更大，因此如果能尽量减少对于低带宽内存的访问（如降低对 global memory 的访问次数），即使稍微增加一些计算指令也是非常值得的。具体来说，我们能否把 3 次访问 global memory 遍历 $x_i$ 降低到 2 次甚至 1 次？上节中我们也分析过，3 个 for 循环是前后依赖的，第 2 个循环依赖第一个循环的计算结果 `vmax`，第 3 个循环依赖第 2 个循环的计算结果 `exp_sum`，如何解除强依赖将两个计算结果融合到一次循环中进行，将是一个重要问题。

这就引出了本文的重点内容——online softmax，借助 online softmax 我们可以实现对前两个 for 循环的融合。所谓 online softmax 就是在原长度为 $N$ 的向量的 softmax 结果的基础上，能够在线地增加新的向量元素，动态地求出此时 $N+1$ 长度向量的 softmax 结果。我们不妨再来看一下 safe softmax 的计算公式：
$$
y_i = softmax(x_i) = \frac{e^{x_i - x_{max}}}{\sum _{j=0}^{n-1} e^{x_j - x_{max}}}，其中i,j=0,1,2...,n-1
$$

如果序列需要增加 $x_N$，那么此时 $x_{max}$（也就是之前代码中的 `vmax`），将有两种结果，要么是前 $N$ 个元素的最大值，要么是 $x_N$，方便起见我们把前 $N$ 个元素的最大值 $x_{max}$ 记为 $m_{N}$，前 $N$ 个元素的 $\sum e^{x_j - x_{max}}$ 记为 $d_{N}$，则有：
$$
m_{N+1} = max(m_N, x_N)，起始索引从 0 开始
$$
而，
$$
\begin{split}
    d_{N+1} &= \sum _{j=0}^{N} e^{x_j - m_{N+1}} \\
            &= \sum _{j=0}^{N-1} e^{x_j - m_{N+1}} + e^{x_N - m_{N+1}} \\
            &= d_N e^{m_N - m_{N+1}} + e^{x_N - m_{N+1}}
\end{split}  
$$
新增元素 $x_N$ 之后，全局最大值可能改变，则对于原来的指数和需要补乘一个系数 $e^{m_N - m_{N+1}}$ 进行归一化调整，如果全局最大值没有改变，则这个系数就等于 1，再加上新元素对应的分量 $e^{x_N - m_{N+1}}$，就是新的 $d_{N+1}$ 了。

更一般地，对于给定长度 $N$ 的向量 $a$，其最大值记为 $m_a$，指数和记为 $d_a$，如果需要新增 $M$ 个元素，把这 $M$ 个元素记为向量 $b$，扩增向量的最大值记为 $m_b$，指数和记为 $d_b$，则整体向量的最大值和指数和我们可以使用如下公式：
$$
\begin{split}
    m &= max(m_a, m_b) \\
    d &= d_a e^{m_a - m} + d_b e^{m_b - m} 
\end{split}
$$

根据上述两个计算公式，就可以把前两个 for 循环融合到一起，实现 online softmax，具体 kernel 逻辑如下：

- 第一个 for 循环，遍历 $x_i$，每个线程处理多个元素，线程内部维护 $m$ 和 $d$ 两个变量，利用 $d_{N+1} = d_N e^{m_N - m_{N+1}} + e^{x_N - m_{N+1}}$ 公式更新 $m$ 和 $d$ 两个变量，完成后相当于每个线程维护一个向量的 $m$ 和 $d$ 
- 利用公式 $d = d_a e^{m_a - m} + d_b e^{m_b - m}$，通过 block 内规约，将每个线程维护的向量的 softmax 结果合并为一行元素的 $m$ 和 $d$
- 第二个 for 循环，遍历 $x_i$，根据整体的 $m$ 和 $d$ 计算一行元素的 softmax 结果。

可见通过 online softmax 仅需要两次遍历即可计算出 softmax 结果，具体代码如下：
```cpp
struct __align__(8) MD_F
{
    float m; // max val
    float d; // exp sum
};

struct MDFOp
{
    __device__ __forceinline__ MD_F operator()(MD_F &a, MD_F &b)
    {
        MD_F ret;
        ret.m = max(a.m, b.m);
        ret.d = a.d * __expf(a.m - ret.m) + b.d * __expf(b.m - ret.m);
        return ret;
    }
};

__global__ void onlineSoftmaxKernel(const float *__restrict__ mat, float *__restrict__ output, const int ncol)
{
    MD_F mdf_tmp, mdf_val;
    mdf_val.d = 0.0f;
    mdf_val.m = -1e20f;

#pragma unroll
    for (int i = threadIdx.x; i < ncol; i += blockDim.x)
    {
        mdf_tmp.m = mat[blockIdx.x * ncol + i];
        mdf_tmp.d = 1.0f;
        mdf_val = MDFOp()(mdf_tmp, mdf_val);
    }
    __syncthreads();

    typedef cub::BlockReduce<MD_F, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ MD_F mdf_total;
    mdf_val = BlockReduce(tempStorage).Reduce(mdf_val, MDFOp());
    if (threadIdx.x == 0)
    {
        mdf_total = mdf_val;
    }
    __syncthreads();

#pragma unroll
    for (int i = threadIdx.x; i < ncol; i += blockDim.x)
    {
        output[blockIdx.x * ncol + i] = __expf(mat[blockIdx.x * ncol + i] - mdf_total.m) / mdf_total.d;
    }
}

void launchOnlineSoftmaxKernel(const float *__restrict__ mat, float *__restrict__ output, const int ncol, const int nrow, cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(nrow);
    onlineSoftmaxKernel<<<grid, block, 0, stream>>>(mat, output, ncol);
}
```

在代码中我们首先定义了一个结构体 `MD_F`，用于存储 `float` 类型的 $m$ 和 $d$ 两个变量，并进行内存对齐确保内存访问效率，为了方便更新 $m$ 和 $d$，针对结构体 `MD_F`，笔者定义了对应的运算符逻辑 `MDFOp`，$m$ 的更新很好理解，$d$ 的更新实际上就是公式 $d = d_a e^{m_a - m} + d_b e^{m_b - m}$，这个公式实际上是通用公式，当向量新增 1 个元素时，$d_b = e^{x_b - x_b}$ 等于 1，于是就退化成公式 $d_{N+1} = d_N e^{m_N - m_{N+1}} + e^{x_N - m_{N+1}}$，所以每次从 `mat` 中读取元素的时候我们会将其 `mdf_tmp.d` 赋值为 `1.0f`。

block 内的规约操作，笔者调用了 CUB 库处理，懒得单独写一个特化的 `blockAllReduce` 函数，所以直接调库处理了。有个地方需要注意，不同于我们自己写的 `blockAllReduce` 函数在函数内部会加入同步操作，CUB API 计算完成后只是将结果存入线程 ID 为 0 的寄存器变量中，需要用户自行存入共享内存变量并进行同步。下面给出自定义的 `blockAllReduce` 函数供各位读者参考。
```cpp
template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) { return max(a, b); }
};

template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) { return a + b; }
};

template <template <typename> class ReduceOp, typename T>
__device__ __inline__ T warpAllReduce(T val)
{
    auto functor = ReduceOp<T>();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val = functor(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}

template <typename T>
__device__ __inline__ T blockAllReduceSum(T val)
{
    __shared__ T shared[32];
    __shared__ T ret;
    int warp_id = (threadIdx.x >> 5);
    int lane_id = (threadIdx.x & 31);

    val = warpAllReduce<SumOp, T>(val);
    if (lane_id == 0)
    {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : (T)(0.0f);
    val = warpAllReduce<SumOp, T>(val);
    if (threadIdx.x == 0)
    {
        ret = val;
    }
    __syncthreads();

    return ret;
}

template <typename T>
__device__ __inline__ T blockAllReduceMax(T val)
{
    __shared__ T shared[32];
    __shared__ T ret;
    int warp_id = (threadIdx.x >> 5);
    int lane_id = (threadIdx.x & 31);

    val = warpAllReduce<MaxOp, T>(val);
    if (lane_id == 0)
    {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : (T)(-FLT_MAX);
    val = warpAllReduce<MaxOp, T>(val);
    if (threadIdx.x == 0)
    {
        ret = val;
    }
    __syncthreads();

    return ret;
}
```

## 3 性能对比
在 NVIDIA RTX 4060 上的测试数据（单位：ms），随着 softmax 序列长度逐渐增长，online softmax 逐步体现出性能优势。

|矩阵尺寸|softmax|online softmax|
|:---:|:---:|:---:|
|[128, 2048]|0.0234944|0.0202918|
|[128, 32728]|0.0781082|0.0609904|
|[128, 65536]|0.422945|0.439531|
|[128, 131072]|1.18813|0.918898|
|[128, 262144]|2.38716|1.84431|
|[128, 2097152]|18.8618|14.5088|
|[128, 4194304]|37.992|29.3278|

## 4 小节
本文介绍了一种 softmax 算子的新的实现方式——online softmax，并给出了相应的 CUDA 代码实现。相比于传统 3-pass safe softmax，online softmax 可以将在求最大值的同时计算出指数和，将 3-pass 缩短为 2-pass，减少了全局内存访问次数，对于序列长度较长的场景，性能优势显著。