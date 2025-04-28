#! https://zhuanlan.zhihu.com/p/713049406
# 【CUDA编程】cuBLAS 库中矩阵乘法参数设置问题

**写在前面**：笔者近期家里事情比较多，每个周末都有一些事情要做，无暇更新文章。本文主要讨论了关于 cuBLAS 库中矩阵乘法参数设置问题，问题来源是笔者自己写的一个 Llama 推理框架中用到了一个变量缓冲区，缓冲区变量会使用矩阵乘法，其中会涉及到 cuBLAS 库中矩阵乘法 lda 等参数设置问题，觉得这个地方有必要拿出来讲清楚，也作为一个记录。

## 1 问题背景
cuBLAS 库是 BLAS 在 CUDA Runtime 的实现。BLAS 全称 basic linear algebra subprograms，即基本线性代数子程序，顾名思义，主要用来进行线性代数的高性能计算，这是 Nvidia 提供的官方库。笔者在早期的一篇文章中也有讨论过使用 cuBLAS 库时需要考虑的列主序问题，但是当时讨论的场景比较简单，本次将结合实际示例进行介绍。

本文要讨论的是一个子矩阵乘法调用 cuBLAS API 时的参数设置问题。所谓子矩阵就是从大矩阵中分割出来的小矩阵，我们知道，矩阵元素在内存中通常是连续存储的，也就是说对于一个形状为 `[m, n]` 的二维矩阵，在内存中一般是以一维的形式连续存储，即相当于一个长度为 `m * n` 的一维数组。那么对于大矩阵而言，假设其形状为 `[rows, cols]`，其在内存中相当于一个长度为 `rows * cols` 的一维数组，但是其子矩阵则不然，假设小矩阵和大矩阵的关系示意图如下：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5pJ09Fq9UibYGicFO9VqKXQWiaOwYsicaoHaR7xfrwme1T7pJUOJRE0SAqdz85ewPVLa2hXt9ibQl9tyyw/640?wx_fmt=png&amp;from=appmsg)

子矩阵常常会因为列数 `n` 小于大矩阵的列数 `cols` 而导致子矩阵不是连续存储的，再加上 cuBLAS 库的列主序问题，在调用 API 的时候参数设置需要特别注意。

## 2 乘法示例
我们以单精度矩阵乘法为例，矩阵初始化方式如下：
```cpp
constexpr int rows = 128;
constexpr int cols = 196;
float h_a[rows * cols];
float h_b[rows * cols];
float h_c[rows * cols];
for (int i=0; i<rows * cols; ++i) {
    h_a[i] = (i % 103) + 0.1;
}
for (int i=0; i<rows * cols; ++i) {
    h_b[i] = (i % 199) + 0.2;
}

int m = 96;
int n = 64;
int k = 32;
```
大矩阵形状均为 `[128, 196]`，两个待计算的小矩阵形状分别为 `[96, 32]` 和 `[32, 64]`，在使用 Python 的 tensorflow 库的情况下进行矩阵运算语法非常简单，具体如下。

```python
sub_fa = fa[:m, :k]
sub_fb = fb[:k, :n]
sub_fc = tf.matmul(sub_fa, sub_fb)
sub_fc

"""
# output:
<tf.Tensor: shape=(96, 64), dtype=float32, numpy=
array([[ 68023.945,  68523.14 ,  69022.34 , ...,  56287.14 ,  56786.34 ,
         53086.64 ],
       [191923.9  , 193133.11 , 194342.3  , ...,  78824.16 ,  80033.34 ,
         79033.64 ],
       [305420.9  , 307340.1  , 309259.3  , ...,  70461.16 ,  72380.34 ,
         74080.64 ],
       ...,
       [ 94090.55 ,  94802.75 ,  95514.945, ...,  66292.74 ,  67004.94 ,
         64115.242],
       [229217.5  , 230639.7  , 232061.92 , ...,  79559.74 ,  80981.945,
         80792.25 ],
       [333444.5  , 335576.72 , 337708.9  , ..., 102920.75 , 105052.945,
         87066.24 ]], dtype=float32)>
"""
```

## 3 cuBLAS API
说回到 CUDA，cuBLAS 库中可用于单精度矩阵乘法计算的 API 主要是两个：一个是 `cublasSgemm` 函数，这也是最常用的；另一个是 `cublasGemmEx` 函数，这个函数不仅仅可用于单精度类型，还支持其他类型，是一个灵活性较强的 API，用户可以传入指定的数据类型枚举值。这两个函数的参数设置区别不大，这里我们主要针对前者的主要参数进行介绍。

```cuda
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, 
                           cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc)
```

- `handle`：cuBLAS 上下文句柄，注意这是一个设备级的句柄，可以在不同的 API 调用时复用。
- `transa`：`A` 矩阵再乘法计算前是否需要转置。
- `transb`：`B` 矩阵再乘法计算前是否需要转置。
- `m`：`A` 矩阵和 `C` 矩阵的行数。
- `n`：`B` 矩阵和 `C` 矩阵的列数。
- `k`：`A` 矩阵的列数和 `B` 矩阵的行数。
- `lda`：`A` 矩阵的主维度。
- `ldb`：`B` 矩阵的主维度。
- `ldc`：`C` 矩阵的主维度。

通常在 c++ 中矩阵是按照行主序在内存中存储的，而 cuBLAS 要求矩阵按列主序存储，这就使得 cuBLAS API 不能直接调用，一般有两种方法：一种是把 `A`、`B` 矩阵在乘法计算前转置，这种方式计算出来的矩阵 `C` 矩阵也是列主序的，不常使用；另一种是利用乘法转置公式 $(AB)^T = B^T A^T$，把 `A`、`B` 矩阵传参时互换，此时相当于计算 `[n, k]` 和 `[k, m]` 两个矩阵的乘法。

关于主维度这个参数，它跟大矩阵的尺寸紧密相关，表示的是**与待计算矩阵尺寸对应的大矩阵的尺寸**，说白了就是按列主序填充矩阵时列与列之间的步长。比如这里我们利用乘法转置公式原理，部分传参如下：
```
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, mat_b, ldb, mat_a, lda, &beta, mat_c, ldc)
```

由于此时待计算矩阵形状分别为 `[n, k]` 和 `[k, m]`，所以 `lda` 应该取矩阵 `B` 的行数 `n` 对应的大矩阵的尺寸，而矩阵 `B` 在原始大矩阵中是 `[k, n]` 的形状，所以 `n` 应该对应 `[rows, cols]` 中的 `cols`。`ldb` 应该取矩阵 `A` 的行数 `k` 对应的大矩阵的尺寸，而矩阵 `A` 在原始大矩阵中是 `[m, k]` 的形状，所以 `k` 应该对应 `[rows, cols]` 中的 `cols`。同理 `ldc` 也为 `cols`，具体函数传参如下：

```cuda
/*
mat_a, mat_b, mat_c is ordered by rowmajor.
*/
void floatGemm(const float *mat_a, const float *mat_b, float *mat_c, int m, int n, int k, int lda, int ldb, int ldc, cudaStream_t stream = 0)
{
    cublasHandle_t handle;
    CHECK_CUBLAS_STATUS(cublasCreate(&handle));

    CHECK_CUBLAS_STATUS(cublasSetStream_v2(handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t start, stop;
     // Allocate CUDA events that we'll use for timing
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));

    CHECK_CUBLAS_STATUS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, mat_b, ldb, mat_a, lda, &beta, mat_c, ldc));

    // Record the stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
    printf("Time = %g ms.\n", msecTotal);

    CHECK_CUBLAS_STATUS(cublasDestroy(handle));
}

void launchFloatGemm(const float *mat_a, const float *mat_b, float *mat_c, int rows, int cols, int m, int n, int k)
{
    float *d_a;
    float *d_b;
    float *d_c;
    
    cudaStream_t master_stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&master_stream));

    CHECK_CUDA_ERROR(cudaMallocAsync((void **)&d_a, sizeof(float) * rows * cols, master_stream));
    CHECK_CUDA_ERROR(cudaMallocAsync((void **)&d_b, sizeof(float) * rows * cols, master_stream));
    CHECK_CUDA_ERROR(cudaMallocAsync((void **)&d_c, sizeof(float) * rows * cols, master_stream));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a, mat_a, sizeof(float) * rows * cols, cudaMemcpyHostToDevice, master_stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b, mat_b, sizeof(float) * rows * cols, cudaMemcpyHostToDevice, master_stream));

    floatGemm(d_a, d_b, d_c, m, n, k, cols, cols, cols, master_stream);

    CHECK_CUDA_ERROR(cudaMemcpyAsync(mat_c, d_c, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost, master_stream));

    if (d_a) CHECK_CUDA_ERROR(cudaFreeAsync(d_a, master_stream));
    if (d_b) CHECK_CUDA_ERROR(cudaFreeAsync(d_b, master_stream));
    if (d_c) CHECK_CUDA_ERROR(cudaFreeAsync(d_c, master_stream));
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(master_stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(master_stream));
}
```

在使用 cublasSgemm 这类 API 之前，需要把数据从主机端移动到设备端，上面我们采用的是 `cudaMemcpy*` API 进行数据拷贝，这是一个 Runtime API，但是 cuBLAS 本身也有 `cublasSetMatrix*` 类 API 专门用具矩阵数据拷贝。在使用这类 API 的时候要特别注意矩阵数据填充是列主序填充的，此外由于这类 API 是专门用于矩阵数据拷贝的，所以其主维度属性在调用 `cublasSetMatrix*` 时就已经传入。

以上面的乘法为例，对于实际的 `A` 矩阵，基于乘法转置公式原理，它其实是作为右矩阵参与乘法计算的，所以其形状应该为 `[k, m]`，主维度是 `k` 对应的大矩阵的尺寸，也就是 `cols`。同理，对于实际的 `B` 矩阵，作为左矩阵参与计算，形状为 `[n, k]`，主维度为 `cols`。

```cuda
// cublasSetMatrixAsync API 中源矩阵和目的矩阵都是按列主序存储，因此这里要把矩阵 A 存成矩阵 AT
CHECK_CUBLAS_STATUS(cublasSetMatrixAsync(k, m, sizeof(float), mat_a, cols, d_a, k, master_stream));
CHECK_CUBLAS_STATUS(cublasSetMatrixAsync(n, k, sizeof(float), mat_b, cols, d_b, n, master_stream));
```

此时 `d_a`、`d_b`、`d_c`在 GPU 内存中已经是连续存储的形式了，所以在调用 cublasSgemm 时需要传入的主维度参数分别为 `k`、`n`、`n`，不再跟大矩阵相关，因为他们在 GPU 内存中已经是“大矩阵”了。完整代码如下：

```cuda
void launchFloatGemm_v2(const float *mat_a, const float *mat_b, float *mat_c, int rows, int cols, int m, int n, int k)
{
    float *d_a;
    float *d_b;
    float *d_c;
    
    cudaStream_t master_stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&master_stream));

    CHECK_CUDA_ERROR(cudaMallocAsync((void **)&d_a, sizeof(float) * m * k, master_stream));
    CHECK_CUDA_ERROR(cudaMallocAsync((void **)&d_b, sizeof(float) * k * n, master_stream));
    CHECK_CUDA_ERROR(cudaMallocAsync((void **)&d_c, sizeof(float) * m * n, master_stream));

    // cublasSetMatrixAsync API 中源矩阵和目的矩阵都是按列主序存储，因此这里要把矩阵 A 存成矩阵 AT
    CHECK_CUBLAS_STATUS(cublasSetMatrixAsync(k, m, sizeof(float), mat_a, cols, d_a, k, master_stream));
    CHECK_CUBLAS_STATUS(cublasSetMatrixAsync(n, k, sizeof(float), mat_b, cols, d_b, n, master_stream));

    floatGemm(d_a, d_b, d_c, m, n, k, k, n, n, master_stream);

    // cublasGetMatrixAsync API 中源矩阵和目的矩阵都是按列主序存储，这里矩阵 d_c 实际是按列存储的矩阵 CT
    CHECK_CUBLAS_STATUS(cublasGetMatrixAsync(n, m, sizeof(float), d_c, n, mat_c, cols, master_stream));
    
    if (d_a) CHECK_CUDA_ERROR(cudaFreeAsync(d_a, master_stream));
    if (d_b) CHECK_CUDA_ERROR(cudaFreeAsync(d_b, master_stream));
    if (d_c) CHECK_CUDA_ERROR(cudaFreeAsync(d_c, master_stream));
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(master_stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(master_stream));
}
```

## 4 小结
本文介绍了子矩阵在进行矩阵乘法场景下，调用 cuBLAS API 时参数设置问题，主要是主维度参数需要特别注意，其表示的是按列主序填充时列与列之间的步长，与矩阵元素在内存中的存储情况有关。
