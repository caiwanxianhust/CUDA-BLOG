#! https://zhuanlan.zhihu.com/p/646999661
# 【CUDA编程】cuBLAS中的列主序问题
**写在前面**：距离上次更新文章已经过去了快5个月，这段时间笔者主要在学习Java语言、Spring MVC等相关的内容，补充一下知识盲区，因此荒废了CUDA编程。至于为什么没有更新Java相关的文章，是因为Java太热门了，文章博客太多，一时不知道写什么。后面笔者将继续学习并更新CUDA相关的高性能计算内容。及时当勉励，岁月不待人！

## 1 cuBLAS简介
cuBLAS 库是 BLAS 在 CUDA 运行时的实现。BLAS 全称 basic linear algebra subprograms，即基本线性代数子程序。顾名思义，主要用来进行线性代数的高性能计算，这是 Nvidia 提供的官方库。  

## 2 问题背景
cuBLAS 早期是通过 Fortran 语言实现的并且在 CPU 中运行，所以为了一定程度上兼容 Fortran 环境，后来的各种实现都带有显著的 Fortran 风格。例如，在 cuBLAS 中矩阵在内存中的存储是列主序（column major），而我们在 C、C++ 中一般是按行主序（row major）存储矩阵，这就给实际应用带来了一个小问题。

### 2.1 行主序和列主序
简单理解：行主序中，在同一行的元素在内存中是相邻的；列主序中，同一列的元素在内存中是相邻的。
考虑矩阵
$$
\left(
\begin{matrix}
    0 & 1 & 2  \\
    3 & 4 & 5  \\
\end{matrix}
\right)
\tag{1}
$$
在行主序的约定下，元素在内存中的顺序为 $\{ 0, 1, 2, 3, 4, 5 \}$；在列主序的约定下，其元素在内存中的顺序为 $\{ 0, 3, 1, 4, 2, 5 \}$。

### 2.2 矩阵乘法
我们考虑这样一个矩阵乘法$A \cdot B = C$，其中$M=2,N=3,K=2$：
$$
\left(
\begin{matrix}
    0 & 1  \\
    2 & 3  \\
\end{matrix}
\right)
\left(
\begin{matrix}
    0 & 1 & 2  \\
    3 & 4 & 5  \\
\end{matrix}
\right)
=
\left(
\begin{matrix}
    3 & 4 & 5  \\
    9 & 14 & 19  \\
\end{matrix}
\right)
\tag{2}
$$
如果要使用cuBLAS库计算上述矩阵乘法，前面说过cuBLAS中矩阵都是按列主序存储的，那么我们的矩阵 $A、B$ 其元素在内存中的顺序应该是什么样的呢？
## 3 cuBLAS传参
### 3.1 主机端按列主序存储矩阵元素
显然，对于矩阵 $A(2\times2)$ 其元素顺序应该是 $\{ 0, 2, 1, 3 \}$，对于矩阵 $B(2\times3)$ 其元素顺序应该是 $\{ 0, 3, 1, 4, 2, 5 \}$。具体地可以用如下传参调用 `cublasSgemm` 函数。  
定义 $A(m,k), B(k,n)$ ：
```cuda
cublasSgemm(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m, n, k,
            &alpha,
            d_A,      // 设备上的A矩阵，m * k * sizeof(float)
            m,        // A的leading_edge 是 m
            d_B,      // 设备上的B矩阵，k * n * sizeof(float)
            k,        // B的leading_edge 是 n
            &beta, d_C,
            m);
```

### 3.2 主机端按行主序存储矩阵元素
大多数情况下，矩阵数据来自于主机端，如果主机端的矩阵元素是按行主序存储的，即矩阵 $A(2\times2)$ 元素顺序是 $\{ 0, 1, 2, 3 \}$，矩阵 $B(2\times3)$ 元素顺序是 $\{ 0, 1, 2, 3, 4, 5 \}$。我们直接按3.1中的传参方式调用 `cublasSgemm` 函数，那么实际计算的将会是：
$$
\left(
\begin{matrix}
    0 & 2  \\
    1 & 3  \\
\end{matrix}
\right)
\left(
\begin{matrix}
    0 & 2 & 4  \\
    1 & 3 & 5  \\
\end{matrix}
\right)
=
\left(
\begin{matrix}
    2 & 6 & 10  \\
    3 & 11 & 19  \\
\end{matrix}
\right)
\tag{3}
$$
这显然不是我们想要的结果，那么在不改变主机端矩阵元素存储顺序的前提下如何调用cuBLAS库函数实现矩阵乘法。 

#### 3.2.1 利用矩阵转置乘法公式
根据行主序和列主序的定义可以发现，其实两者互为转置。即对于行主序矩阵 $A_{row}(N,K)$，在列主序的情况下按照 $(K,N)$ 存储，记为 $A_{col}(K,N)$，有$A_{col}^T(K,N) = A_{row}(N,K)$ 。也就是说主机端按行主序存储的矩阵 $A$，在 cuBLAS 中按列主序解读后会变成其转置矩阵 $A^T$，那么问题就演变成了如何通过转置矩阵计算原矩阵乘法结果。  
有了转置关系，再计算矩阵乘法就容易了，根据矩阵乘法公式 $C = A \cdot B = (B^T \cdot A^T)^T$ 可知，我们调用 `cublasSgemm` 函数计算 $B^T \cdot A^T$ 即可。  
定义 $A(m,k), B(k,n)$，则 $A^T(k,m), B^T(n,k)$，传参方式如下：
```cuda
cublasSgemm(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n, m, k,
            &alpha,
            d_B,      // 设备上的BT矩阵，n * k * sizeof(float)
            n,        // BT的leading_edge 是 n
            d_A,      // 设备上的AT矩阵，k * m * sizeof(float)
            k,        // AT的leading_edge 是 k
            &beta, d_C,
            n);
```
最后计算得到的结果 $d\_C$ 依然是列主序的，即矩阵 $C$ 的转置矩阵 $C^T$，那么刚好我们用 `cublasGetVector` 直接把 $d\_C$ 拷贝到主机端按照行主序解读，相当于又做了一次转置操作，在主机端得到的就是正确的结果。
#### 3.2.2 利用API提供的转置参数
`gemm` 接口中提供了 `transpose` 参数用于在计算前对矩阵进行转置操作，因此我们可以使用 `transpose` 参数将列主序存储的矩阵转置回原来的样子，具体传参方式如下：
```cuda
cublasSgemm(handle,
            CUBLAS_OP_T,
            CUBLAS_OP_T,
            m, n, k,
            &alpha,
            d_A,    // m*k
            k,
            d_B,    // k*n
            n,
            &beta,
            d_C,    // m*n
            m);
```
将 `transpose` 参数由 `CUBLAS_OP_N` 调整为 `CUBLAS_OP_T`，这样计算出的结果就是矩阵 $C$，但由于它在设备端是按列主序存储的，所以如果直接拷贝到主机端，其元素顺序需要特殊注意，以 `3.2` 中的矩阵为例，其元素存储顺序为 $\{ 3, 9, 4, 14, 5, 19 \}$ 。

## 4 小结
- 在 cuBLAS 中矩阵在内存中的存储是列主序（column major），与 C\C++ 的行主序存储有所不同，在使用 cuBLAS 库时需要注意。
- 行主序与列主序的关系是互为转置。
- 在计算矩阵乘法时有三种思路处理列主序问题：1）在主机端手工转置；2）利用矩阵转置乘法公式；3）使用 `gemm` 参数自动转置。

## 5 附录
实例代码如下：  
main.cu
```cuda
#include "error.cuh"
#include <stdio.h>
#include <cublas_v2.h>


namespace {
    int M = 2;
    int K = 2; 
    int N = 3;
    int MK = M * K;
    int KN = K * N;
    int MN = M * N;
}

void printMatrix(int R, int C, double *A, const char *name) {
    printf("%s = \n", name);
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            printf("%10.6f", A[r * C + c]);
        }
        printf("\n");
    }
    printf("\n");
}

void gemm(const double* g_A, const double* g_B, double* g_C, int method) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0;
    double beta = 0.0;
    switch (method)
    {
    case 0:
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
        &alpha, g_B, N, g_A, K, &beta, g_C, N);
        break;
    case 1:
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
        &alpha, g_A, K, g_B, N, &beta, g_C, M);
        break;
    default:
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
        &alpha, g_B, N, g_A, K, &beta, g_C, N);
        break;
    }
    
    cublasDestroy(handle);
}

int main() {

    double *h_A = new double[MK];
    double *h_B = new double[KN];
    double *h_C = new double[MN];
    for (int i=0; i<MK; ++i) h_A[i] = i;
    printMatrix(M, K, h_A, "A");
    for (int i=0; i<KN; i++) h_B[i] = i;
    printMatrix(K, N, h_B, "B");
    for (int i=0; i<MN; ++i) h_C[i] = 0;
    
    double *g_A, *g_B, *g_C;
    CHECK(cudaMalloc((void **)&g_A, sizeof(double) * MK));
    CHECK(cudaMalloc((void **)&g_B, sizeof(double) * KN));
    CHECK(cudaMalloc((void **)&g_C, sizeof(double) * MN));
    
    cublasSetVector(MK, sizeof(double), h_A, 1, g_A, 1);
    cublasSetVector(KN, sizeof(double), h_B, 1, g_B, 1);
    cublasSetVector(MN, sizeof(double), h_C, 1, g_C, 1);

    gemm(g_A, g_B, g_C, 0);
    cublasGetVector(MN, sizeof(double), g_C, 1, h_C, 1);
    printMatrix(M, N, h_C, "C = A * B");

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CHECK(cudaFree(g_A));
    CHECK(cudaFree(g_B));
    CHECK(cudaFree(g_C));
    
    return 0;
}
```
error.cuh
```cuda
#pragma once
#include <stdio.h>

#define CHECK(call)                                             \
do {                                                            \
    const cudaError_t errorCode = call;                         \
    if (errorCode != cudaSuccess) {                             \
        printf("CUDA Error:\n");                                \
        printf("    File:   %s\n", __FILE__);                   \
        printf("    Line:   %d\n", __LINE__);                   \
        printf("    Error code:     %d\n", errorCode);          \
        printf("    Error text:     %s\n",                      \
            cudaGetErrorString(errorCode));                     \
        exit(1);                                                \
    }                                                           \
}                                                               \
while (0)
```
