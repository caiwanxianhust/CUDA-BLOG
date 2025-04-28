#! https://zhuanlan.zhihu.com/p/683341920
![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/GJUG0H1sS5pDpF4vZywoycTziaRPb0TwFWEENK5LPteyqC2nLBCSxNS8Jq0cX9WYXEd599gu3PACDdbVAZEGicrQ/640?wx_fmt=jpeg&amp;from=appmsg)

# 【CUDA编程】使用 cuBLASLt 中的 cublasLtMatmul API 进行矩阵乘法

**写在前面**：本文主要目的是提供一个相对完整的使用 cuBLASLt 的矩阵乘法 API 进行矩阵乘法操作的示例。

## 1 为什么要使用 cuBLASLt 库？

cuBLASLt，全称 cuBLAS Light，顾名思义是一个轻量级的 cuBLAS 库，其中封装了一些新的灵活性强的 API 专门用于一般地矩阵乘法操作（GEMM）。cuBLASLt 库中新增了矩阵数据布局、输入类型、计算类型的等计算要素，使得用户可以通过指定这类参数满足不同的矩阵乘法场景。

cuBLASLt 库不保证支持所有可能的矩阵尺寸大小和参数配置，但是，自 CUDA 12.2 以来，关于 m、n 以及 batch_size 的大小限制问题已在很大程度上得到解决。该库的主要关注点是提供性能最好的内核，当然，这可能有一些隐含的限制需要用户参照文档自行解决。

为什么要用到 cuBLASLt 库？

在日常的 CUDA 程序开发中通常 cuBLAS 库已经足够使用，笔者在此之前也没有使用过 cuBLASLt 库，只是在近期阅读 Faster Transformer v3.0 的源码时，发现 Nvidia 官方源码中利用了 cuBLASLt 及 INT8 Tensor Core 加速矩阵乘法，怀着好奇的目的，笔者学习了一些官方文档中 cublasLtMatmul API 的使用介绍，特此记录而已。

事实上，cuBLAS 库从 11.0 开始底层就尽可能地自动使用 Tensor Core 来加速运算，也就是说 cuBLASLt 库并不是使用 GPU 中 Tensor Core 的唯一途径，用 cuBLAS 也是一样的（不建议自行编写 GEMM Kernel，很难达到官方库的性能）。当矩阵尺寸和指针类型满足一定的内存对齐要求时，使用 Tensor Core 可以获得最佳性能，是否选择使用 Tensor Core，是 cuBLAS 库自行选择的，无需用户显式指定。同样地，cuBLAS 库中也有支持 INT8 矩阵乘法的 API，比如 cublasGemmEx API，有兴趣的读者可以了解。

## 2 cublasLtMatmul API

```cuda
cublasStatus_t cublasLtMatmul(
      cublasLtHandle_t               lightHandle,
      cublasLtMatmulDesc_t           computeDesc,
      const void                    *alpha,
      const void                    *A,
      cublasLtMatrixLayout_t         Adesc,
      const void                    *B,
      cublasLtMatrixLayout_t         Bdesc,
      const void                    *beta,
      const void                    *C,
      cublasLtMatrixLayout_t         Cdesc,
      void                          *D,
      cublasLtMatrixLayout_t         Ddesc,
      const cublasLtMatmulAlgo_t    *algo,
      void                          *workspace,
      size_t                         workspaceSizeInBytes,
      cudaStream_t                   stream);
```

`cublasLtMatmul` 函数实现了下面的矩阵乘法和加法运算：`D = alpha*(A*B) + beta*(C)`，其中 `A`、`B`、`C` 是输入矩阵，`alpha` 、`beta` 作为指定的修正参数。
此函数目前支持就地（in-place）矩阵乘法（`C==D` 和 `Cdesc==Ddesc`）和错位（out-of-place）矩阵乘法（`C!=D`，两个矩阵必须具有相同的数据类型、行数、列数、批处理大小和内存顺序)。在错位的情况下，`C` 的前导维度可以不同于 `D` 的前导维度。

除此之外，在使用该函数时还有一些其他的限制（未来版本可能没有）：
- 只支持 NT 乘法，即矩阵 `A` 和 `C` 必须不转置，矩阵 `B` 必须转置。
- 矩阵 `A`、`C`、`D` 的内存布局为 `CUBLASLT_ORDER_COL32`，而矩阵 B 的内存布局为 `CUBLASLT_ORDER_COL4_4R2_8C`（图灵或安培架构）。

## 3 INT8 矩阵乘法示例
本文要展示的矩阵乘法示例比较简单，计算公式为 `C = A * B`。

### 3.1 矩阵初始化及 Python 实现
前面说过 `cublasLtMatmul` 函数对矩阵的内存布局有要求，所以为了简单起见，我们让矩阵 `A` 和 `B` 的长宽都是 `32` 的倍数。不妨假设 `m`、`n`、`k` 分别为 `96`、`32`、`64`，矩阵 `A` 和 `B` 初始化方式如下：
```python
a = np.zeros(dtype=np.int32, shape=(96, 64))
b = np.zeros(dtype=np.int32, shape=(64, 32))
for i in range(96*64):
    a[i//64, i%64] = i % 64
for i in range(32*64):
    b[i//32, i%32] = i % 32
```
此时矩阵 `A` 和 `B` 形状分别为 `[96, 64]`、`[64, 32]`，具体元素值如下所述。
```python
(array([[ 0,  1,  2, ..., 61, 62, 63],
        [ 0,  1,  2, ..., 61, 62, 63],
        [ 0,  1,  2, ..., 61, 62, 63],
        ...,
        [ 0,  1,  2, ..., 61, 62, 63],
        [ 0,  1,  2, ..., 61, 62, 63],
        [ 0,  1,  2, ..., 61, 62, 63]]),
 array([[ 0,  1,  2, ..., 29, 30, 31],
        [ 0,  1,  2, ..., 29, 30, 31],
        [ 0,  1,  2, ..., 29, 30, 31],
        ...,
        [ 0,  1,  2, ..., 29, 30, 31],
        [ 0,  1,  2, ..., 29, 30, 31],
        [ 0,  1,  2, ..., 29, 30, 31]]))
```

我们使用 numpy 可以很容易的计算出矩阵 `C` 的值
```python
c = np.dot(a, b)
c

#output:
array([[    0,  2016,  4032, ..., 58464, 60480, 62496],
       [    0,  2016,  4032, ..., 58464, 60480, 62496],
       [    0,  2016,  4032, ..., 58464, 60480, 62496],
       ...,
       [    0,  2016,  4032, ..., 58464, 60480, 62496],
       [    0,  2016,  4032, ..., 58464, 60480, 62496],
       [    0,  2016,  4032, ..., 58464, 60480, 62496]])
```

### 3.2 CUDA 实现
下面看一下 CUDA 实现的代码，主要逻辑在 main.cu 文件的 `testlt` 函数中：  
main.cu
```
#include "int8gemm.cuh"


template<typename T>
void printMat(T* mat, int rows, int cols, const char* name) 
{
    std::cout << "*****************************\nthis is Matrix: " << name << std::endl;
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            std::cout << mat[cols * i + j] << "  ";
        }
        std::cout << std::endl;
    }
}


void testlt() 
{
    constexpr int m = 96;
    constexpr int k = 64;
    constexpr int n = 32;
    int8_t h_a[m * k];
    int8_t h_b[k * n];
    int32_t h_c[m * n];
    for (int i=0; i<m*k; ++i) {
        h_a[i] = (i % 64);
    }
    for (int i=0; i<k*n; ++i) {
        h_b[i] = (i % 32);
    }

    printMat(h_a, m, k, "h_a");
    printMat(h_b, k, n, "h_b");

    int8_t *d_a;
    int8_t *d_b;
    int32_t *d_c;
    
    cudaStream_t master_stream;
    CHECKCUDAERROR(cudaStreamCreate(&master_stream));

    CHECKCUDAERROR(cudaMallocAsync((void **)&d_a, sizeof(int8_t) * m * k, master_stream));
    CHECKCUDAERROR(cudaMallocAsync((void **)&d_b, sizeof(int8_t) * k * n, master_stream));
    CHECKCUDAERROR(cudaMallocAsync((void **)&d_c, sizeof(int32_t) * m * n, master_stream));

    CHECKCUBLASSTATUS(cublasSetVectorAsync(m * k, sizeof(int8_t), h_a, 1, d_a, 1, master_stream));
    CHECKCUBLASSTATUS(cublasSetVectorAsync(k * n, sizeof(int8_t), h_b, 1, d_b, 1, master_stream));
    
    cublasLtHandle_t ltHandle;
    CHECKCUBLASSTATUS(cublasLtCreate(&ltHandle));

    LtIgemmTensor(ltHandle, m, n, k, d_a, k, d_b, n, d_c, n, 0, master_stream);

    CHECKCUBLASSTATUS(cublasGetVectorAsync(m * n, sizeof(int32_t), d_c, 1, h_c, 1, master_stream));
    printMat(h_c, m, n, "h_c");

    if (d_a) CHECKCUDAERROR(cudaFreeAsync(d_a, master_stream));
    if (d_b) CHECKCUDAERROR(cudaFreeAsync(d_b, master_stream));
    if (d_c) CHECKCUDAERROR(cudaFreeAsync(d_c, master_stream));
    
    CHECKCUDAERROR(cudaStreamDestroy(master_stream));
    CHECKCUBLASSTATUS(cublasLtDestroy(ltHandle));
}


int main() 
{
    testlt();

    return 0;
}
```

#### 3.2.1 矩阵初始化及数据拷贝
首先是对矩阵 `A` 和 `B` 进行初始化，初始化逻辑与前面相同，将数据存储在主机端的 `h_a`、`h_b`、 `h_c` 三个变量中，然后定义三个主机端变量 `d_a`、`d_b`、 `d_c`，通过 `cudaMallocAsync` 函数申请设备端内存，这里因为要把所有的操作都放在 `master_stream` 流中，所以使用了异步版本的 API。

设备内存分配完毕后，使用 `cublasSetVectorAsync` 函数将主机内存中数据传输到设备内存中，注意这里使用 CUDA Runtime API `cudaMemcpyAsync` 也是一样的。

#### 3.2.2 矩阵内存顺序问题
前面说过，`cublasLtMatmul` 函数要求矩阵 `A`、`C`、`D` 的内存布局为 `CUBLASLT_ORDER_COL32`，而矩阵 B 的内存布局为 `CUBLASLT_ORDER_COL4_4R2_8C`，形如 `CUBLASLT_ORDER_COL32` 或 `CUBLASLT_ORDER_COL4_4R2_8C` 这种内存布局，可以通过 cuBlasLt 库中的 `cublasLtMatrixTransform` API 进行转换，要注意的是在 cuBLASLt 库中矩阵默认是列主序的，所以如果我们的输入矩阵在主机端是行主序时，要么手动把矩阵调整为列主序，要么把矩阵句柄显式设置为行主序，这样在调用 `cublasLtMatrixTransform` API 时才能得到正确结果，笔者推荐用后一种方式，但是在 Faster Transformer v3.0 的源码中采用的是手动转置的方式，这里也给出转置 Kernel。
```cuda
#define COL32_ 32

// transpose matrix
// for (m n) col-major
// grid((m+31)/32, (n+31)/32)
// block(32, 32)
template <typename T>
__global__ void transposeMatrix_kernel(T *dst, const T *src, const int m, const int n)
{
    __shared__ T tile[COL32_][COL32_ + 1];

    int blockx32 = blockIdx.x * COL32_;
    int blocky32 = blockIdx.y * COL32_;
    int x = blockx32 + threadIdx.x;
    int y = blocky32 + threadIdx.y;

    bool check = ((x < m) && (y < n));
    tile[threadIdx.y][threadIdx.x] = check ? __ldg(src + y * m + x) : T(0);

    __syncthreads();

    y = blockx32 + threadIdx.y;
    x = blocky32 + threadIdx.x;

    check = ((x < n) && (y < m));
    if (check)
        dst[y * n + x] = tile[threadIdx.x][threadIdx.y];
}

// for (m, n) col-major matrix
template <typename T>
void transposeMatrix_kernelLauncher(T *dst, const T *src, const int m, const int n, cudaStream_t stream)
{
    transposeMatrix_kernel<T><<<dim3((m + 31) / 32, (n + 31) / 32), dim3(32, 32), 0, stream>>>(dst, src, m, n);
}
```
#### 3.2.3 调用 cublasLtMatmul API
在调用 `cublasLtMatmul` API 前需要先创建一个 cuBLASLt 句柄，这个句柄是和设备绑定的，不同 API 可以复用，在使用完毕后需要显式销毁。这里我们直接在 `testlt` 函数中创建即可，得到句柄 `ltHandle`。然后笔者封装了一个 `LtIgemmTensor` 函数，在这个函数中完成了一系列调用 `cublasLtMatmul` API 的前置工作。

```cuda
int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}

void LtIgemmTensor(cublasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const int8_t *A,
                   int lda,
                   const int8_t *B,
                   int ldb,
                   int32_t *C,
                   int ldc,
                   int mat_order,
                   cudaStream_t stream) {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    int32_t alpha = 1, beta = 0;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    // tensor op igemm kernels require specialized memory order of data
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    int8_t *Atransform = NULL, *Btransform = NULL;
    int32_t *Ctransform                   = NULL;
    cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
    float transformAlpha = 1.0f, transformBeta = 0.0f;
    cublasLtOrder_t order_ROW         = CUBLASLT_ORDER_ROW;
    cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    int ldatransform = 32 * m;
    int ldbtransform = 32 * roundoff(n, 8);
    int ldctransform = 32 * m;

    CHECKCUDAERROR(cudaMallocAsync(reinterpret_cast<void**>(&Atransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldatransform, stream));
    CHECKCUDAERROR(cudaMallocAsync(reinterpret_cast<void**>(&Btransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldbtransform, stream));
    CHECKCUDAERROR(cudaMallocAsync(reinterpret_cast<void**>(&Ctransform), sizeof(int32_t) * roundoff(n, 32) / 32 * ldctransform, stream));

    CHECKCUBLASSTATUS(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));

    CHECKCUBLASSTATUS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    // tensor op igemm kernels only support NT gemm
    CHECKCUBLASSTATUS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(opTranspose)));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for original matrices

    CHECKCUBLASSTATUS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda));
    CHECKCUBLASSTATUS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, ldb));
    CHECKCUBLASSTATUS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc));
    if (mat_order == 0) {
        CHECKCUBLASSTATUS(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_ROW, sizeof(order_ROW)));
        CHECKCUBLASSTATUS(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_ROW, sizeof(order_ROW)));
        CHECKCUBLASSTATUS(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_ROW, sizeof(order_ROW)));
    }

    // ---------------------------------------------------------------------------------------------
    // create descriptors for transformed matrices

    CHECKCUBLASSTATUS(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldatransform));
    CHECKCUBLASSTATUS(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // data memory order is set to CUBLASLT_ORDER_COL4_4R2_8C in order to achieve best performance on Turing devices.
    // for best performance on Ampere, consider setting the memory order to CUBLASLT_ORDER_COL32_2R_4R4.
    CHECKCUBLASSTATUS(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbtransform));
    CHECKCUBLASSTATUS(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));

    CHECKCUBLASSTATUS(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldctransform));
    CHECKCUBLASSTATUS(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // ---------------------------------------------------------------------------------------------
    // transforms and computation

    CHECKCUBLASSTATUS(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, A, Adesc, &transformBeta, NULL, NULL, Atransform, AtransformDesc, 0));

    // B matrix is non-transposed, but transposed matrix is needed - add transpose operation in matrix transform.
    CHECKCUBLASSTATUS(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    CHECKCUBLASSTATUS(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, B, Bdesc, &transformBeta, NULL, NULL, Btransform, BtransformDesc, 0));

    // no need to transform C matrix as beta is assumed to be 0
    CHECKCUBLASSTATUS(cublasLtMatmul(ltHandle,
                                     matmulDesc,
                                     &alpha,
                                     Atransform,
                                     AtransformDesc,
                                     Btransform,
                                     BtransformDesc,
                                     &beta,
                                     Ctransform,
                                     CtransformDesc,
                                     Ctransform,
                                     CtransformDesc,
                                     NULL,
                                     NULL,
                                     0,
                                     stream));

    opTranspose = CUBLAS_OP_N;
    CHECKCUBLASSTATUS(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    // transform outputs to COL order or ROW order
    CHECKCUBLASSTATUS(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, Ctransform, CtransformDesc, &transformBeta, NULL, NULL, C, Cdesc, 0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (CtransformDesc) CHECKCUBLASSTATUS(cublasLtMatrixLayoutDestroy(CtransformDesc));
    if (BtransformDesc) CHECKCUBLASSTATUS(cublasLtMatrixLayoutDestroy(BtransformDesc));
    if (AtransformDesc) CHECKCUBLASSTATUS(cublasLtMatrixLayoutDestroy(AtransformDesc));
    if (Cdesc) CHECKCUBLASSTATUS(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) CHECKCUBLASSTATUS(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CHECKCUBLASSTATUS(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) CHECKCUBLASSTATUS(cublasLtMatmulDescDestroy(matmulDesc));
    if (transformDesc) CHECKCUBLASSTATUS(cublasLtMatrixTransformDescDestroy(transformDesc));

    // wait until device is done before freeing transformed buffers
    CHECKCUDAERROR(cudaStreamSynchronize(stream));
    if (Ctransform) CHECKCUDAERROR(cudaFreeAsync(Ctransform, stream));
    if (Btransform) CHECKCUDAERROR(cudaFreeAsync(Btransform, stream));
    if (Atransform) CHECKCUDAERROR(cudaFreeAsync(Atransform, stream));
}
```

在矩阵乘法计算前，先创建一个矩阵乘法句柄 `matmulDesc`，在创建时也同步指定了 `computetype` 和 `scaleType` 分别为 `CUBLAS_COMPUTE_32I`、`CUDA_R_32I`，表示这是一个 `int8 * int8 -> int32` 的运算。此外再对矩阵 `B` 进行一个转置操作，前面介绍过目前这个 API 只支持 NT 乘法。

乘法句柄创建后，为了满足 `cublasLtMatmul` API 对矩阵内存排序的要求，需要对矩阵 `A`、`B`、`C` 进行转换，因此还需要创建如下变量：
- 三个矩阵句柄 `Adesc`、`Bdesc`、`Cdesc`，用于描述矩阵 `A`、`B`、`C` 的内存布局、数据类型、主维度等。
- 三个临时变量 `Atransform`、`Btransform`、`Ctransform`，用于临时存储转换后的矩阵 `A`、`B`、`C`，注意这里要进行设备内存分配。 
- 三个临时变量对应的矩阵句柄 `AtransformDesc`、`BtransformDesc`、`CtransformDesc`。

创建矩阵句柄 `Adesc`、`Bdesc`、`Cdesc` 时，需要考虑到当前矩阵元素的存储顺序是列主序还是行主序。如果是列主序，则 `lda` 为 `m`，`ldb` 为 `k`，`ldc` 为 `m`；如果是行主序，则 `lda` 为 `k`，`ldb` 为 `n`，`ldc` 为 `n`，除此之外行主序时还需要显式设定矩阵内存布局为 `CUBLASLT_ORDER_ROW`，这里笔者通过形参 `mat_order` 控制，等于 `0` 为行主序，否则为列主序。

同样地，创建矩阵句柄 `AtransformDesc`、`BtransformDesc`、`CtransformDesc` 时，也要设定内存排序。对于矩阵 `Atransform`、`Ctransform`，其内存布局为 `CUBLASLT_ORDER_COL32`，这种内存布局把矩阵按列主序分片存储，每片形状为 `rows * 32`，所以其主维度也为 `rows * 32`。对于矩阵 `Btransform`，其内存布局为 `CUBLASLT_ORDER_COL4_4R2_8C`，其主维度为 `32 * roundoff(n, 8)`。此外，由于前面讲过 API 只支持 NT 乘法，所以 `Btransform` 表示的矩阵实际是 `B.T`（即矩阵 `B` 的转置），此时矩阵行数为 `n`，列数为 `k`。

完成临时矩阵的句柄创建之后，要通过 `cublasLtMatrixTransform` API 将原始矩阵 `A`、`B` 变换为内存布局为 `CUBLASLT_ORDER_COL32`、`CUBLASLT_ORDER_COL4_4R2_8C` 的矩阵 `Atransform`、`Btransform`。调用 `cublasLtMatrixTransform` API 前同样需要创建一个转换句柄 `transformDesc`，这个句柄不是一次性的，下次调用这个函数时可以复用。要注意的是 `Btransform` 表示的矩阵实际是 `B.T`，因此需要额外指定一个变换属性 `CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA`，将其指定为 `CUBLAS_OP_T`，转换后的 `Atransform`、`Btransform` 就可以作为参数传入 `cublasLtMatmul` 函数。

`cublasLtMatmul` 函数的参数意义前面已经介绍过了，本次由于要计算的公式为 `C = A * B`，所以把 `alpha` 设置为 `1`，把 `beta` 设置为 `0`，然后 `C`、`D` 两个参数都传入 `Ctransform`，相当于一次 in-place 乘法，其他参数默认即可。

完成矩阵乘法之后，`Ctransform` 中就存储着计算结果，但是内存布局为 `CUBLASLT_ORDER_COL32`，通常除了在 cuBLAS 类库以外，我们自行编写的 CUDA 程序都是使用行主序的，为了方便后续代码使用，可以调用 `cublasLtMatrixTransform` API 将 `Ctransform` 变换为行主序存储的 `C`。这里需要注意的是，转换句柄 `transformDesc` 在转换 `B` 矩阵时设置了转置属性，这里复用前需要把转置属性再设置为 `CUBLAS_OP_N`（不转置）。

完成所有计算后切记把所有句柄全部销毁。

### 3.3 CUBLASLT_ORDER_COL32 
`CUBLASLT_ORDER_COL32` 是 cuBLASLt 中的一种内存排序方式，以列主序存储矩阵元素，并且把矩阵分片存储。假设矩阵形状为 `[m, n]`，则每一片的大小为 `[m, 32]`，片内采用列主序，填充顺序如下。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qq77XYnvmmxPgkEOQT154Da6FIMQvTu7xoAAxd3ECLqwadicKmBB8jmZrUVHeEhsaY6hvRCLEI6Fg/640?wx_fmt=png&amp;from=appmsg)

假设矩阵 `A` 的形状为 `[96, 64]`，其元素值如下：
```
array([[   0,    1,    2, ...,   61,   62,   63],
       [  64,   65,   66, ...,  125,  126,  127],
       [ 128,  129,  130, ...,  189,  190,  191],
       ...,
       [5952, 5953, 5954, ..., 6013, 6014, 6015],
       [6016, 6017, 6018, ..., 6077, 6078, 6079],
       [6080, 6081, 6082, ..., 6141, 6142, 6143]])
```
假定矩阵 `A` 按列主序存储，则它的元素存储顺序按照一维数组表示如下：
```
    [   0,   64,  128, ..., 5952, 6016, 6080,
        1,   65,  129, ..., 5953, 6017, 6081,
        2,   66,  130, ..., 5954, 6018, 6082,
        ..,
        61,  125,  189, ..., 6013, 6077, 6141,
        62,  126,  190, ..., 6014, 6078, 6142,
        63,  127,  191, ..., 6015, 6079, 6143   ]
```
将其转换为 `CUBLASLT_ORDER_COL32` 格式过程中又经过了一次转置，转换完成后，按如下方式存储：
```
    [   0,    1,    2, ...,   29,   30,   31,
        64,   65,   66, ...,   93,   94,   95,
        128,  129,  130, ...,  157,  158,  159,
       ...,
        6080, 6081, 6082, ..., 6109, 6110, 6111,
        32,   33,   34, ...,   61,   62,   63,
       ...,
        6112, 6113, 6114, ..., 6141, 6142, 6143  ]
```
可以发现，截至当前 `CUBLASLT_ORDER_COL32` 格式相对于最原始的行主序格式的区别只是把矩阵的列按 `32` 分片存储了，相当于：
```
A_col32 = np.concatenate([A[:, :32], A[:, 32:]], axis=0)
```
相当于在原矩阵中用下图的方式索引：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qq77XYnvmmxPgkEOQT154DHxFxicna9zxzGvHfb2YDV0rzFNNOm4MMh2lWUyIsfKBDibmn4OLCa2CA/640?wx_fmt=png&amp;from=appmsg)

所以，如果要在 `A_col32` 中索引原矩阵中的元素 `A[i, j]`，应该用如下方式计算索引偏移量：
```
idx = ((j >> 5) << 5) * m + (i << 5) + (j & 0x1f)
```

## 4 小结

本文提供了一个相对完整的使用 cuBLASLt 的矩阵乘法 API 进行矩阵乘法操作的示例，希望对读者有所帮助。
- 在使用 cuBLAS 类库的时候，需要特别注意列主序的问题。这点是 cuBLAS 相对不那么友好的地方，希望官方什么时候给优化一下。
- 由于 API 对矩阵的内存顺序有要求，在传参之前还需要进行矩阵转换，建议使用 `cublasLtMatrixTransform` API 进行转换，但是有些时候官方源码会把这些转换操作融合到其他 Kernel 或自行编写转换 Kernel，这会给代码的可阅读性和维护性带来压力。
- 笔者针对 `CUBLASLT_ORDER_COL32` 内存顺序给出了详细介绍，也提供了一般的索引方式。