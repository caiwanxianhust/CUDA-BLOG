# 【CUDA编程】SGEMM 优化日志

**写在前面**：2025 年伊始，笔者迎来了人生中的一大喜事——妻子顺利分娩，母子平安。在这段充满喜悦与忙碌的育儿时光中，博客更新不得不暂时搁置。如今，随着新年的到来，笔者决定重新拾起笔杆，继续分享技术心得。

## 1 文章背景
本文聚焦于基于 CUDA 加速的单精度矩阵通用乘法（SGEMM）的优化策略。尽管关于 GEMM 优化的讨论已屡见不鲜，但笔者仍抽出一周时间，对相关优化策略进行了系统梳理，并附上了相应的 CUDA 代码实现。本文的初衷在于记录这一优化过程，而非追求算法上的创新。

坦白而言，在过去的开发经历中，笔者鲜少亲自编写 GEMM 的 Kernel，大多时候都是直接调用 cuBLAS 库。对于 CUDA 开发者而言，要编写出性能超越 cuBLAS 的 GEMM Kernel 实属不易。因此，从实用角度出发，笔者更倾向于直接调用库函数。然而，通过这次深入的优化实践，笔者不仅加深了对 CUDA 编程的理解，也为未来的技术探索奠定了坚实的基础。

在接下来的内容中，笔者将详细阐述 SGEMM 的优化策略，并分享在 CUDA 编程中的一些心得体会。希望这些内容能为读者带来启发，也期待与大家共同探讨更多技术细节。

## 2 计算任务
单精度矩阵通用乘法（Single-precision General Matrix Multiplication，SGEMM）是线性代数中的核心计算任务之一，广泛应用于科学计算、深度学习等领域。其数学定义为：给定两个单精度矩阵 $A \in R^{m \times k}$ 和 $B \in R^{k \times n}$，SGEMM 计算它们的乘积 
$$
C = \alpha A B + \beta C
$$
其中，$C \in R^{m \times n}$ 是结果矩阵，$\alpha$ 和 $\beta$ 为标量系数。

SGEMM 的计算复杂度为 $O(mnk)$，因此其性能优化对提升整体计算效率至关重要。

## 3 Naive 实现
首先矩阵 C 的形状为 `[m, n]`，我们很容易想到是否可以使用一个线程计算一个矩阵 C 元素，基于这个思路，我们可以使用整个线程 grid 完成一个矩阵 C 的计算，每个 block 完成 `block_size` 个元素的计算，每个 thread 完成 1 个元素的计算。为了方便索引我们不妨将 grid 和 block 维度都设置为二维，`block_size` 取到最大 1024，具体如下：
```cpp
void launchSgemmNaiveKernel(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                const float alpha, const float beta, cudaStream_t stream)
    {
        dim3 block(32, 32);
        dim3 gird((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        SgemmNaiveKernel<<<gird, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
```

对于每个 thread 而言，要计算矩阵 C 中的一个元素，也就意味着，该 thread 需要分别读取 1 行矩阵 A 的元素、1 列矩阵 B 的元素、以及 1 个矩阵 C 的元素，需要写入 1 个矩阵 C 的元素。即，针对 Kernel 对全局内存（global memory）的访问，每个 thread 的访存情况如下：
- load：$2k + 1$
- store：$1$

扩展到整个计算过程，则针对 global memory 的访问情况如下：
- load：$m n (2k + 1)$
- store：$m n$

计算过程比较简单，在当前线程内计算两个长度为 `k` 的向量（A 的行向量、B 的列向量）内积，然后使用标量系数和矩阵 C 的元素值进行修正即可得到结果，示意图和 Kernel 代码如下。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rG7f8D3stiawKRyR37czTtgF9QMjLFB0CA1fb3XEiaGs7uqxibZVjkPhdDrapjaPBmuKNeFtCicXyGWw/640?wx_fmt=png&amp;from=appmsg)

```cpp
__global__ void SgemmNaiveKernel(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                    const float alpha, const float beta)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i)
        {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}
```

上面这个 Kernel 存在什么问题？对于同一个 block 的相邻 thread 而言，两个线程都需要从 global memory 中读取矩阵 A 的同一行数据，并分别读取矩阵 B 的相邻列数据，这就造成了同一个 block 内的线程从 global memory 中重复读取大量数据。考虑到共享内存（shared memory）在 block 的可见性，我们可以考虑将数据先将数据从 global memory 读取到 shared memory，然后 block 内的线程从 shared memory 中再读取后参与计算，利用 shared memory 远大于 global memory 的带宽来加速内存访问。

## 4 block 分片并使用 shared memory 缓存矩阵分片
上节说到，我们可以利用 shared memory 来进行访存加速，理想情况下，我们以线程 block 为单位，比如一个 block（假设形状为 `[32, 32]`）计算矩阵 C 的一个分片（假设形状为 `[BM, BN]`），这就要求在 shared memory 中需要同时存储一个形状为 `[BM, k]` 的矩阵 A 分片和一个形状为 `[k, BN]` 的矩阵 B 分片，然而对于大尺寸矩阵而言，由于 shared memory 容量有限（单个 block 最多可以使用 64KB），很难一次性加载入矩阵 A 或矩阵 B 的多个行或列，因此我们需要稍作调整。调整后，对于单个 block 而言还是要加载 `(BM+BN)k` 个元素到 shared memory，只不过变成了分步加载，每次加载 `(BM+BN)BK` 个元素，并完成一次矩阵乘加操作，具体示意图如下：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rG7f8D3stiawKRyR37czTtgwPibPubhOQU5PMt7VYnGfNKeXH3ofdlvmjZ07yYGCiahQiaicjghpP0LRA/640?wx_fmt=png&amp;from=appmsg)

kernel 执行配置信息如下：
```cpp
void launchSgemmSmemKernel_v1(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                const float alpha, const float beta, cudaStream_t stream)
{
    const int BM = 32;
    const int BN = 32;
    const int BK = 32;
    dim3 block(BN, BM);
    dim3 gird((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    size_t smem = sizeof(float) * (BM * BK + BN * BK);
    SgemmSmemKernel_v1<BM, BN, BK><<<gird, block, smem, stream>>>(A, B, C, M, N, K, alpha, beta);
}
```

从图上看，对于矩阵 A、B，我们沿着 `k` 方向每次加载一个厚度为 `BK` 的矩阵分片到 shared memory 中，完成矩阵乘法计算，然后累加求和。每个 thread 还是计算矩阵 C 中的一个元素，对于每个 block 而言，访存情况如下：
- load：$(BM+BN)k + BM \cdot BN$
- store: $BM \cdot BN$

扩展到整个计算过程，对 global memory 的访问情况如下：
- load：$\frac{m}{BM} \cdot \frac{n}{BN} (BM+BN)k + mn = \frac{mnk}{16} + mn$
- store: $mn$

可见，不算矩阵 C 的加载的话，针对矩阵 A、B 的 global memory 加载量缩小为原来的 $1/32$，极大地降低了访存次数。根据上述思路可以得到如下 kernel 代码：
```cpp
/** 共享内存、分片
* dim3 block(BN, BM);
* dim3 gird((N+block.x-1)/block.x, (M+block.y-1)/block.y);
*/
template <int BM, int BN, int BK>
__global__ void SgemmSmemKernel_v1(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                    const float alpha, const float beta)
{
    const int offset_a = blockIdx.y * blockDim.y * K;
    const int offset_b = blockIdx.x * blockDim.x;
    const int offset_c = blockIdx.y * blockDim.y * N + blockIdx.x * blockDim.x;

    extern __shared__ float s_ptr[];
    float *s_a = s_ptr;
    float *s_b = s_a + BM * BK;

    float tmp = 0.0f;
#pragma unroll
    for (int i = 0; i < K; i += BK)
    {
#pragma unroll
        for (int j = threadIdx.x; j < BK; j += blockDim.x)
        {
            s_a[threadIdx.y * BK + j] = A[offset_a + threadIdx.y * K + i + j];
        }
#pragma unroll
        for (int j = threadIdx.y; j < BK; j += blockDim.y)
        {
            s_b[threadIdx.x + BK * j] = B[offset_b + i * N + j * N + threadIdx.x];
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < BK; ++j)
        {
            tmp += s_a[threadIdx.y * BK + j] * s_b[threadIdx.x + BK * j];
        }
        __syncthreads();
    }
    C[offset_c + threadIdx.y * N + threadIdx.x] = tmp * alpha + beta * C[offset_c + threadIdx.y * N + threadIdx.x];
}
```
在以上代码中，block 维度被设置为 `[BN, BM]`，每个 thread 仍然处理矩阵 C 的一个元素。block 内总共需要消耗 `(32*32+32*32) * 4B=8KB` shared memory 的容量，距离上限还有不少差距。外层 for 循环是在 `k`  维度上的循环迭代，前两个循环中是针对每个 thread 来说将数据从矩阵分片加载到 `s_a` 和 `s_b`，第三个循环中是对当前线程块分片沿 `BK` 维度逐一进行点积，然后进行累加。

## 5 一个线程处理多个元素
在上一个 kernel 中我们仍然是一个 thread 完成矩阵 C 中一个元素的处理，这就意味着我们可能需要更多的线程才能完成整个矩阵乘法的计算，我们知道对于 GEMM 计算，通常优化瓶颈在于访存带宽而不在于计算吞吐量，所以我们是否可以考虑每个线程一次计算多个元素？具体地，我们把矩阵 A 每次迭代的分片（形状为 `[BM, BK]`）从 `BK` 维度进一步划分，每 `TM` 行分为一组，也就是说在沿 `k` 维度的每一次迭代中中，每个线程完成 `[TM, BK]` 与 `[BK, 1]` 两个小矩阵的乘法，对于形状为 `TM, 1` 的结果矩阵，相当于每个线程从原来只计算结果矩阵的一个元素，增加到计算 `TM` 个元素，下面我们来看一下示意图。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5osO4vd1y4gfbrg95QX7Nbs6vQW9xU5O0Rls9n2DOQ0I6pic3T1pVNYG7NkFaVuF1LU3HPOtnFRCmw/640?wx_fmt=png&amp;from=appmsg)

针对上述思路，我们有如下的 kernel 执行配置：
```cpp
void launchSgemmSmemKernel_v2(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream)
{
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;
    dim3 block(BN * BM / TM);
    dim3 gird((N + BN - 1) / BN, (M + BM - 1) / BM);
    SgemmSmemKernel_v2<BM, BN, BK, TM><<<gird, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
```

从图上看，对于矩阵 A、B，我们沿着 `k` 方向每次加载一个厚度为 `BK` 的矩阵分片到 shared memory 中，完成矩阵乘法计算，然后累加求和。每个 thread 计算矩阵 C 中的 `TM` 个元素，对于每个 block 而言，访存情况如下：
- load：$(BM+BN)k + BM \cdot BN$
- store: $BM \cdot BN$

扩展到整个计算过程，对 global memory 的访问情况如下：
- load：$\frac{m}{BM} \cdot \frac{n}{BN} (BM+BN)k + mn = \frac{mnk}{32} + mn$
- store: $mn$

可见，不算矩阵 C 的加载的话，针对矩阵 A、B 的 global memory 加载量缩小为原来的 $1/64$。根据上述思路可以得到如下 kernel 代码：
```cpp
/** 每个线程计算多个元素，一维
 * dim3 block(BM * BN / TM = 512);
 * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
 */
template <int BM, int BN, int BK, int TM>
__global__ void SgemmSmemKernel_v2(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                    const float alpha, const float beta)
{
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    const int offset_a = blockIdx.y * BM * K;
    const int offset_b = blockIdx.x * BN;
    const int offset_c = blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    __shared__ float s_a[BM * BK];
    __shared__ float s_b[BK * BN];

    float tmp[TM] = {0.0f};
    int row_a = threadIdx.x / BK;
    int col_a = threadIdx.x % BK;
    int row_b = threadIdx.x / BN;
    int col_b = threadIdx.x % BN;
#pragma unroll
    for (int i = 0; i < K; i += BK)
    {
        s_a[threadIdx.x] = A[offset_a + row_a * K + i + col_a];
        s_b[threadIdx.x] = B[offset_b + row_b * N + i * N + col_b];
        __syncthreads();

#pragma unroll
        for (int j = 0; j < BK; ++j)
        {
            float tmp_b = s_b[j * BN + (threadCol)];
#pragma unroll
            for (int tmidx = 0; tmidx < TM; ++tmidx)
            {
                tmp[tmidx] += s_a[(threadRow * TM + tmidx) * BK + j] * tmp_b;
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
    {
        int row_c = (threadRow * TM) + i;
        int col_c = (threadCol);
        C[offset_c + row_c * N + col_c] = alpha * tmp[i] + beta * C[offset_c + row_c * N + col_c];
    }
}
```
在以上代码中，`block_size` 被设置为 `BN * BM / TM`，每个 thread 处理矩阵 C 的 `TM` 个元素。block 内总共需要消耗 `(64*8+64*8) * 4B=4KB` shared memory 的容量，比上一个 Kernel 要小一些。外层 for 循环是在 `k`  维度上的循环迭代，首先是每个 thread 将数据从矩阵分片加载到 `s_a` 和 `s_b`，由于 `BM`、`BN`、`BK`和 `TM` 等几个参数的设置比较巧妙，正好每个线程只需要分别加载 A、B 中的一个元素，第二层循环中是对当前线程块分片沿 `BK` 维度的迭代，此时每个线程沿 `BM` 维度会循环处理 `TM` 个元素而沿 `BN` 维度一次只处理一个元素，所以取出 `s_b` 中的一个元素放入临时寄存器，然后在 `TM` 维度上循环取数计算后放入结果寄存器 `tmp[TM]` 中即可。最后将结果寄存器 `tmp[TM]` 中的 `TM` 个元素更新到矩阵 C 上。

## 6 一个线程处理多个元素，拓展到二维
在上一个 kernel 中我们让每个线程在矩阵分片中沿 `BM` 方向处理 `TM` 行元素，而沿 `BN` 方向只处理一列元素，从而对应矩阵 C 中的 `TM` 个计算结果。那么自然地，我们可以将这个思路拓展到二维，进一步增加每个线程的计算量。具体地，我们让每个线程在矩阵分片 `s_a` 中沿 `BM` 方向处理 `TM` 行元素，在矩阵分片 `s_b` 中沿 `BN` 方向处理 `TN` 列元素，从而对应矩阵 C 中的形状为 `[TM, TN]` 的计算结果。下面我们来看一下示意图：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5osO4vd1y4gfbrg95QX7NbsJiawFQu0m4t3icT5rNPhcqdGkj8f8Bicxz6aF3Q75eCTEzCYXJZWlqofw/640?wx_fmt=png&amp;from=appmsg)

根据上述思路，我们可以设置如下的 kernel 执行配置：
```cpp
void launchSgemmSmemKernel_v3(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 block(BN * BM / (TM * TN));
    dim3 gird((N + BN - 1) / BN, (M + BM - 1) / BM);
    SgemmSmemKernel_v3<BM, BN, BK, TM, TN><<<gird, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
```

从图上看，对于矩阵 A、B，我们沿着 `k` 方向每次加载一个厚度为 `BK` 的矩阵分片到 shared memory 中，完成矩阵乘法计算，然后累加求和。每个 thread 计算矩阵 C 中的 `TM*TN` 个元素，对于每个 block 而言，访存情况如下：
- load：$(BM+BN)k + BM \cdot BN$
- store: $BM \cdot BN$

扩展到整个计算过程，对 global memory 的访问情况如下：
- load：$\frac{m}{BM} \cdot \frac{n}{BN} (BM+BN)k + mn = \frac{mnk}{64} + mn$
- store: $mn$

可见，不算矩阵 C 的加载的话，针对矩阵 A、B 的 global memory 加载量缩小为原来的 $1/128$。根据上述思路可以得到如下 kernel 代码：
```cpp
/** 每个线程计算多个元素，拓展到二维
 * dim3 block(BM * BN / (TM * TN));
 * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
 */
template <int BM, int BN, int BK, int TM, int TN>
__global__ void SgemmSmemKernel_v3(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                    const float alpha, const float beta)
{
    const int offset_a = blockIdx.y * BM * K;
    const int offset_b = blockIdx.x * BN;
    const int offset_c = blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    int row_a, col_a, row_b, col_b;

    __shared__ float s_a[BM * BK];
    __shared__ float s_b[BK * BN];

    float tmp[TM * TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

#pragma unroll
    for (int i = 0; i < K; i += BK)
    {
#pragma unroll
        for (int j = threadIdx.x; j < BM * BK; j += blockDim.x)
        {
            row_a = j / BK;
            col_a = j % BK;
            s_a[j] = A[offset_a + row_a * K + i + col_a];
        }

#pragma unroll
        for (int j = threadIdx.x; j < BK * BN; j += blockDim.x)
        {
            row_b = j / BN;
            col_b = j % BN;
            s_b[row_b * BN + col_b] = B[offset_b + i * N + row_b * N + col_b];
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < BK; ++j)
        {
#pragma unroll
            for (int tmidx = 0; tmidx < TM; ++tmidx)
            {
                reg_a[tmidx] = s_a[threadRow * TM * BK + tmidx * BK + j];
            }
#pragma unroll
            for (int tnidx = 0; tnidx < TN; ++tnidx)
            {
                reg_b[tnidx] = s_b[j * BN + threadCol * TN + tnidx];
            }

#pragma unroll
            for (int tmidx = 0; tmidx < TM; ++tmidx)
            {
#pragma unroll
                for (int tnidx = 0; tnidx < TN; ++tnidx)
                {
                    tmp[tmidx * TN + tnidx] += reg_a[tmidx] * reg_b[tnidx];
                }
            }
        }

        __syncthreads();
    }

    int row_c, col_c;
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            row_c = threadRow * TM + i;
            col_c = threadCol * TN + j;
            C[offset_c + row_c * N + col_c] = alpha * tmp[i * TN + j] + beta * C[offset_c + row_c * N + col_c];
        }
    }
}
```
在以上代码中，`block_size` 被设置为 `BN * BM / (TM * TN)`，每个 thread 处理矩阵 C 的 `TM*TN` 个元素。首先在 kernel 中定义了两个 shared memory 数组，block 内总共需要消耗 `(128*8+128*8) * 4B=8KB` shared memory 的容量。

然后定义了三个寄存器数组，其中 `tmp[TM * TN]` 用于存储当前线程计算的 `TM*TN` 个矩阵乘法结果，后续会更新到矩阵 C 中，而 `reg_a[TM]` 和 `reg_b[TN]` 分别用来临时存储每次沿 `k` 维度迭代时的矩阵分块 `s_a`、`s_b` 中的数据，把待计算数据从 shared memory 加载到寄存器中，免得每次计算都要去 shared memory 中取数。

外层 for 循环是在 `k`  维度上的循环迭代，每次处理一个矩阵分片。首先是每个 thread 将数据从矩阵分片加载到 `s_a` 和 `s_b`，每个 thread 可能加载多个元素，所以这里加了一层循环。第 3 个循环中是对当前线程块分片沿 `BK` 维度的迭代，此时每个线程沿 `BM` 维度会循环处理 `TM` 个元素，沿 `BN` 维度会循环处理 `TN` 个元素，所以分别将这 `TM`、`TN` 个元素存入临时寄存器 `reg_a[TM]` 和 `reg_b[TN]`，然后根据这 `TM`、`TN` 个元素更新小结果矩阵 `[TM, TN]`。最后将结果寄存器 `tmp[TM *TN]` 中的元素更新到矩阵 C 上。

## 7 向量化数据加载
前面的 kernel 中我们每次从 global memory 加载数据到 shared memory 时都是采用每个线程加载一个 `float` 元素，这里我们可以基于 CUDA 内置向量数据类型 `float4`，一次性加载 4 个元素到 shared memory。同时为了从 `s_b` 中往寄存器加载数据时在 `TM` 维度上循环读数据时能够连续读取，从而使编译器能够自动向量化加载，我们将 `s_b` 由 `[BM, BK]` 转置为 `[BK, BM]`，示意图如下：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5osO4vd1y4gfbrg95QX7NbsOn4nuMOyeOuJF53rALrwiay06q5GFYjbXsba5HTKeeCmp48icmdvpylw/640?wx_fmt=png&amp;from=appmsg)

kernel 执行配置不涉及变更：
```cpp
void launchSgemmSmemKernel_v4(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 block(BN * BM / (TM * TN));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    SgemmSmemKernel_v4<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
```

由于只涉及加载指令的变更，所以内存访问情况也与上一个 kernel 相同，具体地，整个计算过程，对 global memory 的访问情况如下：
- load：$\frac{m}{BM} \cdot \frac{n}{BN} (BM+BN)k + mn = \frac{mnk}{64} + mn$
- store: $mn$

在 kernel 代码中主要是从 global memory 中加载数据到 shared memory，以及 `s_a` 加载到 `reg_a` 这两处代码存在变更，前者是涉及 `float *` 类型指针到 `float4 *` 类型指针的转化及元素索引，后者是因为 `s_a` 转置而导致的索引值变更，其他代码没有变化。
```cpp
/** 向量化加载
 * dim3 block(BM * BN / (TM * TN));
 * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
 */
template <int BM, int BN, int BK, int TM, int TN>
__global__ void SgemmSmemKernel_v4(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                    const float alpha, const float beta)
{
    const int offset_a = blockIdx.y * BM * K;
    const int offset_b = blockIdx.x * BN;
    const int offset_c = blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    int row_a, col_a, row_b, col_b;

    // 通过 Padding 避免 bank conflict
    const int extra_col = 4;
    __shared__ float s_a[BK * (BM + extra_col)]; // transpose s_a to [BK, BM + extra_col]
    __shared__ float s_b[BK * BN];

    float tmp[TM * TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

#pragma unroll
    for (int i = 0; i < K; i += BK)
    {
        // 每次从 global mem 加载 4 个 float 到 shared mem
#pragma unroll
        for (int j = (threadIdx.x << 2); j < BM * BK; j += (blockDim.x << 2))
        {
            row_a = j / BK;
            col_a = j % BK;
            float4 tmp = reinterpret_cast<const float4 *>(A + offset_a + row_a * K + i + col_a)[0];

            s_a[col_a * (BM + extra_col) + row_a] = tmp.x;
            s_a[(col_a + 1) * (BM + extra_col) + row_a] = tmp.y;
            s_a[(col_a + 2) * (BM + extra_col) + row_a] = tmp.z;
            s_a[(col_a + 3) * (BM + extra_col) + row_a] = tmp.w;
        }

#pragma unroll
        for (int j = (threadIdx.x << 2); j < BK * BN; j += (blockDim.x << 2))
        {
            row_b = j / BN;
            col_b = j % BN;
            reinterpret_cast<float4 *>(s_b + row_b * BN + col_b)[0] = reinterpret_cast<const float4 *>(B + offset_b + i * N + row_b * N + col_b)[0];
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < BK; ++j)
        {
#pragma unroll
            for (int tmidx = 0; tmidx < TM; ++tmidx)
            {
                reg_a[tmidx] = s_a[threadRow * TM + tmidx + j * (BM + extra_col)];
            }
#pragma unroll
            for (int tnidx = 0; tnidx < TN; ++tnidx)
            {
                reg_b[tnidx] = s_b[j * BN + threadCol * TN + tnidx];
            }

#pragma unroll
            for (int tmidx = 0; tmidx < TM; ++tmidx)
            {
#pragma unroll
                for (int tnidx = 0; tnidx < TN; ++tnidx)
                {
                    tmp[tmidx * TN + tnidx] += reg_a[tmidx] * reg_b[tnidx];
                }
            }
        }
        __syncthreads();
    }

    int row_c, col_c;
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            row_c = threadRow * TM + i;
            col_c = threadCol * TN + j;
            C[offset_c + row_c * N + col_c] = alpha * tmp[i * TN + j] + beta * C[offset_c + row_c * N + col_c];
        }
    }
}
```
由于 `s_a` 进行了转置，所以从 `A` 中往 `s_a` 加载数据的时候，原本连续的数据变成了间隔一整行，再加上一次加载 `4` 个 `float`，所以在 warp 中相邻线程访问 shared memory 的位置间隔了 `4` 行，如果不做处理，那么行数为 `BM=128`，此时存在严重的 bank conflict，因此这里将 `s_a` 的形状调整为 `[BK, BM+extra_col]`，使得 `s_a` 中同一列的数据不在同一个 bank，后续访问 `s_a` 的时候注意索引计算即可。

## 8 warp 分片
在前面的 kernel 中，通过矩阵分片的形式划分了两个层级，即 block 分片（`[BM, BN]`）、thread 分片（`[TM, TN]`）。我们知道，warp 是 GPU 调度和执行的最小单位，GPU 不能单独调度一个线程，而是以 warp 为单位进行调度和执行‌，一个 warp 包含 32 个线程，这些线程在 SIMT（单指令多线程）模式下同步执行相同的指令。在前面的 kernel 中，由于没有可以划分 warp 层级，这就使得 warp 间会重复访问 shared memory 上的数据，具体地，对于每个 warp 而言，都需要访问 `s_a`、`s_b` 中整行或整列的数据，而 warp 内的 thread 通常又会重复访问 `s_a` 或 `s_b` 中的重复数据，这就带来了大量的重复访问，如果我们能够更细粒度的划分矩阵分片，让每个 warp 固定完成一个矩阵分片，则可以更高程度的优化内存访问，取得更好的加速效果。

具体地，我们在 block 分片 `[BM, BN]` 内部引入形状为 `[WM, WN]` 的 warp 分片，即每个 warp 负责形状为 `[WM, WN]` 的矩阵分片的计算任务，这就需要 `BM*BN/(WM*WN)` 个 warp，在 warp 分片内部，每个线程处理 `[TM, TN]` 个小分片的结果，那么如果 `TM * TN * 32 < WM * WN` 时，显然只计算一次是无法完成任务的，这就需要 warp 在 `WM` 和 `WN` 维度上循环迭代，迭代次数分别为 `WMITERS` 和 `WNITERS`，每次迭代处理的矩阵分片为 `[warp_sub_M, warp_sub_N]`，每次迭代中每个 thread 也只计算一次。根据以上思路，我们有如下示意图：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qibITAh4KDtFcFsBDlHLcaewgEJOgBkUibEuPicTPy9MrStG7xtlSvAQACuTKOepyPwEicmVbZc5DVNA/640?wx_fmt=png&amp;from=appmsg)

kernel 执行配置如下：
```cpp
void launchSgemmSmemKernel_v5(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int WM = 64;
    const int WN = 32;
    const int TM = 4;
    const int TN = 4;

    dim3 block(BN * BM / (WM * WN) * 32);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    SgemmSmemKernel_v5<BM, BN, BK, WM, WN, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
```
增加 warp 层级的矩阵分片并没有改变 global memory 加载逻辑，因此对于 global memory 的访存逻辑没有变化，为了整体 kernel 看起来比较简洁，我们把从 global memory 加载到 shared memory 的过程写到一个设备函数 `loadFromGmem` 中，具体代码如下：
```cpp
template <int BM, int BN, int BK>
__device__ void loadFromGmem(const float *A, const float *B, const int M, const int N, const int K,
                                float *s_a, float *s_b, int offset_a, int offset_b,
                                const int bkidx, const int buffer_id, const int extra_col)
{
    int row_a, col_a, row_b, col_b;
    float4 tmp;
    // 每次从 global mem 加载 4 个 float 到 shared mem
#pragma unroll
    for (int j = (threadIdx.x << 2); j < BM * BK; j += (blockDim.x << 2))
    {
        row_a = j / BK;
        col_a = j % BK;

        // 每次从 global mem 加载 4 个 float 到 shared mem
        tmp = reinterpret_cast<const float4 *>(A + offset_a + row_a * K + bkidx + col_a)[0];
        s_a[buffer_id * BK * (BM + extra_col) + col_a * (BM + extra_col) + row_a] = tmp.x;
        s_a[buffer_id * BK * (BM + extra_col) + (col_a + 1) * (BM + extra_col) + row_a] = tmp.y;
        s_a[buffer_id * BK * (BM + extra_col) + (col_a + 2) * (BM + extra_col) + row_a] = tmp.z;
        s_a[buffer_id * BK * (BM + extra_col) + (col_a + 3) * (BM + extra_col) + row_a] = tmp.w;
    }

#pragma unroll
    for (int j = (threadIdx.x << 2); j < BK * BN; j += (blockDim.x << 2))
    {
        row_b = j / BN;
        col_b = j % BN;
        reinterpret_cast<float4 *>(s_b + buffer_id * BK * BN + row_b * BN + col_b)[0] =
            reinterpret_cast<const float4 *>(B + offset_b + bkidx * N + row_b * N + col_b)[0];
    }
}
```
数据加载完成后就是对当前 warp 分片的数据计算工作，在 kernel 中已经对当前 warp 和 thread 进行定位，即计算出了 `warp_row`、`warp_col`、`thread_in_warp_row`、`thread_in_warp_col`，当前 warp 分片解决的是 `s_a` 和 `s_b` 中形状为 `[WM, BK]` 与 `[BK, WN]` 的矩阵乘法问题，所以首先还是要在 `BK` 方向上循环。循环内部还是老步骤，先将当前线程处理的数据从 shared memory 加载到寄存器，两层循环，一层是在 `WMITERS` 和 `WNITERS` 维度上的迭代，另一层是在 `TM` 和 `TN` 维度上。数据加载到寄存器上之后就分别在 `WMITERS`、`WNITERS`、`TM` 和 `TN` 维度上完成计算即可。具体代码逻辑封装在设备函数 `processFromSmem` 中。
```cpp
template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__device__ void processFromSmem(const float *s_a, const float *s_b, float *reg_a, float *reg_b, float *tmp, const int WMITERS, const int WNITERS,
                                const int warp_sub_M, const int warp_sub_N, const int warp_row, const int warp_col, const int thread_in_warp_row,
                                const int thread_in_warp_col, const int buffer_id, const int extra_col)
{
#pragma unroll
    for (int j = 0; j < BK; ++j)
    {
        // 从 shared mem 加载到 register
#pragma unroll
        for (int w_m_id = 0; w_m_id < WMITERS; ++w_m_id)
        {
#pragma unroll
            for (int tmidx = 0; tmidx < TM; ++tmidx)
            {
                reg_a[w_m_id * TM + tmidx] =
                    s_a[buffer_id * BK * (BM + extra_col) + j * (BM + extra_col) + warp_row * WM + w_m_id * warp_sub_M + thread_in_warp_row * TM + tmidx];
            }
        }

#pragma unroll
        for (int w_n_id = 0; w_n_id < WNITERS; ++w_n_id)
        {
#pragma unroll
            for (int tnidx = 0; tnidx < TN; ++tnidx)
            {
                reg_b[w_n_id * TN + tnidx] =
                    s_b[buffer_id * BK * BN + j * BN + warp_col * WN + w_n_id * warp_sub_N + thread_in_warp_col * TN + tnidx];
            }
        }

#pragma unroll
        for (int w_m_id = 0; w_m_id < WMITERS; ++w_m_id)
        {
#pragma unroll
            for (int w_n_id = 0; w_n_id < WNITERS; ++w_n_id)
            {
#pragma unroll
                for (int tmidx = 0; tmidx < TM; ++tmidx)
                {
#pragma unroll
                    for (int tnidx = 0; tnidx < TN; ++tnidx)
                    {
                        tmp[w_m_id * WNITERS * TM * TN + w_n_id * TM * TN + tmidx * TN + tnidx] +=
                            reg_a[w_m_id * TM + tmidx] * reg_b[w_n_id * TN + tnidx];
                    }
                }
            }
        }
    }
}
```
完成当前线程的计算后，需要将计算结果写入 global memory，这一步的主要逻辑就是找到当前线程结果寄存器 `tmp` 中的元素在矩阵 C 中的位置，即索引计算。我们一层一层来捋一下，首先我们先来定位当前 block 对应的矩阵 C 的偏移量（这个可以结合示意图来理解），在计算偏移量时我们先不要一步到位，先考虑坐标位置，显然起始坐标为 `[blockIdx.y * BM, blockIdx.x * BN]`，则偏移量为：
```cpp
blockIdx.y * BM * N + blockIdx.x * BN
```
然后，还有当前 warp 在 block 分片内的起始坐标 `[warp_row * WM, warp_col * WN]`，则偏移量为
```
warp_row * WM * N + warp_col * WN
```
然后还有当前 thread 在 warp 内的坐标：
```
row_c = w_m_id * warp_sub_M + thread_in_warp_row * TM + tmidx;
col_c = w_n_id * warp_sub_N + thread_in_warp_col * TN + tnidx;
```
那么最终的偏移量：
```
blockIdx.y * BM * N + blockIdx.x * BN + warp_row * WM * N + warp_col * WN + row_c * N + col_c
```
得到偏移量之后，借助向量化数据类型 `float4` 写入矩阵 C 即可，回写逻辑封装在设备函数 `writeFromRegToGmem` 中代码如下：
```cpp
template <int BM, int BN, int BK, int TM, int TN>
__device__ void writeFromRegToGmem(const float *tmp, float *C, const int M, const int N, const int WMITERS, const int WNITERS,
                                    const int warp_sub_M, const int warp_sub_N, const int thread_in_warp_row, const int thread_in_warp_col,
                                    const int offset_c, const float alpha, const float beta)
{
    float4 ret4;
    int row_c, col_c, tmp_id;
#pragma unroll
    for (int w_m_id = 0; w_m_id < WMITERS; ++w_m_id)
    {
#pragma unroll
        for (int w_n_id = 0; w_n_id < WNITERS; ++w_n_id)
        {
#pragma unroll
            for (int tmidx = 0; tmidx < TM; ++tmidx)
            {
#pragma unroll
                for (int tnidx = 0; tnidx < TN; tnidx += 4)
                {

                    row_c = w_m_id * warp_sub_M + thread_in_warp_row * TM + tmidx;
                    col_c = w_n_id * warp_sub_N + thread_in_warp_col * TN + tnidx;
                    tmp_id = w_m_id * WNITERS * TM * TN + w_n_id * TM * TN + tmidx * TN + tnidx;
                    ret4 = reinterpret_cast<float4 *>(C + offset_c + row_c * N + col_c)[0];
                    ret4.x = alpha * tmp[tmp_id] + beta * ret4.x;
                    ret4.y = alpha * tmp[tmp_id + 1] + beta * ret4.y;
                    ret4.z = alpha * tmp[tmp_id + 2] + beta * ret4.z;
                    ret4.w = alpha * tmp[tmp_id + 3] + beta * ret4.w;
                    reinterpret_cast<float4 *>(C + offset_c + row_c * N + col_c)[0] = ret4;
                }
            }
        }
    }
}
```
将数据加载、计算、数据回写封装到 3 个设备函数之后，我们的 kernel 代码变得简洁了很多，且思路比较清晰，具体如下：
```cpp
/** warp 分片
 * dim3 block(BM * BN / (TM * TN));
 * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
 */
template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void SgemmSmemKernel_v5(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                    const float alpha, const float beta)
{
    // block 内 warp 二维分布的 id
    const int warp_row = (threadIdx.x >> 5) / (BN / WN);
    const int warp_col = (threadIdx.x >> 5) % (BN / WN);
    // M、N 方向每个 warp 迭代次数
    const int WMITERS = 2;
    const int WNITERS = (WM * WN) / (TM * TN * 32 * WMITERS);
    // warp 每次迭代处理的 M、N 方向上的元素个数
    const int warp_sub_M = WM / WMITERS;
    const int warp_sub_N = WN / WNITERS;
    // warp 内 thread 二维分布的 id
    const int thread_in_warp_row = (threadIdx.x & 0x1f) / (warp_sub_N / TN);
    const int thread_in_warp_col = (threadIdx.x & 0x1f) % (warp_sub_N / TN);
    // warp 级偏移量
    const int offset_a = blockIdx.y * BM * K;
    const int offset_b = blockIdx.x * BN;
    const int offset_c = blockIdx.y * BM * N + warp_row * WM * N + blockIdx.x * BN + warp_col * WN;

    // 通过 Padding 避免 bank conflict
    const int extra_col = 4;
    __shared__ float s_a[BK * (BM + extra_col)]; // transpose s_a to [BK, BM + extra_col]
    __shared__ float s_b[BK * BN];

    float tmp[WMITERS * TM * WNITERS * TN] = {0.0f};
    float reg_a[WMITERS * TM];
    float reg_b[WNITERS * TN];

#pragma unroll
    for (int i = 0; i < K; i += BK)
    {
        // 每次从 global mem 加载 4 个 float 到 shared mem
        loadFromGmem<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, i, 0, extra_col);
        __syncthreads();

        processFromSmem<BM, BN, BK, WM, WN, TM, TN>(s_a, s_b, reg_a, reg_b, tmp, WMITERS, WNITERS, warp_sub_M, warp_sub_N,
                                                    warp_row, warp_col, thread_in_warp_row, thread_in_warp_col, 0, extra_col);
        __syncthreads();
    }

    writeFromRegToGmem<BM, BN, BK, TM, TN>(tmp, C, M, N, WMITERS, WNITERS, warp_sub_M, warp_sub_N, thread_in_warp_row, thread_in_warp_col, offset_c, alpha, beta);
}
```

## 9 双缓冲
双缓冲（Double Buffer），有时又被称作数据预取（Prefetch），核心思想是通过两个缓冲区实现**访存**和**计算**分离，进而达到 overlap 的目的，掩盖指令延迟。

什么叫**访存**和**计算**分离？具体来说，根据之前的逻辑，在一次大的循环里（沿 `k` 维度的循环），每次我们从 global memory 加载矩阵 A、B 的矩阵分片分别存储到 shared memory 数组 `s_a`、`s_b` 中，然后按部就班地从 `s_a`、`s_b` 中取数据计算，这里有一个问题，就是计算和数据加载是有先后顺序的（强依赖），由于 global memory 的带宽有限，访存指令从发出到完成会有一段延迟，因此对 shared memory 的读取要等待 global memory 读取并写入 shared memory 之后，这就有一段延迟，虽然 GPU 可以通过 SM 上切换其他 block 来掩盖这部分延迟，但通常为了保证足够的并行，矩阵分片的尺寸较大，从而分配较多的 shared memory 和寄存器，进而会导致 SM 中分配的 block 数量有限或者寄存器溢出，因此对 global memory 的访存延迟便难以掩盖。

双缓冲的引入，会对 shared memory 和寄存器开辟两倍的空间，为了防止寄存器溢出，笔者在这里只考虑 shared memory 级别的双缓冲，即在 shared memory 中开辟两倍缓冲区，分别称为 `buffer_0` 和 `buffer_1`，当沿着 `k` 维度循环时，在 load 一个矩阵分片到 `buffer_0` 的同时进行 `buffer_1` 的计算，因为两者内存方面没有强依赖，所以在指令层级可以并行，一定程度上掩盖了访存延迟。我们先来看一下代码：
```cpp
/** double_buffer
 * dim3 block(BM * BN / (TM * TN), 2);
 * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
 */
template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void SgemmSmemKernel_v6(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                    const float alpha, const float beta)
{
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    //     printf("in SgemmSmemKernel_v6\n");
    // block 内 warp 二维分布的 id
    const int warp_row = (threadIdx.x >> 5) / (BN / WN);
    const int warp_col = (threadIdx.x >> 5) % (BN / WN);
    // M、N 方向每个 warp 迭代次数
    const int WMITERS = 2;
    const int WNITERS = (WM * WN) / (TM * TN * 32 * WMITERS);
    // warp 每次迭代处理的 M、N 方向上的元素个数
    const int warp_sub_M = WM / WMITERS;
    const int warp_sub_N = WN / WNITERS;
    // warp 内 thread 二维分布的 id
    const int thread_in_warp_row = (threadIdx.x & 0x1f) / (warp_sub_N / TN);
    const int thread_in_warp_col = (threadIdx.x & 0x1f) % (warp_sub_N / TN);
    // warp 级偏移量
    const int offset_a = blockIdx.y * BM * K;
    const int offset_b = blockIdx.x * BN;
    const int offset_c = blockIdx.y * BM * N + warp_row * WM * N + blockIdx.x * BN + warp_col * WN;

    // 通过 Padding 避免 bank conflict
    const int extra_col = 4;
    __shared__ float s_a[2 * BK * (BM + extra_col)]; // transpose s_a to [BK, BM + extra_col]
    __shared__ float s_b[2 * BK * BN];

    float tmp[WMITERS * TM * WNITERS * TN] = {0.0f};
    float reg_a[WMITERS * TM];
    float reg_b[WNITERS * TN];

    int buffer_id = 0;

    loadFromGmem<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, 0, buffer_id, extra_col);
    __syncthreads();

#pragma unroll
    for (int i = 0; i < K - BK; i += BK)
    {
        loadFromGmem<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, i + BK, 1 - buffer_id, extra_col);
        processFromSmem<BM, BN, BK, WM, WN, TM, TN>(s_a, s_b, reg_a, reg_b, tmp, WMITERS, WNITERS, warp_sub_M, warp_sub_N,
                                                    warp_row, warp_col, thread_in_warp_row, thread_in_warp_col, buffer_id, extra_col);
        buffer_id = 1 - buffer_id;
        __syncthreads();
    }

    processFromSmem<BM, BN, BK, WM, WN, TM, TN>(s_a, s_b, reg_a, reg_b, tmp, WMITERS, WNITERS, warp_sub_M, warp_sub_N,
                                                warp_row, warp_col, thread_in_warp_row, thread_in_warp_col, buffer_id, extra_col);

    __syncthreads();
    writeFromRegToGmem<BM, BN, BK, TM, TN>(tmp, C, M, N, WMITERS, WNITERS, warp_sub_M, warp_sub_N,
                                            thread_in_warp_row, thread_in_warp_col, offset_c, alpha, beta);
}

void launchSgemmSmemKernel_v6(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                const float alpha, const float beta, cudaStream_t stream)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int WM = 64;
    const int WN = 32;
    const int TM = 4;
    const int TN = 4;

    assert(K % (2 * BK) == 0);

    dim3 block(BN * BM / (WM * WN) * 32);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    SgemmSmemKernel_v6<BM, BN, BK, WM, WN, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
```

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5peIjJ5ZgJo06vOMLn3N8Q0fkRhtfYcUuaMuua7QQzLEicPnB48ibcYuFQY96rlE0KHVdfcIdVEKSgQ/640?wx_fmt=png&amp;from=appmsg)

可以看出，代码整体上还是顺序执行的逻辑，并没有使用 cuda barrier 或 memcpy_async 等 API，从代码逻辑上来看，依然是 load 一个矩阵分配然后计算一个矩阵分片，有先后关系，感觉达不到 overlap 的目的。实则不然，核心在于代码对应的指令发射与执行完成是一个过程，访存和计算指令对应着不同的硬件单元，也就是说这两种指令可以并行执行，而代码的顺序执行仅仅意味着这两种指令的发射是顺序进行的，但指令发射速度极快，而执行需要时间，并且访存指令相比与计算指令的执行有更长的时钟周期，所以实质上虽然代码是顺序逻辑，但对 `buffer_0` 的加载和对 `buffer_1` 的读取并计算是可以并行执行的。此外，与上一个 Kernel 相比，由于双缓冲机制下加载和计算互不依赖，循环内部少了一个 `__syncthreads()`，在 block 层面少了一次显式同步，进而掩盖了从 global memory 加载到 shared memory 的部分访存延迟。

## 10 tensor core 简单应用
在前面的 kernel 中已经将矩阵分片拆分到 warp 层级并加入了双缓冲机制，而正好针对 warp 层级的计算，CUDA 编程模型也提供了相应的 warp matrix function（也称 WMMA API），该函数从 CUDA 9.0 引入，可以利用 GPU 上的 Tensor Core 来加速形如 $D = \alpha A B + \beta C$ 的通用矩阵乘法运算，与其他线程束函数相同，该函数是一个 warp 内的集合操作，必须所有 warp 线程共同参与，否则可能会挂起。

WMMA API 全都封装在 `nvcuda::wmma` 命名空间中，在使用前需要引入 `mma.h` 头文件，这里要注意，由于 `nvcuda::wmma` 命名空间定义在 `#if defined(__cplusplus) && defined(__CUDACC__)` 里，因此诸如 `using namespace nvcuda;` 这种代码需要写在设备端代码中。由于笔者本文的代码都是在 Turing 架构下实现的，所以需要考虑 WMMA API 在 `__CUDA_ARCH__ = 750` 时的一些限制条件，综合考虑后选择如下的参数类型：
|矩阵|A|B|C|
|---|:---:|:---:|:---:|
|数据类型|`half`|`half`|`float`|
|形状|`16*16`|`16*16`|`16*16`|

WMMA API 中引入了类模板 `fragment`，用于存储 warp matrix function 中参与计算的矩阵信息，支持 `matrix_a`、`matrix_b`、`accumulator` 三种类型，分别对应矩阵 A、矩阵 B、累加矩阵 C 或结果矩阵 D，我们在 Kernel 中先实例化这几个类型：
```cpp
using FragAType = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>;
using FragBType = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
using FragAccType = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
using FragCType = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
```
可以发现，上述代码中还有个 `WKITERS` 参数，这个参数是新引入的，因为在目前 warp 分片的场景下，每个 warp 每次计算 `[WM, BK] * [BK, WN]` 的结果，考虑到 tensorcore 支持的矩阵形状有限（前面说过本次 `WMMA_M`、`WMMA_N`、`WMMA_K` 都为 `16`），所以需要在 `WM`、`WM`、`BK` 维度上进行循环，循环次数分别为：
```cpp
// 单个 warp 处理层面 M、N、K 方向每个 warp 迭代次数
constexpr int WMITERS = WM / WMMA_M;
constexpr int WNITERS = WN / WMMA_N;
constexpr int WKITERS = BK / WMMA_K;
```
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rh8OyZ39DEiao3aXKpK8rJ8jFjxRaf2flTQEZ7wkAzJS06gthV9XTYia8wLHHMicIBmsKdXN82icNSsg/640?wx_fmt=png&amp;from=appmsg)

从图中可以看出，每个 warp 需要存储的小矩阵数量分别为：
```cpp
FragAType a_frag[WMITERS * WKITERS];
FragBType b_frag[WNITERS * WKITERS];
FragAccType acc_frag[WMITERS * WNITERS];
FragCType c_frag[WMITERS * WNITERS];
```

其中累加矩阵 `acc_frag` 存储的是 $A B$ 的结果，结果矩阵 `c_frag` 存储的是最终 $\alpha A B + \beta C$ 的结果。在计算之前需要先把 `acc_frag` 中的元素置零，WMMA API 中提供了 `fill_fragment` 函数。
```cpp
for (int i = 0; i < WMITERS * WNITERS; ++i)
{
    wmma::fill_fragment(acc_frag[i], 0.0f);
}
```
然后就是前面的双缓冲逻辑，先把数据从 global memory 加载到 shared memory，这里要注意的是由于 tensorcore 要求的数据类型为 `half` 所以这里还需要加一个 `float2half` 的操作，即 `s_a`、`s_b` 中直接存储 `half` 类型，对于 `s_b`，为了利用内置类型 `half2`，笔者把原来的加载函数进行了改写，一次加载 2 个元素。
```cpp
template <int BM, int BN, int BK>
__device__ void loadFromGmemToSmemAndConvertToHalf(const float *A, const float *B, const int M, const int N, const int K,
                                                    half *s_a, half *s_b, int offset_a, int offset_b,
                                                    const int bkidx, const int buffer_id, const int extra_col)
{
    int row_a, col_a, row_b, col_b;
    float4 tmp4;
    // 每次从 global mem 加载 4 个 float 到 shared mem
#pragma unroll
    for (int j = (threadIdx.x << 2); j < BM * BK; j += (blockDim.x << 2))
    {
        row_a = j / BK;
        col_a = j % BK;
        // 每次从 global mem 加载 4 个 float 到 shared mem
        tmp4 = reinterpret_cast<const float4 *>(A + offset_a + row_a * K + bkidx + col_a)[0];
        s_a[buffer_id * BK * (BM + extra_col) + col_a * (BM + extra_col) + row_a] = __float2half(tmp4.x);
        s_a[buffer_id * BK * (BM + extra_col) + (col_a + 1) * (BM + extra_col) + row_a] = __float2half(tmp4.y);
        s_a[buffer_id * BK * (BM + extra_col) + (col_a + 2) * (BM + extra_col) + row_a] = __float2half(tmp4.z);
        s_a[buffer_id * BK * (BM + extra_col) + (col_a + 3) * (BM + extra_col) + row_a] = __float2half(tmp4.w);
    }

    float2 tmp2;
#pragma unroll
    for (int j = (threadIdx.x << 1); j < BK * BN; j += (blockDim.x << 1))
    {
        row_b = j / BN;
        col_b = j % BN;
        tmp2 = reinterpret_cast<const float2 *>(B + offset_b + bkidx * N + row_b * N + col_b)[0];
        reinterpret_cast<half2 *>(s_b + buffer_id * BK * BN + row_b * BN + col_b)[0] = __float22half2_rn(tmp2);
    }
```
相应地，从 shared memory 加载数据到寄存器的逻辑也要进行调整，WMMA API 中提供了 warp 级别的操作用于加载 `fragment`，使用比较简单，由于 `fragment` 的形状在模板参数中已经指定，仅需要注意源矩阵的形状即可，这里要注意对于 `s_a` 为了缓解 bank conflict 形状进行了调整。对于矩阵乘法操作，需要调用 `mma_sync` 函数完成，具体代码如下：
```cpp
template <int BM, int BN, int BK, int WM, int WN, int WMITERS, int WNITERS, int WKITERS, int WMMA_M, int WMMA_N, int WMMA_K, typename T1, typename T2, typename T3>
__device__ void processFromSmemByTensorCore(const half *s_a, const half *s_b, T1 *a_frag, T2 *b_frag, T3 *acc_frag,
                                            const int warp_row, const int warp_col, const int buffer_id, const int extra_col)
{
    using namespace nvcuda;
    int shm_offset;
#pragma unroll
    for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
    {
#pragma unroll
        for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
        {
            shm_offset = buffer_id * BK * (BM + extra_col) + wkidx * (BM + extra_col) + warp_row * WM + wmidx * WMMA_M;
            wmma::load_matrix_sync(a_frag[wmidx * WKITERS + wkidx], s_a + shm_offset, BM + extra_col);
        }
    }
#pragma unroll
    for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
    {
#pragma unroll
        for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
        {
            shm_offset = buffer_id * BK * BN + wkidx * BN + warp_col * WN + wnidx * WMMA_N;
            wmma::load_matrix_sync(b_frag[wnidx * WKITERS + wkidx], s_b + shm_offset, BN);
        }
    }
#pragma unroll
    for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
    {
#pragma unroll
        for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
        {
#pragma unroll
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
            {
                wmma::mma_sync(acc_frag[wmidx * WNITERS + wnidx], a_frag[wmidx * WKITERS + wkidx],
                                b_frag[wnidx * WKITERS + wkidx], acc_frag[wmidx * WNITERS + wnidx]);
            }
        }
    }
}
```

完成计算后，$AB$ 结果存储在 `acc_frag` 中，还需要结合 C 矩阵及系数进行调整，因此还需要将 C 矩阵的元素从 global memory 加载到 `c_frag` 中，最后计算 $\alpha AB + \beta C$ 后存入矩阵 C 中，从 `fragment` 对象中读取数据存入 global memory 需要调用 `store_matrix_sync` API，具体代码如下：
```cpp
template <int WMITERS, int WNITERS, int WMMA_M, int WMMA_N, typename T1, typename T2>
__device__ void writeFromRegToGmemByTensorCore(T1 *c_frag, T2 *acc_frag, float *C, const int M, const int N,
                                                const int offset_c, const float alpha, const float beta)
{
    using namespace nvcuda;
#pragma unroll
    for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
    {
#pragma unroll
        for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
        {
            wmma::load_matrix_sync(c_frag[wmidx * WNITERS + wnidx], C + offset_c + wmidx * WMMA_M * N + wnidx * WMMA_N,
                                    N, wmma::mem_row_major);
#pragma unroll
            for (int idx = 0; idx < c_frag[wmidx * WNITERS + wnidx].num_elements; ++idx)
            {
                c_frag[wmidx * WNITERS + wnidx].x[idx] = alpha * acc_frag[wmidx * WNITERS + wnidx].x[idx] + beta * c_frag[wmidx * WNITERS + wnidx].x[idx];
            }
            wmma::store_matrix_sync(C + offset_c + wmidx * WMMA_M * N + wnidx * WMMA_N, c_frag[wmidx * WNITERS + wnidx],
                                    N, wmma::mem_row_major);
        }
    }
}
```
以上只是对 tensorcore 的一次简单应用，未根据矩阵形状、GPU 架构等进行调优，其中针对数据加载函数 `load_matrix_sync` 可能带来的 bank conflict 问题也未进行考虑，因此还有较大地优化空间，笔者这里只是提供一个原始 WMMA API 的用法示例，有兴趣的读者可以结合 Nvidia 官方库 CUTLASS 进一步了解 GEMM 的优化细节。本节对应的 kernel 代码如下：
```cpp
/** tensor core
     * dim3 block(BN * BM / (WM * WN) * 32);
     * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
     */
template <int BM, int BN, int BK, int WM, int WN, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void SgemmSmemKernel_v7(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                    const float alpha, const float beta)
{
    using namespace nvcuda;
    // block 内 warp 二维分布的 id
    const int warp_row = (threadIdx.x >> 5) / (BN / WN);
    const int warp_col = (threadIdx.x >> 5) % (BN / WN);
    // 单个 warp 处理层面 M、N、K 方向每个 warp 迭代次数
    constexpr int WMITERS = WM / WMMA_M;
    constexpr int WNITERS = WN / WMMA_N;
    constexpr int WKITERS = BK / WMMA_K;
    // warp 级偏移量
    const int offset_a = blockIdx.y * BM * K;
    const int offset_b = blockIdx.x * BN;
    const int offset_c = blockIdx.y * BM * N + warp_row * WM * N + blockIdx.x * BN + warp_col * WN;

    // 通过 Padding 避免 bank conflict
    constexpr int extra_col = 8;
    __shared__ half s_a[2 * BK * (BM + extra_col)]; // transpose s_a to [BK, BM + extra_col]
    __shared__ half s_b[2 * BK * BN];

    using FragAType = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>;
    using FragBType = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
    using FragAccType = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
    using FragCType = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
    FragAType a_frag[WMITERS * WKITERS];
    FragBType b_frag[WNITERS * WKITERS];
    FragAccType acc_frag[WMITERS * WNITERS];
    FragCType c_frag[WMITERS * WNITERS];

#pragma unroll
    for (int i = 0; i < WMITERS * WNITERS; ++i)
    {
        wmma::fill_fragment(acc_frag[i], 0.0f);
    }

    int buffer_id = 0;

    loadFromGmemToSmemAndConvertToHalf<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, 0, buffer_id, extra_col);
    __syncthreads();

#pragma unroll
    for (int i = 0; i < K - BK; i += BK)
    {
        loadFromGmemToSmemAndConvertToHalf<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, i + BK, 1 - buffer_id, extra_col);

        processFromSmemByTensorCore<BM, BN, BK, WM, WN, WMITERS, WNITERS, WKITERS, WMMA_M, WMMA_N, WMMA_K, FragAType, FragBType, FragAccType>(
            s_a, s_b, a_frag, b_frag, acc_frag, warp_row, warp_col, buffer_id, extra_col);

        buffer_id = 1 - buffer_id;
        __syncthreads();
    }

    processFromSmemByTensorCore<BM, BN, BK, WM, WN, WMITERS, WNITERS, WKITERS, WMMA_M, WMMA_N, WMMA_K, FragAType, FragBType, FragAccType>(
        s_a, s_b, a_frag, b_frag, acc_frag, warp_row, warp_col, buffer_id, extra_col);
    __syncthreads();

    writeFromRegToGmemByTensorCore<WMITERS, WNITERS, WMMA_M, WMMA_N, FragCType, FragAccType>(c_frag, acc_frag, C, M, N, offset_c, alpha, beta);
}

void launchSgemmSmemKernel_v7(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                const float alpha, const float beta, cudaStream_t stream)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int WM = 64;
    const int WN = 32;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    dim3 block(BN * BM / (WM * WN) * 32);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    SgemmSmemKernel_v7<BM, BN, BK, WM, WN, WMMA_M, WMMA_N, WMMA_K><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
```

## 11 小节
本文由浅入深地介绍了 SGEMM 任务的 CUDA 加速策略，并基于笔者自身理解提供了相应的 kernel 实现，要说明的是，本文介绍的内容并非 SGEMM 的优化终点，GEMM 的优化任务一直是大模型推理等高性能计算领域的经典任务之一，笔者之前习惯了调库，这次花了一点时间从头走一遍迭代优化过程后，获益良多。