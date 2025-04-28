#! https://zhuanlan.zhihu.com/p/646997739
# 【CUDA编程】CUDA并行化中的直方图问题
**写在前面**：本文是笔者使用CUDA编程对直方图算法的一个简单实现。本文开发环境：windows10 + CUDA10.0 + driver Versin 456.71

## 1 直方图
直方图算法是指对一个数组 `[x_1, x_2, x_3,...x_n]` 进行指定条件下的频数统计的算法，比如对于整数数组统计其对 `3` 取模的频数，对灰度图像统计其灰度值频数等等，总的来说就是根据指定条件进行分桶，求各桶的元素个数。

## 2 基于CPU编程的直方图算法
考虑有 `N` 个元素的数组 `x`，元素取值范围为 `0~255`，我们要统计数组元素取值分别为 `0~255` 的频数，就是我们通常说的灰度直方图。下面给出笔者基于CPU编程的C++代码，计算思路比较简单，无须赘述。

```cuda
void histCpu(
    const unsigned char *h_buffer, 
    unsigned int *h_histo, 
    const int N) {
 
    for (int i=0; i<N; ++i) h_histo[h_buffer[i]]++;
}
```
参数说明如下：
- `h_buffer`: 数组 `x`，`h_` 前缀表示这是个主机端的数组，数组长度为 `N`。
- `h_histo`: 计算结果，直方图数组，长度为 `256`，同样存放在主机端。

在这个例子中，我们考虑一个长度为 `1024 * 1024 * 100` 的一维数组，在主函数中我们将数组 `x` 的每个元素随机初始化为 `0~255`。接着调用 `timing` 函数对 `histCpu` 函数进行计时，为了计时方便，咱们重复运行 `20` 次，使用 `nvcc -O3 hist.cu -o hist.exe` 编译后运行 `hist.exe`，该函数耗时约 `41ms`。
```cuda
Using cpu:
Time = 41.8013 +- 0.243068 ms.
409691  409567  409485  409382  409586  409540  409622  409780  409479  409452  409711  409651  409644  409841  409582  409587
```

## 3 基于GPU全局内存原子操作的直方图算法
核函数代码如下：

```
__global__ void histKernel(
    const unsigned char *d_buffer, 
    const int N, 
    unsigned int *d_histo) {

    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < N) atomicAdd(&d_histo[d_buffer[n]], 1);
}
```
参数说明如下：
- `d_buffer`: 数组 `x`，`d_` 前缀表示这是个设备端的数组，数组长度为 `N`。
- `d_histo`: 计算结果，直方图数组，长度为 `256`，同样存放在设备端。

对CUDA编程来说，各个线程并行读取全局内存中 `d_buffer` 的元素，然后更新 `d_histo` 的值即可。注意这里不能像像前面CPU代码一样直接 `d_histo[value]++`，因为CPU代码每一轮读写是串联执行的，所以读写是安全的，而CUDA代码每个线程是并行的，如果这样写就有可能出现某个线程还没来得及更新 `d_histo[value]` 时， `d_histo[value]` 就被其他线程取走更新了，这时候后面线程读取的 `d_histo[value]` 就是未更新的值，造成计算错误。所以这里要加一个原子操作 `atomicAdd`，每个线程有序轮流对这个变量 `d_histo[value]` 更新。这样一来就几乎完全没发挥到CUDA并行的优势，编译运行后如下：
```cuda
Using kernel:
Time = 28.2375 +- 3.02136 ms.
409691  409567  409485  409382  409586  409540  409622  409780  409479  409452  409711  409651  409644  409841  409582  409587
```
可以看到，运行时间大约为 `28ms`，相比于CPU代码提升不大，甚至于有可能在数组长度比较小的时候反而CPU代码更快一些。

## 4 基于GPU共享内存原子操作的直方图算法
前面我们介绍了基于全局内存的算法，核函数中直接对全局内存进行大量读写操作，各个线程都直接操作全局内存执行原子操作，并行效率低。这里我们使用共享内存做一次简单优化，具体代码如下：
```cuda
__global__ void histshared(
    const unsigned char *d_buffer, 
    const int N, 
    unsigned int *d_histo) {

    int n = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ unsigned int s_histo[256];
    s_histo[threadIdx.x] = 0;
    __syncthreads();
    if (n < N) {
        atomicAdd(&s_histo[d_buffer[n]], 1);
        __syncthreads();
        atomicAdd(&d_histo[threadIdx.x], s_histo[threadIdx.x]);
    }
}
```
我们将线程块大小设置为 `256`，和直方图数组的长度保持一致，这样我们可以将线程ID与直方图数组索引联系起来，每个线程只负责对应的value值的频数统计。定义一个共享内存变量 `s_histo`，临时存放线程块内的直方图数据，该变量对整个线程块可见，在每个线程内执行原子操作更新 `s_histo`， 利用线程块同步函数 `__syncthreads()` 确保线程块内 `s_histo` 更新完毕后，将 `s_histo` 更新到 `d_histo`。
仔细一看，这样做似乎并没有减少对全局内存的访问，确实如此，每个线程内都进行了一次全局内存的读取和 `atomicAdd` 操作。这样能否起到优化效果呢？我们看下运行结果：
```cuda
Using kernel shared:
Time = 4.50488 +- 0.288098 ms.
409691  409567  409485  409382  409586  409540  409622  409780  409479  409452  409711  409651  409644  409841  409582  409587
```
可以看到，运行时间大约为 `4.5ms`，相比全局内存可以说是很大的提升了。这是为什么呢？前面说过我们明明没有减少对全局内存的操作，另外还增加了共享内存的开销。  
原因在于，我们将直接更新 `d_histo` 分解为两步。先更新 `s_histo`，使用 `s_histo` 预先对线程块内的直方图进行计算，我们知道共享内存的读写速度非常快远超全局内存，我们在一个线程块内对共享内存执行原子操作，这时候极端情况有最多256个线程排队执行原子操作。第二步我们把 `s_histo` 更新到 `d_histo` 时，线程块内部更是互不影响，因为各自负责各自原地址的读写，只会在网格维度可能存在排队情况。相比于只使用全局内存相当于我们把线程块内的原子操作从全局内存搬到了共享内存，利用共享内存的性能进行了优化。

## 5 进一步提升线程利用率
我们注意到在前面利用共享内存计算的过程中，对于每一个线程块，我们统计一个直方图，这相当于把256个数据分到256个桶中，这样必然会有大量的桶是没有数据的（`s_histo[threadIdx.x] == 0`），意味着大量的线程在把 `s_histo` 更新到 `d_histo` 时，使用原子操作给全局内存变量执行加 `0` 操作，操作没有意义却必须轮流执行，线程利用率极低。尤其对于二维图像灰度直方图应用来说，该问题更是非常突出，因为对于图像而言一个线程块区域内灰度变化通常不会太大，只会集中于某个很小的灰度值范围，必然导致大量的桶没有数据。那么如何有效减小无意义的加 `0` 原子操作呢？  
能否加 `if` 判断条件，比如写出如下代码：  
 `if (s_histo[threadIdx.x]) atomicAdd(&d_histo[threadIdx.x], s_histo[threadIdx.x])`   
理论上这会带来一个新的问题：**线程束分支发散**。在一个线程束内部如果存在分支条件使得不同线程执行不同代码的时候，会导致分支发散，在程序开发过程中我们应尽量避免分支发散。所以，当 `s_histo[threadIdx.x] != 0` 的线程执行原子操作时，`s_histo[threadIdx.x] == 0` 线程会闲置。相当于这个思路，在增加一步判断操作和一次共享内存读取操作的同时还带来了分支发散问题，不一定能有效提升计算速度，实测运行时间如下：  
```cuda
Using kernel shared:
Time = 4.49045 +- 0.304713 ms.
409691  409567  409485  409382  409586  409540  409622  409780  409479  409452  409711  409651  409644  409841  409582  409587
```
可以看到，运行时间大约为 `4.5ms`，提升效果可以忽略。  
还有一种方式：在一个线程内处理多个数据，增加 `s_histo` 统计的数据量，这样最终`s_histo[threadIdx.x] == 0` 的线程也会大大减少。具体地，我们可以取 `gridSize=10240`，这样 `gridSize * blockSize << N`，定义一个步长 `offset = gridSize * blockSize`，在一个线程内我们循环处理 `n, n+offset, n+2*offset, ...` 等多个数据，具体代码如下：
```cuda
__global__ void histParallelism(
    const unsigned char *d_buffer, 
    const int N, 
    unsigned int *d_histo) {

    int n = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ unsigned int s_histo[256];
    s_histo[threadIdx.x] = 0;
    __syncthreads();
    const int offset = gridDim.x * blockDim.x;
    for (; n<N; n+=offset) {
        atomicAdd(&s_histo[d_buffer[n]], 1); 
    }
    __syncthreads();
    atomicAdd(&d_histo[threadIdx.x], s_histo[threadIdx.x]);
}
```
在核函数内部我们通过循环按步长处理了全局内存变量 `d_buffer` 的数据，该代码运行时间如下：
```cuda
Using kernel Parallelism:
Time = 0.85092 +- 0.0901206 ms.
409691  409567  409485  409382  409586  409540  409622  409780  409479  409452  409711  409651  409644  409841  409582  409587
```
可以看到，运行时间缩短到了 `0.85ms`，相比于最开始的全局变量版本提升了 `35` 倍；相比于仅使用共享内存的算法，耗时也只是其 `20%` ，性能提升显著。

## 6 小结
本文内容总结如下：
- 原子函数对第一参数指向的数据进行一次“读-改-写”的原子操作，所谓原子操作即表示这是一次一气呵成、不可分割的操作。对所有的线程来说，这个操作是一个线程一个线程轮流执行，没有明显次序。另外，原子函数的第一个参数可以指向全局内存，也可以指向共享内存。
- 当一个线程束中的线程顺序地执行判断语句中的不同分支时，我们称发生了分支发散，分支发散会降低线程束执行效率，因此在核函数代码中应该尽量避免分支发散。
- 为了提升线程利用率，有时候网格内的线程数量不一定要和数组长度对应，可以设置小一些，然后在一个线程内按步长处理多个数组元素，但要注意全局内存的合并访问问题，以免非合并访问全局内存导致得不偿失。

## 6 附录
本文代码如下：
hist.cu
```cuda
#include "error.cuh"
#include "stdio.h"

const int N = 1024 * 1024 * 100;
const int NUM_REAPEATS = 20;

void bigRandomBlock(unsigned char* data, const int n) {
    srand(1234);
    for (int i=0; i<n; ++i) data[i] = rand();
}

void histCpu(const unsigned char *h_buffer, unsigned int *h_histo, const int N) {
    for (int i=0; i<N; ++i) h_histo[h_buffer[i]]++;
}

__global__ void histKernel(
    const unsigned char *d_buffer, 
    const int N, 
    unsigned int *d_histo) {

    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < N) atomicAdd(&d_histo[d_buffer[n]], 1);
}

__global__ void histParallelism(
    const unsigned char *d_buffer, 
    const int N, 
    unsigned int *d_histo) {

    int n = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ unsigned int s_histo[256];
    s_histo[threadIdx.x] = 0;
    __syncthreads();
    const int offset = gridDim.x * blockDim.x;
    for (; n<N; n+=offset) {
        atomicAdd(&s_histo[d_buffer[n]], 1); 
    }
    __syncthreads();
    atomicAdd(&d_histo[threadIdx.x], s_histo[threadIdx.x]);
}

__global__ void histshared(
    const unsigned char *d_buffer, 
    const int N, 
    unsigned int *d_histo) {

    int n = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ unsigned int s_histo[256];
    s_histo[threadIdx.x] = 0;
    __syncthreads();
    if (n < N) {
        atomicAdd(&s_histo[d_buffer[n]], 1);
        __syncthreads();
        if (s_histo[threadIdx.x]) atomicAdd(&d_histo[threadIdx.x], s_histo[threadIdx.x]);
    }
}

void hist(
    unsigned char *h_buffer,
    unsigned char *d_buffer,
    unsigned int *h_histo,
    unsigned int *d_histo,
    const int N, 
    const int method) {

    switch (method)
    {
    case 0:
        histCpu(h_buffer, h_histo, N);
        break;
    case 1:
        histKernel<<<(N - 1) / 256 + 1, 256>>>(d_buffer, N, d_histo);
        break;
    case 2:
        histParallelism<<<10240, 256>>>(d_buffer, N, d_histo);
        break;
    case 3:
        histshared<<<(N - 1) / 256 + 1, 256>>>(d_buffer, N, d_histo);
        break;
    
    default:
        break;
    }
}

void timing(
    unsigned char *h_buffer,
    unsigned char *d_buffer,
    unsigned int *h_histo,
    unsigned int *d_histo,
    const int hmem,
    const int N, 
    const int method) {

    float tSum = 0.0;
    float t2Sum = 0.0;
    for (int i=0; i<NUM_REAPEATS; ++i) {
        for (int i=0; i<256; i++) h_histo[i] = 0;
        CHECK(cudaMemcpy(d_histo, h_histo, hmem, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        hist(h_buffer, d_buffer, h_histo, d_histo, N, method);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsedTime;
        CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        tSum += elapsedTime;
        t2Sum += elapsedTime * elapsedTime;
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    float tAVG = tSum / NUM_REAPEATS;
    float tERR = sqrt(t2Sum / NUM_REAPEATS - tAVG * tAVG);
    printf("Time = %g +- %g ms.\n", tAVG, tERR);
    if (method > 0) CHECK(cudaMemcpy(h_histo, d_histo, hmem, cudaMemcpyDeviceToHost));
}


int main() {
    unsigned char *h_buffer = new unsigned char[N];
    bigRandomBlock(h_buffer, N);
    unsigned int *h_histo = new unsigned int[256];
    for (int i=0; i<256; i++) h_histo[i] = 0;
    unsigned char *d_buffer;
    unsigned int *d_histo;
    const int bmem = sizeof(unsigned char) * N;
    const int hmem = sizeof(unsigned int) * 256;
    CHECK(cudaMalloc((void **)&d_buffer, bmem));
    CHECK(cudaMalloc((void **)&d_histo, hmem));
    CHECK(cudaMemcpy(d_buffer, h_buffer, bmem, cudaMemcpyHostToDevice));
    printf("Using cpu:\n");
    timing(h_buffer, d_buffer, h_histo, d_histo, hmem, N, 0);
    for (int i=0; i<256; i+=16) printf("%d  ", h_histo[i]);
    printf("\n");

    printf("Using kernel:\n");
    timing(h_buffer, d_buffer, h_histo, d_histo, hmem, N, 1);
    for (int i=0; i<256; i+=16) printf("%d  ", h_histo[i]);
    printf("\n");

    printf("Using kernel Parallelism:\n");
    timing(h_buffer, d_buffer, h_histo, d_histo, hmem, N, 2);
    for (int i=0; i<256; i+=16) printf("%d  ", h_histo[i]);
    printf("\n");

    printf("Using kernel shared:\n");
    timing(h_buffer, d_buffer, h_histo, d_histo, hmem, N, 3);
    for (int i=0; i<256; i+=16) printf("%d  ", h_histo[i]);
    printf("\n");

    delete[] h_buffer;
    delete[] h_histo;
    CHECK(cudaFree(d_buffer));
    CHECK(cudaFree(d_histo));
    
    return 0;
}
```