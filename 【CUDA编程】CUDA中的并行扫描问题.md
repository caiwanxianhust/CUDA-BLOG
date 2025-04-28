#! https://zhuanlan.zhihu.com/p/647000368
# 【CUDA编程】CUDA中的并行扫描问题
**写在前面**：笔者之前阅读tensorflow2.x文档时，发现一些不常用的网络层和优化器官网没有给出API接口，虽然tensorflow提供了可供开发者自定义网络层、优化函数的基类，但是这种自定义算子的操作如果使用Python代码实现，效率偏低，然后笔者趁此机会学习了一下CUDA编程，希望了解一下GPU下的编程知识。本文是笔者使用CUDA编程对扫描算法的一个简单实现，为什么是扫描呢，因为像扫描、规约这一类算法天生就很难实现并行，因为其每个元素的计算过程严重依赖其他元素。本文开发环境：windows10 + CUDA10.0 + driver Versin 456.71

## 1 扫描
前缀和（prefix sum）也常称为扫描（scan），无论在CPU编程或是GPU编程中都是一项不太容易并行的任务。扫描操作定义如下：  
给定一个序列 $x_1, x_2, x_3, ..., x_n$，扫描操作后的输出序列为 $y_1, y_2, y_3, ..., y_n$，根据前缀和是否包含当前元素，分为包含扫描和非包含扫描，分别定义如下：  
$$
包含扫描：y_n = \sum _{i=0} ^n x_i
$$
$$
非包含扫描：y_n = \sum _{i=0} ^{n-1} x_i
$$
看起来还不够直观，举例如下，假设有一个序列：  
`1, 2, 3, 4, 5, 6, ...`  
对其进行包含扫描和非包含扫描后结果为：  
`1, 3, 6, 10, 15, 21, ...`  
和
`0, 1, 3, 6, 10, 15, ...`  
以上是针对加法进行算术操作的，也可以针对其他算术定义扫描操作。例如，将加法换成乘法，包含扫描的结果就变为：  
`1, 2, 6, 24, 120, 720, ...`

## 2 Hillis Steele Scan（并行前缀扫描算法）
Hillis Steele Scan算法，也称为并行前缀扫描算法，是一种以并行方式运行的扫描操作算法。以长度为 `N` 的数组 `x[]` 的并行扫描算法为例：
![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5ogREu3PwJ1Gzjl0BxJLmFdrWOHDemcZWZx9fvxqZpFib1OJIyW8cqdIDTknJblyE1KxE9ssgvGdibQ/0?wx_fmt=png)
- 迭代次数 $d \in [1, log_2N]$ 
- 每轮迭代所有元素并行操作
- 每次迭代中元素 $x_d[i] = x_{d-1} + x_{d-1}[i-2^d]$
- 迭代 $log_2N$ 次后得到的序列即为扫描结果。

## 3 基于CPU编程的扫描算法
考虑一个有 `N` 个元素的数组 `x`，假如我们需要计算该数组的前缀和。下面笔者给出了一个实现该计算的c++代码，计算思路比较简单，无须赘述。
```cpp
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int N = 1000000;
const int NUM_REAPEATS = 20;

void scan(const real *x, real *y) {
    for (int i=0; i<N; i++) {
        y[i] = (i == 0) ? x[i] : y[i-1] + x[i];
    }
}

void timing(const real *x, real *y) {
    float tSum = 0.0;
    float t2Sum = 0.0;
    for (int i=0; i<NUM_REAPEATS; ++i) {
        clock_t start, end; 
        start = clock();
        scan(x, y);
        end = clock();
        float elapsedTime = (float)(end - start) / CLOCKS_PER_SEC * 1000;
        tSum += elapsedTime;
        t2Sum += elapsedTime * elapsedTime;
    }
    float tAVG = tSum / NUM_REAPEATS;
    float tERR = sqrt(t2Sum / NUM_REAPEATS - tAVG * tAVG);
    printf("Time = %g +- %g ms.\n", tAVG, tERR);
}

int main() {
    real *x = new real[N];
    real *y = new real[N];
    for (int i=0; i<N; i++) x[i] = 1.23;
    timing(x, y);
    for (int i = N - 10; i < N; i++) printf("%f  ", y[i]);
    delete[] x;
    delete[] y;

    return 0;
}

```
在这个例子中，我们考虑一个长度为 `1e6` 的一维数组，在主函数中我们将数组 `x` 的每个元素初始化为 `1.23`。接着调用 `timing` 函数对 `scan` 函数进行计时，为了计时方便，咱们重复运行20次，使用 `g++ -O3 scanCpu.cpp -o scanCpu.exe` 编译后运行 `scanCpu.exe`，该程序输出：
```cpp
Time = 2.15 +- 0.357071 ms.
1239312.125000  1239313.375000  1239314.625000  1239315.875000  1239317.125000  1239318.375000  1239319.625000  1239320.875000  1239322.125000  1239323.375000
```
该结果前 `3` 位正确，从第 `4` 位开始有误差，总共耗时 `2.15ms` 左右。

## 4 基于GPU全局内存的扫描算法
我们知道，一块CPU通常只拥有少数几个快速的计算核心，比如笔者的Intel Core 9600KF就是6核12线程的，其芯片结构决定了最多只能支持12个线程并行计算，CPU中更多的晶体管是用于数据缓存和流程控制。而一块典型的GPU拥有几百到几千个不那么快速的计算核心，比如笔者的Nvidia RTX 2070SUPER有2560个CUDA核心，所以与CPU编程思想不同，由于GPU的多核心特性，GPU编程要充分利用其并行优势，尽量让每个核心都同时参与计算工作。  
全局内存是在GPU芯片外的内存（注意这里是显卡的内存和我们平常说的内存不是一个），主要负责主机与设备及设备与设备之间传递数据，该内存对所有线程都可见，可读可写，是GPU中容量最大的内存，同时也是延迟最高的内存。我们平常使用 `cudaMalloc()` 函数分配的内存就是全局内存，全局内存的生命周期由主机端控制。  
如果数组长度较小（`N<=1024`）可以在一个线程块完成计算，那么可以给出以下仅使用全局内存的核函数代码：
```cuda
__global__ void globalMemScan(real *d_x) {
    real y = 0.0;
    for (int offset = 1; offset < N; offset <<= 1) {
        if (threadIdx.x >= offset) y = d_x[threadIdx.x] + d_x[threadIdx.x - offset];
        __syncthreads();
        if (threadIdx.x >= offset) d_x[threadIdx.x] = y;
    }
}
```
可以看到，代码的基本思想就是，每个线程分别执行核函数 `globalMemScan`，各自完成自己的计算，每个线程只专注于数组中一个元素的更新迭代，`threadIdx` 表示线程 `ID`。  
这里为什么不能使用 `d_x[threadIdx.x] += d_x[threadIdx.x - offset]` 呢？因为对于多线程程序，两个不同线程中指令的执行次序可能可代码中展现的次序不同，有可能第 `3` 个线程已经执行完第 `n` 轮扫描进行下一轮了，第 `4` 个线程还在执行第 `n` 轮，这样会导致第 `4` 个线程读取 `x[2]` 时，读取的是已经更新后的 `x[2]`，导致计算错误，所以这里我们在寄存器申请一块内存定义一个变量 `y` 暂时存放 `d_x[threadIdx.x] + d_x[threadIdx.x - offset]`，然后调用函数 `__syncthreads()`，该函数是CUDA提供的内置函数，为线程块同步函数，保证线程块内所有线程执行到同一个位置，完成第 `n` 轮计算后，再更新 `x[i]`的值，然后再进入下一轮。

那么如果数组长度 `N>1024` 怎么办呢？CUDA中对线程块大小做了限制，最多不超过 `1024`，我们显然不可能只弄一个线程块了，这里我们可以分两步走：第一步，在每个线程块独立扫描，得到的结果为线程块内的扫描结果，并将每个线程块的规约值记录在 `d_y`中。第二步，对 `d_y` 进行扫描。第三步，将分组扫描结果与 `d_y`扫描结果相加，相当于每个线程块加上一个基准值，这个基准值为 `d_y[blockIdx.x - 1]` ，得到最终的整体扫描结果。具体核函数代码如下：
```cuda
__global__ void globalMemScan(real *d_x, real *d_y) {
    real *x = d_x + blockDim.x * blockIdx.x;
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    real y = 0.0;
    if (n < N) {
        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            if (threadIdx.x >= offset) y = x[threadIdx.x] + x[threadIdx.x - offset];
            __syncthreads();
            if (threadIdx.x >= offset) x[threadIdx.x] = y;
        }
        if (threadIdx.x == blockDim.x - 1) d_y[blockIdx.x] = x[threadIdx.x];
    } 
}

__global__ void addBaseValue(real *d_x, real *d_y) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    real y = blockIdx.x > 0 ? d_y[blockIdx.x - 1] : 0.0;
    if (n < N) {
        d_x[n] += y;
    } 
}
```
核函数调用代码如下：
```cuda
globalMemScan<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y);
globalMemScan<<<1, GRID_SIZE>>>(d_y, d_z);
addBaseValue<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y);
```
- 现将长度为 `N` 的数组分成 `GRID_SIZE` 组，在组内进行扫描，得到分组扫描结果，并将组内元素和存入 `d_y` 中。
- 将长度为 `GRID_SIZE` 的数组 `d_y` 进行整体扫描，`d_z` 是元素和，没什么用处。
- 将分组扫描结果与 `d_y`扫描结果相加。
  

示意图如下：
![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5pWDpXOfkYEP6wXEUPgNK9sjXjicyfA8cwmypNMIp2MNK2JewNwABTCWfex8S5OXcoqHCcI5oON64A/0?wx_fmt=png)

用命令 `nvcc -arch=sm_75 scan.cu -o scan.exe` 编译后运行 `scan.exe`，可以得到在RTC 2070SUPER设备下使用单精度浮点数时，仅使用全局内存的CUDA代码执行时间为 `0.105752ms` 左右，约为CPU代码耗时的 `1/20`。
```cuda
using global mem:
Time = 0.105752 +- 0.0408563 ms.
1229988.875000  1229990.125000  1229991.375000  1229992.625000  1229993.875000  1229995.000000  1229996.250000  1229997.500000  1229998.750000  1230000.000000
```
可以看出相比于CPU代码计算，GPU在精度上也高不少。

## 5 基于GPU共享内存的扫描算法
和全局内存不同，共享内存位于GPU芯片上，具有仅次于寄存器的读写速度，共享内存对整个线程块可见，其生命周期也与线程块一致，但相对于全局内存，共享内存数量有限，笔者的2070SUPER只有48KB共享内存。  
每个线程块都有一个共享内存变量的副本，各自维护一套共享内存变量的值，一个线程块内所有的线程都可以访问该线程块的共享内存变量副本，但是不能访问其他线程块的共享内存变量副本，共享内存的作用是减少对全局内存的访问，比如对于高频读写任务，可以先申请一块共享内存，把全局内存变量赋值进去，然后对共享内存进行读写，避免了频繁读写高延迟的全局内存。基于共享内存的扫描代码如下：
```cuda
__global__ void sharedMemScan(real *d_x, real *d_y) {
    extern __shared__ real s_x[];
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    s_x[threadIdx.x] = n < N ? d_x[n] : 0.0;
    __syncthreads();
    real y = 0.0;
    if (n < N) {
        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            if (threadIdx.x >= offset) y = s_x[threadIdx.x] + s_x[threadIdx.x - offset];
            __syncthreads();
            if (threadIdx.x >= offset) s_x[threadIdx.x] = y;
        }
        d_x[n] = s_x[threadIdx.x];
        if (threadIdx.x == blockDim.x - 1) d_y[blockIdx.x] = s_x[threadIdx.x];
    } 
}
```
笔者在核函数内使用 `extern __shared__ real s_x[]` 申请了一块动态共享内存，内存大小通过执行配置参数传入。先将全局内存变量 `d_x` 的数据写入共享内存变量 `s_x`，然后对共享内存变量进行迭代操作。其他逻辑与前面全局内存下的代码一致。  
用命令 `nvcc -arch=sm_75 scan.cu -o scan.exe` 编译后运行 `scan.exe`，可以得到在RTC 2070SUPER设备下使用单精度浮点数时，使用共享内存的CUDA代码执行时间为 `0.0950032ms` 左右，约为全局内存代码耗时的 `90%`，提升并不大。一般来说，在核函数中对内存访问次数越多，使用共享内存带来的加速效果才越明显。
```cuda
using shared mem:
Time = 0.0950032 +- 0.0330897 ms.
1229988.875000  1229990.125000  1229991.375000  1229992.625000  1229993.875000  1229995.000000  1229996.250000  1229997.500000  1229998.750000  1230000.000000
```

## 6 使用thrust库函数进行扫描
thrust是一个实现了众多基本并行算法的模板库，该库自动包含在CUDA工具箱中，库中所有函数和类型都在thrust命名空间中定义。用起来非常简单，可以不用编写核函数在主机端直接调用即可。这里给出调用代码。
```cuda
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

thrust::inclusive_scan(thrust::device, d_x, d_x + N, d_x);
```
用命令 `nvcc -arch=sm_75 scan.cu -o scan.exe` 编译后运行 `scan.exe`，可以得到在RTC 2070SUPER设备下使用单精度浮点数时，使用thrust库函数 `thrust::inclusive_scan` 的CUDA代码执行时间为 `0.135622ms` 左右，比我们自己写的核函数代码耗时稍长一些，计算精度也稍差。
```cuda
using thrust lib:
Time = 0.135622 +- 0.194265 ms.
1229988.625000  1229989.875000  1229991.125000  1229992.375000  1229993.625000  1229994.875000  1229995.875000  1229997.125000  1229998.375000  1229999.625000
```
## 7 小结
本文内容总结如下：
- 扫描问题是一项不太容易并行的问题，我们可以通过Hillis Steele Scan算法进行一定程度的并行优化。
- 与CPU编程思想不同，由于GPU的多核心特性，GPU编程要充分利用其并行优势，尽量让每个核心都同时参与计算工作，对于可并行的计算工作，CUDA程序运行速度远比C++程序快。
- 全局内存对所有线程都可见，可读可写，是GPU中容量最大的内存，同时也是延迟最高的内存。
- 共享内存具有仅次于寄存器的读写速度，对整个线程块可见，其生命周期也与线程块一致，但相对于全局内存，共享内存数量有限，对于频繁读写的任务可以使用共享内存加速。
- 受线程块大小的限制如果一次扫描无法完成任务，可以多次扫描，然后求和。
- thrust库相当于C++中的STL，用起来非常简单，可以不用编写核函数在主机端直接调用。对于常见的计算来说，库函数能够获得的性能往往是比较高的，但对于特定问题，使用库函数并不一定能胜过自己实现的代码。

## 附录
本文代码如下：  
scan.cu
```cuda
#include <stdio.h>
#include "error.cuh"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif


const int N = 1000000;
const int M = sizeof(int) * N;
const int NUM_REAPEATS = 20;
const int BLOCK_SIZE = 1024;
const int GRID_SIZE = (N - 1) / BLOCK_SIZE + 1;


__global__ void globalMemScan(real *d_x, real *d_y) {
    real *x = d_x + blockDim.x * blockIdx.x;
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    real y = 0.0;
    if (n < N) {
        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            if (threadIdx.x >= offset) y = x[threadIdx.x] + x[threadIdx.x - offset];
            __syncthreads();
            if (threadIdx.x >= offset) x[threadIdx.x] = y;
        }
        if (threadIdx.x == blockDim.x - 1) d_y[blockIdx.x] = x[threadIdx.x];
    } 
}

__global__ void addBaseValue(real *d_x, real *d_y) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    real y = blockIdx.x > 0 ? d_y[blockIdx.x - 1] : 0.0;
    if (n < N) {
        d_x[n] += y;
    } 
}

__global__ void sharedMemScan(real *d_x, real *d_y) {
    extern __shared__ real s_x[];
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    s_x[threadIdx.x] = n < N ? d_x[n] : 0.0;
    __syncthreads();
    real y = 0.0;
    if (n < N) {
        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            if (threadIdx.x >= offset) y = s_x[threadIdx.x] + s_x[threadIdx.x - offset];
            __syncthreads();
            if (threadIdx.x >= offset) s_x[threadIdx.x] = y;
        }
        d_x[n] = s_x[threadIdx.x];
        if (threadIdx.x == blockDim.x - 1) d_y[blockIdx.x] = s_x[threadIdx.x];
    } 
}


void scan(real *d_x, real *d_y, real *d_z, const int method) {
    switch (method)
    {
    case 0:
        globalMemScan<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y);
        globalMemScan<<<1, GRID_SIZE>>>(d_y, d_z);
        addBaseValue<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y);
        break;
    case 1:
        sharedMemScan<<<GRID_SIZE, BLOCK_SIZE, sizeof(real) * BLOCK_SIZE>>>(d_x, d_y);
        sharedMemScan<<<1, GRID_SIZE, sizeof(real) * GRID_SIZE>>>(d_y, d_z);
        addBaseValue<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y);
        break;
    case 2:
        thrust::inclusive_scan(thrust::device, d_x, d_x + N, d_x);
        break;
    
    default:
        break;
    }
}

void timing(const real *h_x, real *d_x, real *d_y, real *d_z, real *h_ret, const int method) {

    float tSum = 0.0;
    float t2Sum = 0.0;
    for (int i=0; i<NUM_REAPEATS; ++i) {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        scan(d_x, d_y, d_z, method);

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
    CHECK(cudaMemcpy(h_ret, d_x, M, cudaMemcpyDeviceToHost));
}

int main() {
    real *h_x = new real[N];
    real *h_y = new real[N];
    real *h_ret = new real[N];
    for (int i=0; i<N; i++) h_x[i] = 1.23;
    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **)&d_x, M));
    CHECK(cudaMalloc((void **)&d_y, sizeof(real) * GRID_SIZE));
    CHECK(cudaMalloc((void **)&d_z, sizeof(real)));
    
    printf("using global mem:\n");
    timing(h_x, d_x, d_y, d_z, h_ret, 0);
    for (int i = N - 10; i < N; i++) printf("%f  ", h_ret[i]);
    printf("\n");
    printf("using shared mem:\n");
    timing(h_x, d_x, d_y, d_z, h_ret, 1);
    for (int i = N - 10; i < N; i++) printf("%f  ", h_ret[i]);
    printf("\n");
    printf("using thrust lib:\n");
    timing(h_x, d_x, d_y, d_z, h_ret, 2);
    for (int i = N - 10; i < N; i++) printf("%f  ", h_ret[i]);
    printf("\n");

    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    delete[] h_x;
    delete[] h_y;
    delete[] h_ret;
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