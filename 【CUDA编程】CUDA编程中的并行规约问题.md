#! https://zhuanlan.zhihu.com/p/646998011
# 【CUDA编程】CUDA编程中的并行规约问题
**写在前面**：本文笔者从一个简单的规约问题开始，逐步介绍CUDA编程中的一些高阶加速技巧。本文开发环境：windows10 + CUDA10.0 + driver Versin 456.71。硬件环境：Intel Core 9600KF + RTX 2070SUPER。

## 1 规约
对于一个数组 $x = [x_1, x_2, x_3, ..., x_n]$，规约问题是的是计算如下表达式的值：
$$
reduceResult = x_1 \bigoplus x_2 \bigoplus x_3 \bigoplus x_4 ...x_n
$$
其中，$\bigoplus$ 表示二元运算符，例如最大值、最小值、求和、平方和、逻辑运算（与、或）等。在上诉表达式中，运算的结合律是成立的，也就是可以按照任意顺序进行计算。本文我们以求和运算为例，即，
$$
reduce\_sum = \sum _{i=1}^{n} x_i
$$

## 2 基于CPU编程的规约算法
考虑一个有 `N` 个元素的数组 `x`，假如我们需要对该数组规约求和。下面笔者给出了一个实现该计算的c++代码，计算思路比较简单，无须赘述。
```cuda
void reduceCpu(
    const double *h_x,
    double *h_y,
    const int N) {

    for (int i=0; i<N; ++i) h_y[0] += h_x[i];
}
```
在这个例子中，我们考虑一个长度为 `1024 * 1024 * 100` 的一维数组，在主函数中我们将数组 `x` 的每个元素初始化为 `1.23`。接着调用 `timing` 函数对 `reduce` 函数进行计时，为了计时方便，咱们重复运行20次，使用 `nvcc -O3 reduce.cpp -o reduce.exe` 编译后运行 `reduce.exe`，该程序输出：
```cuda
Using CPU:
sum = 128974848.131039
Time = 102.176 +- 2.2898 ms.
```
该结果前 `2` 位正确，从第 `3` 位开始有误差，总共耗时 `102.18ms` 左右。

## 3 折半规约
对于数组规约的并行计算问题，我们要从一个数组出发得到一个数，所以必须采用某种迭代方案。比如，我们可以将数组后半部分的各个元素与前半部分的元素一一相加，不断重复此过程那么在 `logN` 次迭代之后，最后数组的第一个元素就是最初数组中各个元素的和。这就是所谓的折半规约法。  
折半规约示意图如下：
![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5r0TzDaD1mX1IRavuO3ibvBrT3ricLjf1xljufNmZJae3mgVEwsFkwIfnIqp0PdOL0SInOAmQHjdO7Q/0?wx_fmt=png)

## 4 基于GPU共享内存的规约算法
笔者在前面两篇文章中介绍过全局内存和共享内存的特性和应用，由于仅使用全局内存的代码较为简单，且运行速度较慢，本文将不讨论仅使用全局内存的算法。这里笔者给出一个基于共享内存的算法作为baseline。具体核函数代码如下：
```cuda
__global__ void reduceAtomic(
    const double *d_x,
    double *d_y,
    const int N) {

    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ double s_y[];
    s_y[threadIdx.x] = n < N ? d_x[n] : 0.0; 
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&(d_y[0]), s_y[0]);
}
```
首先定义了一个共享内存数组 `s_y`，将全局内存数组 `d_x` 中的数据复制到共享内存中。然后调用函数 `__syncthreads()` 进行线程块内的同步，注意在利用共享内存进行线程之间的通信之前，都要进行同步，确保共享内存变量中的数据对线程块内的所有线程来说都准备就绪。这里如果不调用函数 `__syncthreads()` 则很有可能后面执行 `s_y[threadIdx.x] += s_y[threadIdx.x + offset]` 时 `s_y[threadIdx.x + offset]` 还未被初始化为全局内存中的数据，导致计算错误。  
根据折半规约的基本思想，通过一个循环将线程块内的数据都规约到共享内存变量的第一个元素 `s_y[0]` 中，因为共享内存变量的生命周期仅在核函数内，所以必须在核函数结束前将共享内存中的结果保存到全局内存。这里通过一个原子操作在线程ID为 `0` 时将结果更新到 `d_y[0]` 中。很显然，这里会有多达 `gridSize` 个线程轮流对 `d_y[0]` 进行更新，效率不会太高，我们后面再进行优化。先看一下运行时间：
```cuda
Using shared mem:
sum = 128974847.999872
Time = 12.8501 +- 1.1987 ms.
```
可见，计算精度与CPU代码相差无几，但是运行速度提升了 `8` 倍，运行时间约 `12.85ms`。

## 5 用线程束同步函数代替线程块同步函数
注意到，前面基于共享内存的代码中，我们多次使用了线程块内同步函数 `__syncthreads()`，该函数可以确保线程块内所有线程都执行到了同一个位置，如果某些线程执行速度较快，那么这些线程将等待其他线程执行到指定位置，才能一起往下执行，我们知道线程块内线程数量最大可以为 `1024` ，所以这个函数代价是很大的。那么有没有一种更为廉价的同步方式呢？  
从硬件上来看，一个GPU被分为若干个流多处理器（SM），不同型号的GPU通常具有不同数目的SM。核函数中定义的线程块在执行时将被分配到还没有完全占满的SM中，一个线程块中的线程不会被分到不同的SM中，而总是在一个SM中，但一个SM可以包含多个线程块。不同线程块在SM上可以并发或者顺序执行，但是一般情况下他们之间不能同步。从更细的粒度来看，一个SM以32个线程为单位产生、管理、调度、执行线程，这样的32个线程称为一个线程束。即一个SM可以处理多个线程块，一个线程块又可以分为多个线程束，例如一个含有128线程的线程块将被分为4个线程束。线程束内部是具有同步机制的。  
在我们的规约问题中，当所涉及的线程都在一个线程束内时，可以将线程块同步函数 `__syncthreads()` 换成一个更加廉价的线程束同步函数 `__syncwarp()`，具体核函数代码如下：
```cuda
__global__ void reduceWarp(
    const double *d_x, 
    double *d_y, 
    const int N) {
    
    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ double s_y[];
    s_y[threadIdx.x] = n < N ? d_x[n] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncwarp();
    }
    if (threadIdx.x == 0) atomicAdd(&d_y[0], s_y[0]);
}
```
也就是说，当 `offset >= 32` 时，这时候参与的线程不在一个线程束内，我们在每一次折半规约后使用线程块同步函数  `__syncthreads()` 同步；当 `offset <= 16` 时，参与的线程极其需要同步的线程都在一个线程束内，我们在每一次折半规约后可以使用束内同步函数 `__syncwarp()` 执行同步。下面我们看一下运行时间：
```cuda
Using shared mem and warp:
sum = 128974847.999872
Time = 10.1623 +- 0.324607 ms.
```
相比只使用 线程块同步函数，加入束内同步函数后大概快了 `20%`，耗时 `10.16ms`，也是一个比较可观的优化了。但是缺点也很明显，代码明显变得复杂了不少，且可读性有所降低。  

## 6 利用线程束洗牌函数进行优化
笔者前面两篇文章介绍了共享内存、全局内存的使用，两者均可用于线程间的通信合作。今天我们来研究一个比较特殊的机制，叫做线程束洗牌指令。支持线程束洗牌指令的设备最低也要3.0以上。  
洗牌指令（shuffle instruction）作用在线程束内，允许两个线程见相互访问对方的寄存器。这就给线程束内的线程相互交换信息提供了了一种新的渠道，我们知道，核函数内部的变量优先存在寄存器中，一个线程束可以看做是32个内核并行执行，换句话说这32个核函数中寄存器变量在硬件上其实都是邻居，这样就为相互访问提供了物理基础，线程束内线程相互访问数据不通过共享内存或者全局内存，使得通信效率高很多，线程束洗牌指令传递数据，延迟极低，且不消耗内存，因此线程束洗牌指令是线程束内线程通讯的极佳方式。
根据我们折半规约的基本思想，在低线程号的线程中往往需要访问高线程号对应的共享内存变量的元素，这时候我们可以使用寄存器变量代替共享内存变量，通过束内下移指令访问高线程号的线程寄存器内的数据，具体核函数代码如下：
```cuda
__global__ void reduceShfl(
    const double *d_x, 
    double *d_y, 
    const int N) {

    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ double s_y[];
    s_y[threadIdx.x] = n < N ? d_x[n] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncthreads();
    }
    double y = s_y[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1) {
        y += __shfl_down_sync(0xffffffff, y, offset);
    }
    if (threadIdx.x == 0) atomicAdd(&d_y[0], y);
}
```
其中 `__shfl_down_sync(mask, v, d, w=32)` 的作用是，在标号为 `t` 的参与线程中返回标号为 `t + d` 的线程中变量 `v` 的值，当标号 `t + d >= w` 时返回原来的 `v`，`mask` 表示参与线程的掩码，是一个32位无符号整数，`1` 表示参与，`0` 表示不参与，`mask = 0xffffffff` 表示全部线程都参与。简单来说，这是一种线程束内数据下移操作。  
在核函数 `reduceShfl` 中当 `offset <= 16`时，参与线程都在一个线程束内，我们定义一个寄存器变量 `y = s_y[threadIdx.x]` 取代共享内存进行线程间的通信。然后在每一轮折半规约时，将高线程寄存器变量的数据，加到参与线程的寄存器变量上。值得注意的是，束内洗牌指令可以自动处理同步“读-写”竞争问题，对所有参与线程来说，洗牌指令总是先读取各个线程中 `y` 的值，然后再将洗牌操作的结果写入到各个线程中的 `y`，因此无需再调用束内同步函数。编译后运行结果如下：
```cuda
Using shared mem and shfl:
sum = 128974847.999872
Time = 10.5641 +- 0.363848 ms.
```
结果显示使用束内洗牌指令后，耗时 `10.56ms`，运行时间反而比上面共享内存+束内同步函数机制要长一些，没有起到优化效果。原因可能是本身共享内存读写已经很快了，使用束内洗牌指令虽然理论上更快一些但是在我们的核函数内使用洗牌指令的计算太少，导致加速并不明显。

## 7 利用协作组进行规约计算
我们知道，在并行算法中，免不了要进行线程间的协作，要协作，就必须有同步机制。**协作组**可以看做线程块和线程束同步机制的推广，他提供了更为灵活的线程协作方式，包括线程块内部的同步与协作，线程块之间的同步与协作，甚至是设备间的同步与协作。简单来说就是从软件层面提供了一个新的概念（协作组），通过协作组开发人员可以以其他粒度定义和同步线程组，以前开发人员如果需要表达更广泛的并行交互模式，只能通过自己临时编写的代码来组合调用内部函数来达到效果，有了协作组就可以直接调用协作组提供的方法来进行线程间的协作而无需关心其底层实现机制。协作组相关的数据类型和函数都定义在命名空间 `cooperative_groups` 内。下面我们将基于束内洗牌指令的核函数通过协作组实现，具体核函数代码如下：
```cuda
__global__ void reduceCG(
    const double *d_x, 
    double *d_y, 
    const int N) {
    
    namespace cg = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();
    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ double s_y[];
    s_y[threadIdx.x] = n < N ? d_x[n] : 0.0;
    g.sync();
    for (int offset = g.size() >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        g.sync();
    }
    double y = s_y[threadIdx.x];
    cg::thread_block_tile<32> g32 = cg::tiled_partition<32>(g);
    for (int offset = g32.size() >> 1; offset > 0; offset >>= 1) {
        y += g32.shfl_down(y, offset);
    }
    if (threadIdx.x == 0) atomicAdd(&d_y[0], y);
}
```
核函数中，我们先定义了一个线程组 `g`，他就是我们非常熟悉的线程块，只不过这里把他包装成一个类型，如果想对 `g` 内所有线程执行同步操作，我们直接调用 `g.sync()` 即可，在这里其底层完全等价于 `__syncthreads()`。后面为了使用束内洗牌指令进行优化，我们可以调用 `tiled_partition<32>()` 函数模板将线程组 `g` 划分为更小粒度的线程组 `g32`，其底层完全等价于线程束，`thread_block_tile<32>` 也提供了对应于束内洗牌指令的方法，如 `_shfl_down` 底层完全等价于 `__shfl_down_sync(mask, v, d, w=32)` 只不过少了两个参数：掩码和宽度，因为线程组内所有线程都必须参与计算，所以无需掩码，同时线程块片的大小本身就是宽度，所以也无需传入宽度参数。运行结果如下：
```cuda
Using shared mem and cooperative groups:
sum = 128974847.999872
Time = 10.2634 +- 0.372888 ms.
```
协作组版本运行时间为 `10.26ms`，可以看到截止目前我们的算法优化进入到了一个瓶颈，在不改变思路的情况下，使用了很多高阶技巧但都没有更大的提升。那么如何突破这个限制？


## 8 提高线程利用率
在笔者上一篇文章[【CUDA编程】CUDA并行化中的直方图问题](https://zhuanlan.zhihu.com/p/646997739) 中提到，可以通过预先在一个线程中处理多个数据的方式提高线程利用率来进一步加速GPU计算。  
我们注意到，在规约核函数中，线程利用率并不高，当我们使用大小为 `1024` 的线程块时，若 `offset = 512` 时，只用了 `1/2` 的线程，其余线程闲置；若 `offset = 256` 时，只用了 `1/4` 的线程，其余线程闲置。最终当 `offset = 1` 时，只用了 `1/1024` 的线程进行计算，其余线程闲置。规约过程一共用了 `log1024 = 10` 步，所以规约过程中平均利用率只有 `(1/2 + 1/4 + ...)/10 = 1/10`，这会造成很大的资源浪费。  
如果能够在规约之前进行一部分计算，应该可以从整体上提高线程利用率，让所有的线程尽量有事可做。具体地，我们可以取 `gridSize=10240`，这样 `gridSize * blockSize << N`，定义一个步长 `offset = gridSize * blockSize`，在一个线程内我们先按步长将全局内存变量中的数据全部累加进共享内存数组的一个元素中。具体实现如下：
```cuda
__global__ void reduceParallelism(
    const double *d_x, 
    double *d_y, 
    const int N) {

    namespace cg = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ double s_y[];
    double y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (; n < N; n+=stride) y += d_x[n];
    s_y[threadIdx.x] = y;
    g.sync();
    for (int offset = g.size() >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        g.sync();
    }
    y = s_y[threadIdx.x];
    cg::thread_block_tile<32> g32 = cg::tiled_partition<32>(cg::this_thread_block());
    for (int offset = g32.size() >> 1; offset > 0; offset >>= 1) {
        y += g32.shfl_down(y, offset);
    }
    if (threadIdx.x == 0) d_y[blockIdx.x] = y;
}
```
为了尽可能优化加速，我们在一开始定义了一个寄存器变量 `y`，用来在循环中对读取的全局内存数据进行累加，然后把 `y` 赋值给全局内存数组元素，这样会更高效一些。最后相对于之前使用原子操作将规约结果直接加到全局内存变量 `d_y[0]` 上，这里我们只是将规约结果写入 `d_y[blockIdx.x]` 中，这样做有什么优势？  
笔者在上一篇文章中介绍过，原子操作是一个线程一个线程轮流执行“读-改-写” 操作，这样会导致整个线程网格内，所有线程块的第一个线程会排队轮流读写 `d_y[0]`，这种串行操作代价是很大的。换一种思路，如果我们将线程块内规约结果分别写入一个长度为线程块总数量的全局内存变量中是不是就不存在读写竞争问题了？这样在一次规约后我们把一个长度为 `N` 的数组规约到了一个长度为 `gridSize` 的数组 `d_y` 中，然后我们再把长度为 `gridSize` 的数组 `d_y` 规约到一个值即可。调用核函数的方式如下：
```cuda
reduceParallelism<<<10240, BLOCK_SIZE, sizeof(double)*BLOCK_SIZE>>>(d_x, d_tmp, N);
reduceParallelism<<<1, 1024, sizeof(double)*1024>>>(d_tmp, d_y, 10240);
```
先把 `d_x` 规约到一个临时变量 `d_tmp` 中，再把 `d_tmp` 规约到 `d_y`，运行时间如下：
```cuda
Using shared mem, cooperative groups and parallelism:
sum = 128974848.000000
Time = 2.36651 +- 0.248117 ms.
```
在笔者的RTX 2070SUPER中测试，优化后运行时间仅为 `2.36ms` 相比之前的版本提升了4倍，性能提升显著。

## 9 小结
本文内容总结如下：
- 一个SM以32个线程为单位产生、管理、调度、执行线程，这样的32个线程称为一个线程束。当所涉及的线程都在一个线程束内时，可以将线程块同步函数 `__syncthreads()` 换成一个更加廉价的线程束同步函数 `__syncwarp()`。
- 核函数内部的变量优先存在寄存器中，一个线程束中寄存器变量在硬件上其实都是邻居，这样就为相互访问提供了物理基础，线程束内线程相互访问数据不通过共享内存或者全局内存，使得通信效率高很多，线程束洗牌指令传递数据，延迟极低，且不消耗内存，因此线程束洗牌指令是线程束内线程通讯的极佳方式。
- 协作组提供了更为灵活的线程协作方式，包括线程块内部的同步与协作，线程块之间的同步与协作，甚至是设备间的同步与协作。通过协作组开发人员可以以其他粒度定义和同步线程组，直接调用协作组提供的方法来进行线程间的协作而无需关心其底层实现机制。
- 当性能优化进入一个瓶颈时，高阶优化技巧不一定能起到显著效果，可以将思路放在计算方法的优化上，提升线程利用率，也许能起到很好的效果。

## 附录
本文代码如下：
```cuda
#include "error.cuh"
#include "stdio.h"
#include <cooperative_groups.h>

const int N = 1024 * 1024 * 100;
const int M = sizeof(double) * N;
const int NUM_REAPEATS = 20;
const int BLOCK_SIZE = 1024;


void reduceCpu(
    const double *h_x,
    double *h_y,
    const int N) {

    for (int i=0; i<N; ++i) h_y[0] += h_x[i];
}

__global__ void reduceAtomic(
    const double *d_x,
    double *d_y,
    const int N) {

    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ double s_y[];
    s_y[threadIdx.x] = n < N ? d_x[n] : 0.0; 
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&(d_y[0]), s_y[0]);
}

__global__ void reduceWarp(
    const double *d_x, 
    double *d_y, 
    const int N) {
    
    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ double s_y[];
    s_y[threadIdx.x] = n < N ? d_x[n] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncwarp();
    }
    if (threadIdx.x == 0) atomicAdd(&d_y[0], s_y[0]);
}

__global__ void reduceShfl(
    const double *d_x, 
    double *d_y, 
    const int N) {

    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ double s_y[];
    s_y[threadIdx.x] = n < N ? d_x[n] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncthreads();
    }
    double y = s_y[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1) {
        y += __shfl_down_sync(0xffffffff, y, offset);
    }
    if (threadIdx.x == 0) atomicAdd(&d_y[0], y);
}

__global__ void reduceCG(
    const double *d_x, 
    double *d_y, 
    const int N) {
    
    namespace cg = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();
    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ double s_y[];
    s_y[threadIdx.x] = n < N ? d_x[n] : 0.0;
    g.sync();
    for (int offset = g.size() >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        g.sync();
    }
    double y = s_y[threadIdx.x];
    cg::thread_block_tile<32> g32 = cg::tiled_partition<32>(g);
    for (int offset = g32.size() >> 1; offset > 0; offset >>= 1) {
        y += g32.shfl_down(y, offset);
    }
    if (threadIdx.x == 0) atomicAdd(&d_y[0], y);
}

__global__ void reduceParallelism(
    const double *d_x, 
    double *d_y, 
    const int N) {

    namespace cg = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ double s_y[];
    double y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (; n < N; n+=stride) y += d_x[n];
    s_y[threadIdx.x] = y;
    g.sync();
    for (int offset = g.size() >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        g.sync();
    }
    y = s_y[threadIdx.x];
    cg::thread_block_tile<32> g32 = cg::tiled_partition<32>(cg::this_thread_block());
    for (int offset = g32.size() >> 1; offset > 0; offset >>= 1) {
        y += g32.shfl_down(y, offset);
    }
    if (threadIdx.x == 0) d_y[blockIdx.x] = y;
}

void reduce(
    const double *h_x,
    const double *d_x,
    double *h_y,
    double *d_y,
    double *d_tmp,
    const int N,
    const int method) {

    switch (method)
    {
    case 0:
        reduceCpu(h_x, h_y, N);
        break;
    case 1:
        reduceAtomic<<<(N-1)/BLOCK_SIZE+1, BLOCK_SIZE, sizeof(double)*BLOCK_SIZE>>>(d_x, d_y, N);
        break;
    case 2:
        reduceWarp<<<(N-1)/BLOCK_SIZE+1, BLOCK_SIZE, sizeof(double)*BLOCK_SIZE>>>(d_x, d_y, N);
        break;
    case 3:
        reduceShfl<<<(N-1)/BLOCK_SIZE+1, BLOCK_SIZE, sizeof(double)*BLOCK_SIZE>>>(d_x, d_y, N);
        break;
    case 4:
        reduceCG<<<(N-1)/BLOCK_SIZE+1, BLOCK_SIZE, sizeof(double)*BLOCK_SIZE>>>(d_x, d_y, N);
        break;
    case 5:
        reduceParallelism<<<10240, BLOCK_SIZE, sizeof(double)*BLOCK_SIZE>>>(d_x, d_tmp, N);
        reduceParallelism<<<1, 1024, sizeof(double)*1024>>>(d_tmp, d_y, 10240);
        break;
    
    default:
        break;
    }
}

void timing(
    const double *h_x,
    const double *d_x,
    double *h_y,
    double *d_y,
    double *d_tmp,
    const int N, 
    const int method) {
    
    float tSum = 0.0;
    float t2Sum = 0.0;
    for (int i=0; i<NUM_REAPEATS; ++i) {
        h_y[0] = 0.0;
        CHECK(cudaMemcpy(d_y, h_y, sizeof(double), cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        reduce(h_x, d_x, h_y, d_y, d_tmp, N, method);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsedTime;
        CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        tSum += elapsedTime;
        t2Sum += elapsedTime * elapsedTime;
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    if (method > 0) CHECK(cudaMemcpy(h_y, d_y, sizeof(double), cudaMemcpyDeviceToHost));
    float tAVG = tSum / NUM_REAPEATS;
    float tERR = sqrt(t2Sum / NUM_REAPEATS - tAVG * tAVG);
    printf("sum = %f \n", h_y[0]);
    printf("Time = %g +- %g ms.\n", tAVG, tERR);
}



int main() {
    double *h_x = new double[N];
    double h_y[1] = {0.0};
    for (int i=0; i<N; i++) h_x[i] = 1.23;
    double *d_x, *d_y;
    CHECK(cudaMalloc((void **)&d_x, M));
    CHECK(cudaMalloc((void **)&d_y, sizeof(double)));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    double *d_tmp;
    CHECK(cudaMalloc((void **)&d_tmp, sizeof(double) * 10240)); 

    printf("Using CPU:\n");
    timing(h_x, d_x, h_y, d_y, d_tmp, N, 0);
    
    printf("Using shared mem:\n");
    timing(h_x, d_x, h_y, d_y, d_tmp, N, 1);

    printf("Using shared mem and warp:\n");
    timing(h_x, d_x, h_y, d_y, d_tmp, N, 2);

    printf("Using shared mem and shfl:\n");
    timing(h_x, d_x, h_y, d_y, d_tmp, N, 3);

    printf("Using shared mem and cooperative groups:\n");
    timing(h_x, d_x, h_y, d_y, d_tmp, N, 4);

    printf("Using shared mem, cooperative groups and parallelism:\n");
    timing(h_x, d_x, h_y, d_y, d_tmp, N, 5);


    delete[] h_x;
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_tmp));
    return 0;
}
```