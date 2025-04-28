
# 【CUDA编程】CUDA 共享内存及分布式共享内存

**写在前面**：本文详细介绍了 CUDA 编程模型中的共享内存和分布式共享内存，共享内存是一块可由开发人员自行管理的高速缓存，在 CUDA 程序优化加速中起着至关重要的作用。随着线程集群概念的引入，分布式共享内存随之也作为一种集群内数据交换的手段，为 CUDA 程序优化加速工作提供新的途径。

## 1 共享内存 
### 1.1 共享内存基本概念

前面介绍过，共享内存是设备端的内存，属于片上内存（on-the-chip），其读写速度仅次于寄存器，远大于全局内存。从硬件层面来说，共享内存与全局内存不同，它是 SM 级的内存，每个 SM 分别具备一块共享内存，从 Volta 架构开始，L1 缓存、纹理缓存和共享内存三者在物理上被统一成了一个数据缓存，其中共享内存容量可以使用 CUDA Runtime API 进行设置。从软件层面来说，共享内存对整个 block 可见，其生命周期也与 block 一致，也就是说，针对 Kernel 中的共享内存变量，每个 block 拥有一个单独的共享内存变量副本，同一个 block 内的线程访问的是同一个对象，不同 block 的线程访问的是不同的对象。共享内存作为一块可由开发人员自行管理的缓存，主要作用有两个：

- 针对共享内存高带宽的特性，可以将其作为暂存器使用，以最大限度地减少 CUDA 程序中 block 对全局内存的访问。具体地，可以提前将全局内存的数据存入共享内存暂存起来，以备后续计算过程中使用。
- 针对共享内存对整个 block 可见的特性，可将其用于 block 内线程通信。具体地，可以把 block 内某个线程计算的数据存入共享内存，以便 block 内的其他线程使用。

在 Kernel 中要将一个变量定义为共享内存变量时，需要在定义语句中加上限定符 `__shared__`。例如我们需要在共享内存中定义一个长度为 128 的 `float` 数组，可以使用如下语句：
```cuda
__shared__ float s_y[128];
```

### 1.2 共享内存的硬件形式
从硬件层面来说，共享内存是一块 SM 级的存储介质，也就是说整个 SM 上的所有 SubCore 共用同一块共享内存，我们知道在 GPU 进行线程调度时，一个 block 中的线程只会被调度到同一个 SM 上执行，这也是共享内存在 block 内可见并用于 block 内通信的重要原因。随着 GPU 架构不断更新，共享内存的硬件形式也在不断变化，从 Volta 和 Turing 架构开始，L1 缓存、纹理缓存和共享内存三者在物理上被统一成了一个有 128 KB 或 96 KB 的数据缓存，即 Unified Cache，用户可以使用 CUDA Runtime API 来指定其中共享内存所占的容量大小，具体地，通过调用 `cudaFuncSetAttribute()` 函数来设置。

### 1.3 共享内存的 Bank Conflict
共享内存在物理上被划分为连续的 32 个 bank，这些 bank 宽度相同且能被同时访问，从 Maxwell 架构开始，每个 bank 宽度为 32bit，即 4 个字节，bank 带宽为每个时钟周期 32bit。

如果来自一个 warp 的多个线程访问到共享内存中相同 32bit 内的某个字节时，不会产生 bank conflict，读取操作会在一次广播中完成，如果是写入操作，也不会有 bank conflict，且仅会被其中一个线程写入，具体是哪个线程未定义；而如果来自一个 warp 的多个线程访问的是同一个 bank 的不同 32bit 的地址，则会产生 bank conflict。我们可以把共享内存的 32 个 bank 想象为由很多层组成，每个 bank 每层有 32bit，假设同一 warp 内的不同线程访问到某个 bank 的同一层的数据，此时不会发生 bank conflict，但如果同一 warp 内的不同线程访问到某个 bank 的不同层的数据，此时将产生 bank conflict。

下面以一个经典的矩阵转置任务为例，介绍一下如何利用共享内存进行优化加速。

在矩阵转置任务中，核心思想就是每个 block 单独处理一个矩阵块（tile）的转置，假设一个 tile 的大小为 `(32, 32)`，如果直接在全局内存中进行转置处理，就得到下面的代码（仅展示代码片段）：

```cuda
int blockx32 = blockIdx.x * COL32_;
int blocky32 = blockIdx.y * COL32_;
int x = blockx32 + threadIdx.x;
int y = blocky32 + threadIdx.y;

dst[x * m + y] = src[y * n + x]
```

这个代码的问题很显然，在往 `dst` 矩阵中写数据的时候，针对全局内存的写入是非合并访问的，将极大地降低性能。因此，我们可以考虑将一个 tile 的数据先存入共享内存，然后再从共享内存写入全局内存中，保证读取和写入全局内存的环节都是合并访问的，比如下面的代码片段：

```cuda
__shared__ T tile[COL32_][COL32_];

int blockx32 = blockIdx.x * COL32_;
int blocky32 = blockIdx.y * COL32_;
int x = blockx32 + threadIdx.x;
int y = blocky32 + threadIdx.y;

tile[threadIdx.y][threadIdx.x] = src[y * n + x];
__syncthreads();

x = blockx32 + threadIdx.y;
y = blocky32 + threadIdx.x;

dst[x * m + y] = tile[threadIdx.x][threadIdx.y];
```

可以看出，为了使对全局内存的读写变成合并访问，在 Kernel 中对于共享内存的读操作有所变化，相邻线程读取的数据间隔 `COL32_` 个元素，这将会带来一个新的问题。同一个线程束内的线程访问的数据间隔 32 个元素，假设每个元素占 4 个字节，而共享内存分为 32 个 bank，那么正好一个 warp 内的线程访问到了同一个 bank 的 32 层，显然这将造成 32 路 bank conflict，降低共享内存访问性能。怎么解决这个问题呢，很容易，只要让同一个 warp 内的线程访问的数据间隔不是 32 个元素即可，具体地，我们可以将共享内存变量定义稍作修改，比如下面这样：
```cuda
__shared__ T tile[COL32_][COL32_ + 1];
```

通过这种方式在读取共享内存时 warp 内相邻的线程读取的数据间隔 33 个元素，不在同一个 bank，有效避免了 bank conflict，完整代码如下所示：

```cuda
#define COL32_ 32

// transpose matrix
// for (m n) row-major
// grid((m+31)/32, (n+31)/32)
// block(32, 32)
template <typename T>
__global__ void transposeMatrixFromRowMajorKernel(T *dst, const T *src, const int m, const int n)
{
    __shared__ T tile[COL32_][COL32_ + 1];

    int blockx32 = blockIdx.x * COL32_;
    int blocky32 = blockIdx.y * COL32_;
    int x = blockx32 + threadIdx.x;
    int y = blocky32 + threadIdx.y;

    bool check = ((x < n) && (y < m));
    tile[threadIdx.y][threadIdx.x] = check ? __ldg(src + y * n + x) : T(0);

    __syncthreads();

    x = blockx32 + threadIdx.y;
    y = blocky32 + threadIdx.x;

    check = ((x < n) && (y < m));
    if (check)
        dst[x * m + y] = tile[threadIdx.x][threadIdx.y];
}

// for (m, n) row-major matrix
template <typename T>
void transposeMatrixFromRowMajorKernelLauncher(T *dst, const T *src, const int m, const int n, cudaStream_t stream)
{
    transposeMatrixFromRowMajorKernel<T><<<dim3((n + 31) / 32, (m + 31) / 32), dim3(32, 32), 0, stream>>>(dst, src, m, n);
}
```


## 2 分布式共享内存 
前面介绍过，CUDA 在计算能力 9.0 的设备上引入了线程块集群的概念，并为集群中的线程提供了访问集群中所有 block 的共享内存的能力。也就是说在线程块集群中，线程可以跨 block 访问共享内存，我们把这种共享内存称为**分布式共享内存**，对应的地址空间称为**分布式共享内存地址空间**。显而易见，分布式共享内存的可见性是整个线程块集群，集群中的任意线程都可以在分布式共享内存地址空间中读、写或执行原子操作。分布式共享内存是共享内存的一个拓展，容量大小等于线程块集群中 block 数量乘以每个 block 的共享内存大小。

分布式共享内存的引入，提供了一种更大尺度范围的线程数据交换手段，但是与全局内存不同的是，共享内存的生命周期是与 block 一致的，这就使得集群内线程对分布式共享内存中的数据访问必须发生在分布式共享内存的生命周期以内。具体地，在访问分布式共享内存时，要求集群中所有的 block 都已经启动，并且还需要确保对分布式共享内存的所有操作必须发生在集群中的 block 执行完成之前。如果确保这一点呢，可以通过 Cluster Group API 提供的栅栏同步函数 `cluster.sync()` 来同步集群中所有的 block。

基于分布式共享内存可以在线程块集群内跨 block 数据交换的机制，应用程序可以利用这个能力来获取加速效果。具体地，下面以一个简单的直方图计算任务为例，讨论如何使用线程块集群和分布式共享内存进行加速。

通常我们计算直方图任务的标准方法是将数组划分为多个子数组，在每个线程块的共享内存中进行子数组的直方图统计，然后基于子数组的统计结果对全局内存执行原子操作。这种方法会受到共享内存容量的限制，一旦直方图 bins 较多，超出单个 block 共享内存容量限制，此时需要在全局内存直接计算直方图，从而计算全局内存中的原子，效率较低。对于这种场景，可以通过分布式共享内存来进行优化，具体地，将原本存储在单个 block 内共享内存的直方图 bins 划分为多个部分，分别存储在 cluster 内的多个 block 的共享内存，在 block 内访问分布式共享内存进行直方图统计，具体代码如下。

```cuda
#include <cooperative_groups.h>

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input, size_t array_size)
{
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    // Cluster initialization, size and calculating local bin offsets.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        smem[i] = 0; //Initialize shared memory histogram to zeros
    }

    // cluster synchronization ensures that shared memory is initialized to zero in
    // all thread blocks in the cluster. It also ensures that all thread blocks
    // have started executing and they exist concurrently.
    cluster.sync();

    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
    {
        int ldata = input[i];

        //Find the right histogram bin.
        int binid = ldata;
        if (ldata < 0)
            binid = 0;
        else if (ldata >= nbins)
            binid = nbins - 1;

        //Find destination block rank and offset for computing
        //distributed shared memory histogram
        int dst_block_rank = (int)(binid / bins_per_block);
        int dst_offset = binid % bins_per_block;

        //Pointer to target block shared memory
        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

        //Perform atomic update of the histogram bin
        atomicAdd(dst_smem + dst_offset, 1);
    }

    // cluster synchronization is required to ensure all distributed shared
    // memory operations are completed and no thread block exits while
    // other thread blocks are still accessing distributed shared memory
    cluster.sync();

    // Perform global memory histogram, using the local distributed memory histogram
    int *lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        atomicAdd(&lbins[i], smem[i]);
    }
}
```

前面介绍过，使用了线程块集群功能的 Kernel 在启动时需要设置集群的大小（即 cluster\_size），cluster\_size 取决于直方图 bins 的数量。也就是说，如果直方图的规模足够小，单个 block 中的共享内存已经足够存储，那么把 cluster\_size 直接设置为 $1$ 也可以。可以参考下面的代码根据需要设置合适的共享内存数量 cluster\_size，并启动 Kernel。
```cuda
// Launch via extensible launch
{
    cudaLaunchConfig_t config = {0};
    config.gridDim = array_size / threads_per_block;
    config.blockDim = threads_per_block;

    // cluster_size depends on the histogram size.
    // ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
    int cluster_size = 2; // size 2 is an example here
    int nbins_per_block = nbins / cluster_size;

    //dynamic shared memory size is per block.
    //Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
    config.dynamicSmemBytes = nbins_per_block * sizeof(int);

    CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);
}
```

## 3 小结
本文详细介绍了 CUDA 编程模型中共享内存和分布式共享内存的概念，总结如下：
- 共享内存是一块 SM 级的内存，每个 SM 分别具备一块共享内存，从 Volta 架构开始，L1 缓存、纹理缓存和共享内存三者在物理上被统一成了一个数据缓存，其中共享内存容量可以使用 CUDA Runtime API 进行设置。
- 共享内存对整个 block 可见，其生命周期也与 block 一致，也就是说，针对 Kernel 中的共享内存变量，每个 block 拥有一个单独的共享内存变量副本，同一个 block 内的线程访问的是同一个对象，不同 block 的线程访问的是不同的对象。
- CUDA 在计算能力 9.0 的设备上引入了线程块集群的概念，同时也引入了分布式共享内存。分布式共享内存的可见性是整个线程块集群，集群中的任意线程都可以在分布式共享内存地址空间中读、写或执行原子操作。
- 分布式共享内存是共享内存的一个拓展，无需单独定义分布式共享内存变量，其容量大小等于线程块集群中 block 数量乘以每个 block 的共享内存大小。