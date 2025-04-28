#! https://zhuanlan.zhihu.com/p/652255520
# 【CUDA编程】束内规约与块内规约问题

**写在前面**：规约问题在 CUDA 编程中应用非常广泛，笔者最近在研究 Faster Transformer 源码，趁此机会结合 Nivida 官方的代码对规约手段进行总结。

## 1 应用背景
关于规约的定义，相信能读到这篇文章的读者都不陌生，笔者在早期的文章中也介绍过一些规约方法，基本思想都是**折半规约**，主要应用于较大元素规模的向量规约，有兴趣的读者可以移步[【CUDA编程】CUDA编程中的并行规约问题](https://zhuanlan.zhihu.com/p/646998011)。  
本文要介绍的规约场景与之前有所不同，主要应用于矩阵规约，也就是说本文假设的输入变量的维度是 2 维的，形状为 `[batch_size, hidden_units]`，规约之后的输出变量形状为 `[batch_size, ]`。  
接下来，本文将以规约求和为例介绍两种规约方式：**束内规约**、**块内规约**。

## 2 束内规约
束内规约，也就是在一个线程束内对某个变量进行规约。我们知道 CUDA 架构下指令是以线程束（相邻的 32 个线程）为基本单元执行的，线程束内也可以通过束内洗牌指令进行通信，所以这提供了一个很好的束内规约思路。下面是 Nvidia 提供的基础的一个规约设备函数。
```cpp
template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}
```
这个设备函数可以求出当前线程所在线程束的指定变量的规约和，原理涉及洗牌指令的计算逻辑，不再赘述。  
当矩阵宽度 `hidden_units` 较小时，通常可以使用一个 warp 处理一行数据，一个 block 内可以处理多行数据，笔者给出具体的核函数如下：
```cpp
// 一个 warp 处理一行数据
template<typename T>
__global__ void matrix2DWarpReduceSum(const T* inp, T*out, const uint32_t hidden_units) {
    uint32_t tid = threadIdx.x;
    uint32_t lane_id = tid % 32;
    uint32_t warp_id = tid / 32;
    uint32_t warp_num = blockDim.x / 32;
    uint32_t offset = blockIdx.x * warp_num * hidden_units + warp_id * hidden_units;
    T val = 0.0f;
    for (uint32_t i=lane_id; i<hidden_units; i+=32) {
        val += inp[offset + i];
    }
    __syncwarp();
    T warpSum;
    warpSum = warpReduceSum<T>(val);
    if (lane_id == 0) {
      out[blockIdx.x * warp_num + warp_id] = warpSum;
    }
}

template<typename T>
void launchMatrix2DWarpReduceSum(const T* d_x, T* d_y, const uint32_t batch_size, const uint32_t hidden_units) {
  constexpr uint32_t warp_num = BLOCK_SIZE / 32;
  uint32_t gird_size = (batch_size - 1) / (warp_num) + 1;
  matrix2DWarpReduceSum<T><<<gird_size, BLOCK_SIZE>>>(d_x, d_y, hidden_units);
}
```
先确定 `block_size`，这里笔者直接取 `128`，由于是一个 warp 处理一行数据，所以一个 block 可以处理 `warp_num` 行数据，总共需要 `grid_size` 个 block。  
核函数内部首先计算当前线程所在的 warp 编号 `warp_id` 用来定位当前处理元素在哪一行，然后确定线程在 warp 内的编号 `lane_id` 用来定位该线程具体处理那些元素。由于矩阵宽度 `hidden_units` 实际肯定还是比 `32` 大的，所以不可能说一个线程只处理一个元素，因此每个线程会处理多个元素，步长为 `32`，例如当 `hidden_units` 为 `128` 时，`lane_id = 0` 的线程将处理位置为 `0、32、64、96` 的四个元素，`lane_id = 1` 的线程将处理位置为 `1、33、65、97` 的四个元素，以此类推，这个计算过程是没有并行的。循环计算一轮后，对线程束内每个线程的 `val` 进行束内规约就可以得到一行元素的规约和。  

## 3 块内规约
块内规约，就是在一个线程块内求规约值，通常块内规约会通过束内规约来实现，以下是 Nvidia 提供的一个块内规约设备函数。  
```cpp
template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;  

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;
  __syncthreads();
  
  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}
```
规约思路分为两步，首先通过束内规约求出当前线程所在 warp 的规约值，存入 `shared` 中，然后把 `warpSum` 赋值给 `threadIdx.x` 小于 32 的线程内的变量 `val`，这 32 个线程正好也在一个线程束内，然后再执行一次束内规约就得到块内规约值，计算思路非常巧妙。  
另外针对块内规约的问题，官方 cub 库其实提供了 API，开发者可以导入头文件 cub/cub.cuh 后直接使用，注意低版本的 cuda 不支持此 API。我们来看下 API 的调用方式。
```cpp
#include <cub/cub.cuh>

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<template<typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
  if (threadIdx.x == 0) { result_broadcast = result; }
  __syncthreads();
  return result_broadcast;
}
```
除了必要的待规约变量、`block_size` 以外，还需要传入一个计算函数，笔者给出了示例 `SumOp`。  
当矩阵宽度 `hidden_units` 较大时，通常可以使用一个 block 处理一行数据，笔者给出具体的核函数如下：
```cpp
template<typename T>
__global__ void matrix2DBlockReduceSum(const T* inp, T*out, const uint32_t hidden_units) {
  T val = 0.0f;
  uint32_t offset = blockIdx.x * hidden_units;
  for (uint32_t i=threadIdx.x; i<hidden_units; i+=blockDim.x) {
    val += inp[offset + i];
  }
  __syncthreads();
  T blockSum;
  blockSum = blockReduceSum<T>(val);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = blockSum;
  }
}

template<typename T>
void launchMatrix2DBlockReduceSum(const T* d_x, T* d_y, const uint32_t batch_size, const uint32_t hidden_units) {
  uint32_t gird_size = batch_size;
  matrix2DBlockReduceSum<T><<<gird_size, BLOCK_SIZE>>>(d_x, d_y, hidden_units);
}
```
同样，`block_size` 这里笔者直接取 `128`，由于是一个 block 处理一行数据，总共需要 `batch_size` 个 block。  
由于矩阵宽度 `hidden_units` 实际肯定还是比 `block_size` 大的，所以不可能说一个线程只处理一个元素，因此每个线程会处理多个元素，步长为 `block_size`，例如当 `hidden_units` 为 `512` 时，`lane_id = 0` 的线程将处理位置为 `0、128、256、384` 的四个元素，`lane_id = 1` 的线程将处理位置为 `1、129、257、385` 的四个元素，以此类推，这个计算过程是没有并行的。循环计算一轮后，对 block 内每个线程的 `val` 进行块内规约就可以得到一行元素的规约和。  

## 4 向量化数据提升访存带宽
使用向量化操作能够提升内存读写的带宽，而 CUDA 里也提供了一系列数据类型来支持向量化操作，如 float2、float4，就是将 2 个或 4 个 float 数据作为一个整体。为了增加代码的复用性，笔者这里封装了一个 Packed 数据结构，用于对不同的数据类型进行打包。  
```cpp
template <typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed
{
    __device__ Packed()
    {
        // do nothing
    }
    union
    {
        T elem[pack_size]; // 这里联合体只有一个成员，为了方便后期扩展
    };
};
```
结构体内有一个 `elem` 数组变量，整个结构的内存对齐设置为 `sizeof(T) * pack_size`，说白了其实就是把 `pack_size` 个 `T` 类型的数据“捆绑”在一起组成一个新的数据结构，读写内存的时候只需要一次读写就可以读 `pack_size` 个数据，目的是减小内存读写次数。  
那么这个 `pack_size` 能不能无限大呢？显然不能，CUDA 里最大支持 `128` bit 的访问粒度，也就是说对于 float 类型（占 4 个字节，32 bit），一次最多读写 4 个，也就是说 float 的 `pack_size` 最多取到 `4`，本文笔者的示例代码中数据类型都以 float 为例，`pack_size` 取 `4`。  

### 4.1 pack 后的束内规约示例代码
将 `matrix2DWarpReduceSum` 改写为 pack 版的核函数也很简单，计算思路都是一致的，只不过原来一次访问一个元素，现在一次访问一个 pack 的元素，在执行核函数之前笔者加了一个断言，保证 `hidden_units` 能够被 `pack_size` 整除，具体代码如下。
```cpp
template <int pack_size, typename T>
__global__ void matrix2DWarpReduceSumPack(const T* d_x, T* d_y, const uint32_t hidden_units, const uint32_t num_packs) {
  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x & 0x1f;
  const uint32_t warp_num = blockDim.x / 32;
  const uint32_t offset = blockIdx.x * warp_num * hidden_units + warp_id * hidden_units;
  const Packed<T, pack_size>* buf = reinterpret_cast<const Packed<T, pack_size>*>(d_x + offset);
  Packed<T, pack_size> pack;
  T val = 0.0f;
  for (uint32_t pack_id=lane_id; pack_id<num_packs; pack_id+=32) {
    pack = buf[pack_id];
    for (uint32_t i=0; i<pack_size; ++i) {
      val += pack.elem[i];
    }
  }
  __syncwarp();
  T warpSum;
  warpSum = warpReduceSum<T>(val);
  if (lane_id == 0) {
    d_y[blockIdx.x * warp_num + warp_id] = warpSum;
  }
}

template<typename T>
void launchMatrix2DWarpReduceSumPack(const T* d_x, T* d_y, const uint32_t batch_size, const uint32_t hidden_units) {
  constexpr uint32_t warp_num = BLOCK_SIZE / 32;
  uint32_t gird_size = (batch_size - 1) / (warp_num) + 1;
  constexpr uint32_t pack_size = 4;
  // 一行元素的 pack 数量
  uint32_t num_packs = hidden_units / pack_size;
  assert(hidden_units % pack_size == 0);
  matrix2DWarpReduceSumPack<pack_size, T><<<gird_size, BLOCK_SIZE>>>(d_x, d_y, hidden_units, num_packs);
}
```
核函数内部就一句核心代码，将 `const T*` 指针转换成 `const Packed<T, pack_size>*`。
```cpp
const Packed<T, pack_size>* buf = reinterpret_cast<const Packed<T, pack_size>*>(d_x + offset);
```
然后用 `pack_id` 索引一次取一个 pack 的数据，注意这里对 pack 索引的时候不要写错了。跟前面一样，相邻的线程处理相邻的 pack 数据，这是为了全局内存的合并访问。加法计算次数还是那么多次，因为 `Packed` 结构体并不能直接参与计算，还是要用 `elem` 里面的元素计算，这个核函数也就节省了访存次数而已。  

### 4.2 pack 后的块内规约示例代码
`matrix2DBlockReduceSumPack` 核函数的实现就更简单了，直接上代码。  
```cpp
template <int pack_size, typename T>
__global__ void matrix2DBlockReduceSumPack(const T* d_x, T* d_y, const uint32_t hidden_units, const uint32_t num_packs) {
  T val = 0.0f;
  uint32_t offset = blockIdx.x * hidden_units;
  const Packed<T, pack_size>* buf = reinterpret_cast<const Packed<T, pack_size>*>(d_x + offset);
  Packed<T, pack_size> pack;
  for (uint32_t pack_id=threadIdx.x; pack_id<num_packs; pack_id+=blockDim.x) {
    pack = buf[pack_id];
    for (uint32_t i=0; i<pack_size; ++i) {
      val += pack.elem[i];
    }
  }
  __syncthreads();
  T blockSum;
  blockSum = blockReduceSum<T>(val);
  if (threadIdx.x == 0) {
    d_y[blockIdx.x] = blockSum;
  }
}

template<typename T>
void launchMatrix2DBlockReduceSumPack(const T* d_x, T* d_y, const uint32_t batch_size, const uint32_t hidden_units) {
  uint32_t gird_size = batch_size;
  constexpr uint32_t pack_size = 4;
  assert(hidden_units % pack_size == 0);
  uint32_t num_packs = hidden_units / pack_size;
  matrix2DBlockReduceSumPack<pack_size, T><<<gird_size, BLOCK_SIZE>>>(d_x, d_y, hidden_units, num_packs);
}
```

## 5 小结
在深度学习算子的开发过程中，规约是一个非常常见的场景，以 Softmax 为例就有 reduceMax 和 reduceSum 的应用，本文给出了两种规约实现方式，可供读者参考使用。实际开发过程中，规约计算一般是隐藏在其他 kernel 中的，并不会奢侈到单独写个规约 kernel，所以要求开发人员领会思路活学活用。