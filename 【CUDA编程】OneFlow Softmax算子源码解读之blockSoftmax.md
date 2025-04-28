#! https://zhuanlan.zhihu.com/p/646998408
# 【CUDA编程】OneFlow Softmax算子源码解读之BlockSoftmax
**写在前面**：笔者这段时间工作太忙，身心俱疲，博客停更了一段时间，现在重新捡起来。本文主要解读 OneFlow 框架的第二种 Softmax 源码实现细节，即 block 级别的 Softmax。

## 1 整体逻辑
我们知道，对于形状为 `(num_rows, num_cols)` 的矩阵来说，其 Softmax 计算结果只与当前元素所在行的元素相关，所以实现 cuda kernel 的关键就是采用多大维度的线程组来处理一行元素。BlockSoftmax 的核心思想是使用一个 block 处理一行元素的计算，借助共享内存保存中间结果数据以及进行线程间的通信。有兴趣的读者可以去如下地址阅读源码：https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh   
线程网络结构如下：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qaW7j1zmgQYP6lvOaYmFMUFTibVbuKgfltGWjJ4mribxiaNmEn4tRVdGBv6AfTlOLqsmGa03L8Tv9hg/640?wx_fmt=png)

## 2 源码解析
### 2.1 数据 Pack 提升访问带宽
数据 pack 方面笔者在上一篇文章已经详细介绍，主要目的是提升内存访问带宽，这里不再赘述。
### 2.2 调用链
针对 BlockSMemSoftmax 这个分支，笔者对源码中函数的调用关系梳理后整理如下：
```cuda
TryDispatchSoftmaxBlockSMemImpl
  ->TryDispatchSoftmaxBlockSMemImplPackSize
    ->TryDispatchSoftmaxBlockSMemImplBlockSize
      ->LaunchSoftmaxBlockSMemImpl
        ->SoftmaxBlockSMemImpl(kernel)
```
接下来笔者将从上到下逐个解读其实现细节。

### 2.3 TryDispatchSoftmaxBlockSMemImpl
该函数被 `DispatchSoftmax` 函数调用，其内部逻辑非常简单，实例化了一个 `TryDispatchSoftmaxBlockSMemImplPackSize` 类并调用了其重载的`()`运算符函数，所有的参数都是透传，没有其他逻辑。
```cuda
template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                   const int64_t rows, const int64_t cols,
                                                   bool* success) {
  return TryDispatchSoftmaxBlockSMemImplPackSize<LOAD, STORE, ComputeType, algorithm>()(
      stream, load, store, rows, cols, success);
}
```

### 2.4 TryDispatchSoftmaxBlockSMemImplPackSize
顾名思义，`pack_size` 参数是在这个结构体内部确定的。该结构体内部重载了一个小括号运算符，其函数内部只做了一件事，对矩阵的列数进行判断从而确定 `pack_size`，如果矩阵列数是偶数，`pack_size` 取 2，否则取 1。
```cuda
template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct TryDispatchSoftmaxBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, bool* success) {
    if (cols % 2 == 0) {
      return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2, algorithm>(
          stream, load, store, rows, cols, success);
    } else {
      return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1, algorithm>(
          stream, load, store, rows, cols, success);
    }
  }
};
```

### 2.5 TryDispatchSoftmaxBlockSMemImplBlockSize
顾名思义，`block_size` 参数是在该函数内部确定的。关于 `block_size` 参数的确定方法笔者在另一篇文章（[【CUDA编程】OneFlow Element-Wise 算子源码解读](https://zhuanlan.zhihu.com/p/646990764)）中有详细介绍，因为 blockSMemSoftmax 方案主要使用共享内存而不是寄存器，所以这里我们取消寄存器限制，可以得到 `block_size` 参数在理想情况下应在 128 和 1024 之间。因此函数 `TryDispatchSoftmaxBlockSMemImplBlockSize` 中分别定义了 4 个变量，对应 4 种情况。
```cuda
// 设置4个不同的block_size
constexpr int block_size_conf_1 = 128;
constexpr int block_size_conf_2 = 256;
constexpr int block_size_conf_3 = 512;
constexpr int block_size_conf_4 = 1024;
```

#### 2.5.1 SM 占有率限制
我们知道，block 是运行在 SM 上的，笔者在前面的文章说过，单个 SM 所能承载的最大 block 数和最大 thread 数是有上限的，为了保证单个 SM 被 thread 填满（即 SM 占用率 100%），我们要求 `block_size` 最小取 128。

#### 2.5.2 线程块同步机制限制
我们知道单个 SM 上的线程总量等于单个 SM 上的 block 数量乘以 `block_size`，所以在确定 `block_size` 前我们不妨思考一个问题：单个 SM 上 block 的数量越多越好还是越少越好？  
当一个 block 内的线程共同完成一项计算任务时，通常 block 内线程要做同步防止出现读写竞争问题。极端情况下，我们假设单个 SM 上只有一个 block，当 SM 中正在调度执行的一个 block 到达同步点时，SM 内可执行 warp 将逐渐减少至 0，会导致计算资源空闲，相当于整个 SM 在等待剩余的 warp 逐步执行，造成资源浪费。若此时 SM 上同时有其他 block 在执行，则在一个 block 到达同步点时仍然有其他 block 可以执行。所以从这个层面上来说，单个 SM 可同时调度的 block 越多越好，单个 SM 上 block 数量越多的同时，`block_size` 越小。  

#### 2.5.3 cudaOccupancyMaxActiveBlocksPerMultiprocessor 函数
前面说过单个 SM 上同时调度的 block 数量越多越好，那么我们如何取到这个最大的 block 数量？首先直接取官方给出的单个 SM 可承载的 block 数上限值肯定是不行的，这只是一个理论上限，实际未必能保证这些 block 被同时调度，同时调度的影响因素还有：共享内存、CUDA 函数、`block_size` 等等。这里 Nvidia 官方提供了一个预估函数 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`，该函数会根据设备的硬件资源限制，例如 SM 上的计算核心数量、共享内存的大小等，来确定可调度块的数量。随后，它会计算出特定内核函数中每个线程块的负载平衡，并选出在给定硬件约束下可调度线程块数量的最大值。最后，该函数将返回一个值，表示在当前硬件资源限制下每个 SM 可允许的最大线程块数量，这个值是一个整数。该函数的返回结果可以用来估算程序并行效率，帮助开发人员优化程序以使在 GPU 上运行更有效率。

#### 2.5.4 代码逻辑
终上所述，如何选取最合适的 `block_size` 参数呢？**首先 SM 能同时调度的 block 数量越大越好；当 SM 能同时调度的 Block 数不变的情况下，block_size 越大越好，越大就有越高的并行度。因此代码中在选择 block_size 时，对不同 block_size 都计算了 cudaOccupancyMaxActiveBlocksPerMultiprocessor，若结果相同，使用较大的 block_size**。  
简单来说，优先让 SM 同时调度的 block 数量达到最大，其次让 `block_size` 达到最大。
```cuda
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImplBlockSize(cudaStream_t stream, LOAD load,
                                                            STORE store, const int64_t rows,
                                                            const int64_t cols, bool* success) {
  // 设置4个不同的block_size
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  // 计算blockSoftmax方案需要的共享内存大小
  const size_t smem = cols * sizeof(ComputeType);
  int max_active_blocks_conf_1;
  {
    // 占用计算器API cudaOccupancyMaxActiveBlocksPerMultiprocessor可以根据 kernel 的 block 大小和共享内存使用情况提供占用率预测。
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1, algorithm>,
        block_size_conf_1, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_1 <= 0) {
    *success = false;
    return cudaSuccess;
  }
  int max_active_blocks_conf_4;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_4,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4, algorithm>,
        block_size_conf_4, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4,
                                      algorithm>(stream, load, store, smem, rows, cols);
  }
  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3, algorithm>,
        block_size_conf_3, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3,
                                      algorithm>(stream, load, store, smem, rows, cols);
  }
  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2, algorithm>,
        block_size_conf_2, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2,
                                      algorithm>(stream, load, store, smem, rows, cols);
  }
  *success = true;
  return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1,
                                    algorithm>(stream, load, store, smem, rows, cols);
}

```
源码中首先计算了 `block_size = 128` 时的 SM 同时调度的 block 数量 `max_active_blocks_conf_1`，并以此作为 SM 同时调度的最大 block 数量，然后分别计算其他三种 `block_size` 的 `max_active_blocks_conf` 如果等于最大的 block 数量，则取较大的 `block_size`。

### 2.6 核函数 SoftmaxBlockSMemImpl
接下来就是 BlockSoftmax 的核函数 `SoftmaxBlockSMemImpl`，先来看一下代码，接下来笔者将逐步解读源码作者的实现意图。

```cuda
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
         Algorithm algorithm>
__global__ void SoftmaxBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                     const int64_t cols) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  // 一个 Block 处理一行元素
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    // 当前线程的最大值初始化为 -inf
    ComputeType thread_max = -Inf<ComputeType>();
    // 以向量化的方式加载一行数据，然后执行pack reduce操作
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        thread_max = max(thread_max, pack[i]);
      }
    }
    // 执行block reduce获取当前行（由一个 Block 进行处理）的最大值
    const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    ComputeType thread_sum = 0;
    for (int col = tid; col < cols; col += block_size) {
      if (algorithm == Algorithm::kSoftmax) {
        const ComputeType exp_x = Exp(buf[col] - row_max);
        buf[col] = exp_x;
        thread_sum += exp_x;
      } else {
        const ComputeType x = buf[col] - row_max;
        buf[col] = x;
        thread_sum += Exp(x);
      }
    }
    // 同理，获得当前行的sum
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    // 计算结果并写回到全局内存中
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          pack[i] = Div(buf[i * num_packs + pack_id], row_sum);
        } else if (algorithm == Algorithm::kLogSoftmax) {
          pack[i] = buf[i * num_packs + pack_id] - Log(row_sum);
        } else {
          __trap();
        }
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}
```
#### 2.6.1 定义共享内存变量
核函数内部首先定义了一个共享内存数组变量 `shared_buf`，内存对齐大小为 `sizeof(double)`，随后在核函数内部做了一个断言，校验 `pack_size` 是否能够整除 `cols`。在 `shared_buf` 变量定义语句中，`extern` 是使用动态共享内存的一种声明方式，表示内存大小将在调用核函数时通过 `<<<>>>` 的第三个参数指定，这里为矩阵的一行元素对应的内存大小，即 `cols * sizeof(ComputeType)`。然后使用 `reinterpret_cast<ComputeType*>` 将 `shared_buf` 的首地址转换为 `ComputeType*` 类型的指针并赋给指针 `buf`。这段代码的意义是将共享内存 `shared_buf` 的首地址强制转换为`ComputeType` 类型的指针，使得在后续代码中可以通过 `buf` 来访问共享内存，进行后续的 GPU 并行计算操作。这里笔者没想明白为什么要绕个圈子，而不是直接用如下代码直接定义。
```cuda
extern __shared__ __align__(sizeof(double)) ComputeType buf[];
```
咨询源码作者后收到反馈，“如果采用上述方式定义会编译报错”，然后笔者亲测没有报错，可能是两边开发环境不同导致，有兴趣地读者可以尝试下，笔者的环境为 win10+cuda11.7+rtx2070super。  

#### 2.6.2 计算每个线程的 thread_max 
主体部分是一个 Grip-loop 的循环，循环步长设置为网格大小。Grip-loop 内部定义了一个寄存器变量 `thread_max`，这个变量用来存储当前线程处理的元素中的最大值。  
接下来是两层循环求出 `thread_max`，这里和 WarpSoftmax 一样，一个线程处理的 pack 是不相连的，这里主要是因为 GPU 在从内存取数的时候，为了优化内存访问性能，一次取数时会将临近空间的数据也取出存入缓存，而 GPU 指令是按照线程束为基本单位进行执行的，这样的话，一次取数可以满足相邻的多个线程的使用需求，直接去缓存取即可，无需再次访问内存。因此为了最大程度提高访问效率，相邻线程访问的数据是紧邻的。在循环体内部定义了一个寄存器数组变量 `pack[pack_size]`，将内存中一个 pack 的数据加载到数组 `pack` 中，然后在 `pack` 内求 `thread_max`，顺便将数据也加载到共享内存变量 `buf` 中。 

#### 2.6.3 Bank Conflicts 优化
从往 `buf` 里加载数据的代码中可以发现，在 `buf` 中内存排列有些**反常**，一个 pack 内的元素不是相邻排列的，而是隔了 `num_packs` 个元素，这里可以把 `buf` 想象成一个 `pack_size` 行 `num_packs` 列的矩阵，一个 pack 内的元素是**按列存储**的。为什么要按列往共享内存里写入数据？  
这里涉及一个**Bank Conflicts** 概念，这也是在使用共享内存时需要重点关注的问题。为了获得更高的内存带宽，共享内存在物理上被分为了 **32 个宽度相等（开普勒架构为8个字节，其他架构4个字节）的 bank**，这些 bank 可以被**同时访问**。为什么恰好是 32 个？因为前面说过 GPU 中执行指令是以线程束（32个线程）为基本单位的，这样可以保证**每个线程同时访问一个 bank 的数据**，带宽达到最大。那么 Bank Conflicts 的定义来了，**在一个 warp 内，有 2 个或以上的线程访问了同一个 bank 上的不同地址的内存**。  
现在我们假设 `buf` 里使用如下方式加载数据，即一个 pack 里的元素相邻排列，则代码如下：
```cuda
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[pack_id * pack_size + i] = pack[i];
        thread_max = max(thread_max, pack[i]);
      }
```
当 `pack_size = 1` 时，每个线程连续写 4 个字节时，每个 warp 刚好完整访问 shared memory 的一行，这个时候并不会出现 bank conflict。而当 `pack_size = 2` 时，每个线程写连续 2 个 4 字节时（可以看成8个字节），此时 0 号线程访问的地址在第 0 和第 1 个 bank，1 号线程访问的地址在第 2 和第 3 个 bank，以此类推，16 号线程访问地址又在第 0 和第 1 个 bank 内，此时 16 号线程和 0 号线程访问了同一个 bank 的不同地址，此时即产生了 Bank Conflicts。见图a。  
为了避免 Bank Conflicts，可以将一个 pack 的数据按列存储，这样当矩阵元素的字节大于 4 或 `pack_size >= 2` 时，可以保证每个线程访问的数据都在自己的 bank 内，如图b。
![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5pjQMTLAfjDaOXRpvfxKXdwuZc54A9PQ3CNOgZa9MvwVCDX5CPQvtfFYianI5G2NxxwYBlaaRPMgRg/640?wx_fmt=png)

#### 2.6.4 block reduce 获取 row_max
得到每一个线程处理的元素的最大值后，还需要计算每一行的最大值。在 WarpSoftmax 中，由于每一行是由一个 warp 处理的，所以我们使用束内洗牌指令即可得到矩阵单行最大值，而在 BlockSoftmax 中，每一行是由一个 block 处理的，这时候不能再使用束内指令来规约了，为了保证较高的性能，应使用共享内存进行线程间的数据通信。这里源码封装了一个 `BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max)` 函数，在函数内直接使用 Nvidia 官方 cub 库进行计算，官方文档见：https://nvlabs.github.io/cub/classcub_1_1_block_reduce.html#a089953b3bdfe7c48208632d0cc2ac1fb  

```cuda
const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
```
函数体中声明了两个共享内存变量 `temp_storage` 和 `result_broadcast`，可见该库函数底层也是通过共享内存实现的，这里如果不用官方库，也可以自定义规约函数，有兴趣的读者可以参考笔者的另一篇文章[【CUDA编程】CUDA编程中的并行规约问题](https://zhuanlan.zhihu.com/p/646998011)  

#### 2.6.5 thread_sum 和 row_sum
定义一个寄存器变量 `thread_sum` 存储当前线程处理的元素的指数和。由于矩阵元素已经加载进共享内存 `buf` 中，所以这一次遍历求和不需要再访问全局内存，直接在 `buf` 中以 `block_size` 为步长求和即可，遍历的同时也将 `buf` 的内存替换成 `exp_x` 这是为了方便后续的计算。这里为什么要以 `block_size` 为步长？也是因为前面为了避免 bank conflicts 将 `buf` 的内存排列设定为相邻线程的元素相邻存储，所以我们只要以 `block_size` 为步长遍历即可完成该线程处理的所有 pack 元素的遍历。  
获取到 `thread_sum` 后，同样使用前面 `BlockAllReduce` 函数进行块内规约计算出 `row_sum`。

#### 2.6.6 计算 Softmax
首先对当前线程以 `block_size` 为步长做一次循环，然后在 pack 内利用上一步计算的 `row_max` 和 `buf` 计算出 Softmax 值，最后将 pack 内的计算结果写入全局内存中。

## 3 小结
总结一下 BlockSoftmax 源码中的一些值得学习的内容：
- 在选取 `block_size` 时，要结合 kernel 的计算逻辑，灵活选取。比如作者在实现 BlockSoftmax 时考虑到这种实现方式不会大量使用寄存器，所以去除了寄存器限制；考虑到块内同步和共享内存，所以使用了官方库函数预估 SM 占用率。这些都是很值得读者学习的，兵无常势，水无常形。
- BlockSoftmax 的核心是块内规约，利用共享内存读写速度远大于全局内存的特性，提高 kernel 性能。
- 在使用共享内存时，一定要避免 Bank Conflicts，可以提升至少 20% 的访存效率。