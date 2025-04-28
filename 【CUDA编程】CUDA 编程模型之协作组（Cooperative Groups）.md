#! https://zhuanlan.zhihu.com/p/673304744
# CUDA 编程模型之协作组（Cooperative Groups）

**写在前面**：本文是笔者手稿的第 8 章的内容，文中有不少新概念笔者之前都没有接触和使用过，有些受限于设备计算能力也无从验证，于是在阅读了一些博客和文档后，基于笔者自身的理解进行了阐述，如有错漏之处，请读者们务必指出，感谢！

## 1 简介 

**协作组**（Cooperative Groups）是 CUDA 9 中引入的 CUDA 编程模型的扩展，用于组织通信线程组。协作组允许开发人员自定义线程通信的粒度，从而实现更丰富、更有效的并行结构。

在CUDA 编程模型中提供了一种单一、简单的方式用来同步合作线程：使用 `__syncthreads()` 函数进行的线程块内的栅栏同步。然而，应用程序开发过程中，开发人员还希望以其他粒度定义和同步线程组，以实现更高的性能、设计灵活性和软件复用性（线程组级别相关的函数接口）。为了表达更宽泛的并行交互模式，很多追求高性能的程序员采用他们自己编写的临时和不安全的代码来同步单个 warp 内的线程或单一 GPU 上的不同线程块。尽管可以取得不错的性能提升，但这也导致了零碎代码的泛滥，这种零碎代码随着时间和 GPU 的更新换代变得难以维护、升级和拓展。基于以上背景，协作组通过提供安全且面向未来的机制，支持高性能易拓展的代码实现。

## 2 协作组的新功能 
### 2.1 CUDA 12.2

为 `gridgroup` 和 `threadblock` 添加了 `barrierArrive` 和 `barrierwait` 成员函数。具体 API 说明参见 [barrier_arrive 和 barrier_wait](#611-barrier_arrive-和-barrier_wait) 小节。

### 2.2 CUDA 12.1

新增了 `invoke_one` 和 `invoke_one_broadcast` API，具体见 [invoke_one 和 invoke_one_broadcast](#641-invoke_one-和-invoke_one_broadcast) 小节。

### 2.3 CUDA 12.0

- 以下实验性 API 现在已经被移到主命名空间：
  - 在 CUDA 11.7 中新增了异步规约和扫描的更新。
  - 在 CUDA 11.1 中添新增了大于 32 的 `thread_block_tile`。
  
- 不再需要使用 `block_tile_memory` 对象来提供内存，以便在 Compute Capability 8.0 或更高版本上创建这些大的线程块分块（block tiles）。


## 3 编程模型概念 

协作组编程模型描述了 CUDA 线程块内和跨线程块的同步模式。为用户提供了自定义线程组的方法，以及线程组内同步的接口。另外还提供了强制执行某些限制的新的启动 API，从而保证组内同步可以正常工作。这些原语实现了 CUDA 内协作并行的新模式，包括生产者-消费者并行、机会并行和整个网格的全局同步。

协作组编程模型由以下几部分组成：
- 表示协作线程组的数据类型；
- 获取由 CUDA 启动 API 定义的隐式线程组（例如，线程块）；
- 将现有组划分为新组的集体操作；
- 用于数据移动和操作的集体算法（例如异步拷贝、规约、扫描）；
- 同步组内所有线程的操作；
- 检查组属性的操作；
- 提供低级别、特定于组且通常是硬件加速的集体操作。

协作组中提出了一种由许多个线程组成的组（group）的概念，将组表示为一级程序对象，相关函数可以收到表示参与线程组的明确对象，这种方式也让程序员意图明确，消除了导致零碎代码、不合理的编译器优化限制的不健全架构假设，并且更好地适配新的 GPU 版本。

要编写高效的代码，最好使用专门的组（通用会失去很多编译时优化），并通过引用将这些组对象传递给打算以某种协作方式使用这些线程的函数。

协作线程组需要 CUDA 9.0 或更高版本。要使用协作线程组，请包含头文件：

```cuda
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <cooperative_groups/reduce.h>
// Optionally include for inclusive_scan() and exclusive_scan() collectives
#include <cooperative_groups/scan.h>
```

并使用 Cooperative Groups 命名空间：

```cuda
using namespace cooperative_groups;
// Alternatively use an alias to avoid polluting the namespace with collective algorithms
namespace cg = cooperative_groups;
```

代码可以使用 nvcc 以正常方式编译，但是如果希望使用 memcpy_async、reduce 或 scan 等功能并且 host 编译器的默认语言不是 C++11 或更高版本，那么编译时必须添加 `--std=c++11` 到命令行。

### 3.1 示例 

为了阐述协作线程组的概念，以下示例尝试执行 block 范围的并行规约求和。在协作组引入之前，写这段代码的时候在实现上有隐藏的约束：

```cuda
__device__ int sum(int *x, int n) {
    // ...
    __syncthreads();
    return total;
}

__global__ void parallel_kernel(float *x) {
    // ...
    // Entire thread block must call sum
    sum(x, n);
}
```

block 中的所有线程都必须到达 `__syncthreads()` 屏障，然而，这个约束对可能想要使用 `sum(…)` 的开发人员是隐藏的。对于协作组，更好的写法是：

```cuda
__device__ int sum(const thread_block& g, int *x, int n) {
    // ...
    g.sync()
    return total;
}

__global__ void parallel_kernel(...) {
    // ...
    // Entire thread block must call sum
    thread_block tb = this_thread_block();
    sum(tb, x, n);
    // ...
}
```

## 4 组类型 
### 4.1 隐式组 

**隐式组**（Implicit Groups）代表 Kernel 的启动配置。无论 Kernel 代码如何编写，始终具有一定数量的线程（thread）、线程块（block）和线程块维度、单个线程网格（grid）和线程网格维度。另外，如果使用多设备协同启动 API，还可以有多个 grid（每个设备一个 grid）。这些组为分解成更细粒度的组提供了一个起点，这些细粒度组通常是硬件加速的并且专门用于特定问题。

尽管开发人员可以在代码中的任何位置创建隐式线程组，但这样做有很大隐患。为隐式线程组创建句柄是一个组内的集体操作，即组中的所有线程都会参与。如果隐式线程组是在并非所有线程都到达的条件分支中创建的，则可能会导致死锁或数据损坏。出于这个原因，建议预先为隐式线程组创建一个句柄（尽可能早，在任何分支发生之前）并在整个内核中使用该句柄。出于同样的原因，线程组句柄必须在声明时初始化（没有默认构造函数），并且不建议复制构造线程组句柄。

#### 4.1.1 线程块级别的协作组 

对 CUDA 程序员来说，线程块（block）是一个非常熟悉的概念。协作组扩展引入了一种新的线程块（block）数据类型，`thread_block`，以在 Kernel 中明确表示这个概念。

```cuda
class thread_block;
```

通过以下方式创建：

```cuda
thread_block g = this_thread_block();
```

公有成员函数：

- `static void sync()`：同步组内线程，相当于 `g.barrier_wait(g.barrier_arrive())`。
- `thread_block::arrival_token barrier_arrive()`：到达 `thread_block` 的屏障，函数返回一个 token，在调用 `barrier_wait()` 时传入。
- `void barrier_wait(thread_block::arrival_token&& t)`：在 `thread_block` 屏障上等待，将 `barrier_arrive()` 返回的 token 作为右值引用。
- `static unsigned int thread_rank()`：线程在组内的标号，区间为 `[0, num_threads)`。
- `static dim3 group_index()`：当前 block 在 grid 中的三维索引，相当于 `blockIdx`。
- `static dim3 thread_index()`：当前 thread 在 block 中的三维索引，相当于 `threadIdx`。
- `static dim3 dim_threads()`：当前启动的 block 的维度，以线程为单位。
- `static unsigned int num_threads()`：当前组中总的线程数量。


旧版成员函数（别名）：

- `static unsigned int size()`：当前组中总的线程数量（等价于 `num_threads()`）。
- `static dim3 group_dim()`：当前启动的 block 的维度（等价于 `dim_threads()`），以线程为单位。


示例代码如下：
```cuda
/// Loading an integer from global into shared memory
__global__ void kernel(int *globalInput) {
    __shared__ int x;
    thread_block g = this_thread_block();
    // Choose a leader in the thread block
    if (g.thread_rank() == 0) {
        // load from global into shared for all threads to work with
        x = (*globalInput);
    }
    // After loading data into shared memory, you want to synchronize
    // if all threads in your thread block need to see it
    g.sync(); // equivalent to __syncthreads();
}
```

组中的所有线程必须参与集体操作，否则行为未定义。

`thread_block` 数据类型派生自更通用的 `thread_group` 数据类型，可用于表示更广泛的组类别。

#### 4.1.2 集群级别的协作组 

该组对象表示在单个集群（cluster）中启动的所有线程，参见线程块集群的定义。这些 API 适用于计算能力 9.0 及以上的所有硬件。在这种情况下，当启动非集群网格时，API 自动假定一个 $1\times1\times1$ 集群。

```cuda
class cluster_group;
```

通过以下方式创建：

```cuda
cluster_group g = this_cluster();
```
  
公有成员函数：

- `static void sync()`：同步组内线程。
- `static unsigned int thread_rank()`：线程在组内的标号，区间为 `[0, num_threads)`。
- `static unsigned int block_rank()`：线程块（block）在组内的标号，区间为 `[0, num_blocks)`。
- `static unsigned int num_threads()`：当前组中总的线程数量。
- `static unsigned int num_blocks()`：当前组中总的线程块（block）数量。
- `static dim3 dim_threads()`：当前启动的 cluster 的维度，以线程为单位。
- `static dim3 dim_blocks()`：当前启动的 cluster 的维度，以线程块（block）为单位。
- `static dim3 block_index()`：当前 block 在启动的 cluster 中的三维索引。
- `static unsigned int query_shared_rank(const void *addr)`：获取共享内存地址所属的线程块的标号。
- `static T* map_shared_rank(T *addr, int rank)`：获取 cluster 中另一个 block 的共享内存变量的地址。


旧版成员函数（别名）：

- `static unsigned int size()`：当前组中总的线程数量（等价于 `num_threads()`）。


`query_shared_rank` 和 `map_shared_rank` 两个函数涉及分布式共享内存的概念，具体可参阅 \ref{sec:runtime-Distributed-Shared-Memory}分布式共享内存。

#### 4.1.3 线程网格级别的协作组 

该组对象表示在单个网格中启动的所有线程。除了 `sync()` 之外的 API 随时可用，但如果要实现跨网格同步，需要使用协作启动 API（cooperative launch API）。

```cuda
class grid_group;
```

通过以下方式创建：

```cuda
  grid_group g = this_grid();
```

公有成员函数：

- `bool is_valid() const`：返回 bool 值，表示该 `grid_group` 是否可以同步。
- `void sync() const`：同步组内线程，相当于 `g.barrier_wait(g.barrier_arrive())`。
- `grid_group::arrival_token barrier_arrive()`：到达 `grid` 的屏障，函数返回一个 token，在调用 `barrier_wait()` 时传入。
- `void barrier_wait(grid_group::arrival_token&& t)`：在 `grid` 屏障上等待，将 `barrier_arrive()` 返回的 token 作为右值引用。
- `static unsigned long long thread_rank()`：线程在组内的标号，区间为 `[0, num_threads)`。
- `static unsigned long long block_rank()`：线程块（block）在组内的标号，区间为 `[0, num_blocks)`。
- `static unsigned long long cluster_rank()`：集群（cluster）在组内的标号，区间为 `[0, num_clusters)`。
- `static unsigned long long num_threads()`：当前组中总的线程数量。
- `static unsigned long long num_blocks()`：当前组中总的线程块（block）数量。
- `static unsigned long long num_clusters()`：当前组中总的集群（cluster）数量。
- `static dim3 dim_blocks()`：当前启动的 grid 的维度，以线程块（block）为单位。
- `static dim3 dim_clusters()`：当前启动的 grid 的维度，以集群（cluster）为单位。
- `static dim3 block_index()`：当前 block 在启动的 grid 中的三维索引。
- `static dim3 cluster_index()`：当前 cluster  在启动的 grid 中的三维索引。


旧版成员函数（别名）：

- `static unsigned long long size()`：当前组中总的线程数量（等价于 `num_threads()`）。
- `static dim3 group_dim()`：当前启动的 grid 的维度（等价于 `dim_blocks()`），以线程块（block）为单位。


#### 4.1.4 多个线程网格级别的协作组 

该组对象表示在多设备协作启动的所有线程。与 `grid.group` 不同，所有 API 都要求您使用了适当的启动 API。

```cuda
class multi_grid_group;
```

通过以下方式创建：

```cuda
// Kernel must be launched with the cooperative multi-device API
multi_grid_group g = this_multi_grid();
```

公有成员函数：

- `bool is_valid() const`：返回 bool 值，表示该 `multi_grid_group` 是否可以使用。
- `void sync() const`：同步组内线程。
- `unsigned long long num_threads() const`：当前组中总的线程数量。
- `unsigned long long thread_rank() const`：线程在组内的标号，区间为 `[0, num_threads)`。
- `unsigned int grid_rank() const`：线程网格 grid 在组内的标号，区间为 `[0, num_grids)`。
- `unsigned int num_grids() const`：启动的总的线程网格 grid 的数量。


旧版成员函数（别名）：

- `unsigned long long size()`：当前组中总的线程数量（等价于 `num_threads()`）。


弃用通知：多网格组已在 CUDA 11.3 中针对所有设备弃用。

### 4.2 显式组 
与隐式组（Implicit groups）相对应的，还有一个显式组（Explicit groups）的概念。隐式线程组是无法由用户自定义分组大小的，因为在 kernel 启动配置中就确定了。比如将每个 block 中的所有线程分到一个组；将部分 block 组成的 cluster 中的所有线程分到一个组；或者将一个 grid 中的所有线程分到一个组；甚至将多个 grid 中的所有线程分到一个组。这些组本身就是``隐式''存在的，只是协作组通过隐式组的概念把它们具象化了，使得开发人员可以通过句柄对其进行明确访问。而显示组可以理解为用户可以自定义分组大小，比如开发人员想将一个 block 中的部分线程分到一个组，进而就有了线程块分片（Thread Block Tile，姑且称之为**分片组**）的概念，再比如想将一个 warp 中的部分线程分到一个组，进而就有了**合并组**（Coalesced Groups）的概念，下面将进行详细介绍。

#### 4.2.1 线程块分片（Thread Block Tile） 

官方提供了一个分片组（tiled group）的类模板 `thread_block_tile`，其中模板参数用于指定片（tile）的大小。如果在编译时知道片的大小，执行时可能会更高效。当然，如果片的大小在编译期无法确定，那么也可以在运行时当做 `tiled_partition` 的参数传入，也是可行的。

```cuda
template <unsigned int Size, typename ParentT = void>
class thread_block_tile;
```

通过以下方式创建：

```cuda
template <unsigned int Size, typename ParentT>
_CG_QUALIFIER thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)
```

其中，`Size` 必须是 $2$ 的非负整数幂且小于等于 $1024$。下面的备注部分介绍了在具有 7.5 或更低计算能力的硬件上创建大小大于 $32$ 的分片所需的额外步骤。

`ParentT` 是划分组的基准类型，也就是父组的类型，是自动推断的，会将此信息存储在组对象的句柄中。

公有成员函数：

- `void sync() const`：同步组内线程。
- `unsigned long long num_threads() const`：当前组中总的线程数量。
- `unsigned long long thread_rank() const`：线程在组内的标号，区间为 `[0, num_threads)`。
- `unsigned long long meta_group_size() const`：返回对父组进行划分后，单个父组创建的组（分片）数。
- `unsigned long long meta_group_rank() const`：分片（tile）在父组中的标号，区间为 `[0, meta_group_size)`。
- `T shfl(T var, unsigned int src_rank) const`：参阅[束内洗牌函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484780&idx=1&sn=0697baa88af4edd2169ce2ba73c4076e&chksm=e92780d5de5009c328a0664ffae141abbbc45b5ad24ac1b97333cd307578cc40cb5ba2896b2e&token=1222151528&lang=zh_CN#rd)，注意：对于大于 $32$ 的分片大小，组中的所有线程都必须指定相同的 `src_rank`，否则行为未定义。
- `T shfl_up(T var, int delta) const`：参阅[束内洗牌函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484780&idx=1&sn=0697baa88af4edd2169ce2ba73c4076e&chksm=e92780d5de5009c328a0664ffae141abbbc45b5ad24ac1b97333cd307578cc40cb5ba2896b2e&token=1222151528&lang=zh_CN#rd)，仅适用于小于等于 $32$ 的分片大小。
- `T shfl_down(T var, int delta) const`：参阅[束内洗牌函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484780&idx=1&sn=0697baa88af4edd2169ce2ba73c4076e&chksm=e92780d5de5009c328a0664ffae141abbbc45b5ad24ac1b97333cd307578cc40cb5ba2896b2e&token=1222151528&lang=zh_CN#rd)，仅适用于小于等于 $32$ 的分片大小。
- `T shfl_xor(T var, int delta) const`：参阅[束内洗牌函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484780&idx=1&sn=0697baa88af4edd2169ce2ba73c4076e&chksm=e92780d5de5009c328a0664ffae141abbbc45b5ad24ac1b97333cd307578cc40cb5ba2896b2e&token=1222151528&lang=zh_CN#rd)，仅适用于小于等于 $32$ 的分片大小。
- `T any(int predicate) const`：参阅[束内表决函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484792&idx=2&sn=f915492932cde930d1e015bb3c9bebab&chksm=e92780c1de5009d7461e6a15988f5b539cb5c74035dd514ebd7f6d723408a9bbdb4bf274d690&token=1222151528&lang=zh_CN#rd)。
- `T all(int predicate) const`：参阅[束内表决函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484792&idx=2&sn=f915492932cde930d1e015bb3c9bebab&chksm=e92780c1de5009d7461e6a15988f5b539cb5c74035dd514ebd7f6d723408a9bbdb4bf274d690&token=1222151528&lang=zh_CN#rd)。
- `T ballot(int predicate) const`：参阅[束内表决函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484792&idx=2&sn=f915492932cde930d1e015bb3c9bebab&chksm=e92780c1de5009d7461e6a15988f5b539cb5c74035dd514ebd7f6d723408a9bbdb4bf274d690&token=1222151528&lang=zh_CN#rd)，仅适用于小于等于 $32$ 的分片大小。
- `unsigned int match_any(T val) const`：参阅[束内匹配函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484792&idx=1&sn=6db3d624eefed9d4276b2a149e9fa1b8&chksm=e92780c1de5009d77cb3556825ed9cbc35e2bd95aa094e631f4aa7545de90dd44d3fcde1b1e7&token=1222151528&lang=zh_CN#rd)，仅适用于小于等于 $32$ 的分片大小。
- `unsigned int match_all(T val, int &pred) const`：参阅[束内匹配函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484792&idx=1&sn=6db3d624eefed9d4276b2a149e9fa1b8&chksm=e92780c1de5009d77cb3556825ed9cbc35e2bd95aa094e631f4aa7545de90dd44d3fcde1b1e7&token=1222151528&lang=zh_CN#rd)，仅适用于小于等于 $32$ 的分片大小。


旧版成员函数（别名）：

- `unsigned long long size() const`：当前组中总的线程数量（等价于 `num_threads()`）。

备注：

- 这里使用的是 `thread_block_tile` 模板化的数据结构，组的大小作为模板参数而不是参数传递给 `tiled_partition` 调用。
- `shfl`、`shfl_up`、`shfl_down` 和 `shfl_xor` 当使用 C++11 或更高版本编译时，函数接受任何类型的对象。这意味着可以对非整数类型进行 shuffle 操作，只要它们满足以下约束：
  - 符合简单可复制的条件，即 `is_trivially_copyable<T>::value == true`；
  - `sizeof(T) <= 32`。
  
- 在计算能力 7.5 或更低的硬件上，大于 $32$ 个线程的分片需要为其保留少量内存。这可以使用 `cooperative_groups:block_tile_memory` 结构模板来完成，该模板的对象必须位于共享内存或全局内存中。


```cuda
template <unsigned int MaxBlockSize = 1024>
struct block_tile_memory;
```

其中，`MaxBlockSize` 指定当前线程块中的最大线程数。此参数可用于最小化仅使用较小线程数启动的 Kernel 中 `block_tile_memory` 的共享内存使用量。

然后，需要将此 `block_tile_memory` 传递到 `cooperative_groups:This_thread_block` 中，从而允许将生成的 `thread_block` 划分为大小大于 $32$ 的分片。调用 `this_thread_block` 函数包含 `block_tile_memory` 入参的重载版本是一个集体操作，`thread_block` 中的所有线程必须全都一起调用。

`block_tile_memory` 也可以在计算能力 8.0 或更高版本的硬件上使用，以便能够针对多个不同的计算能力编写一份源代码。在不需要的情况下，当在共享内存中实例化时，不消耗内存。

示例

以下代码将创建两个线程块分片组，大小分别为 $32$ 和 $4$：
```cuda
/// The following code will create two sets of tiled groups, of size 32 and 4 respectively:
/// The latter has the provenance encoded in the type, while the first stores it in the handle
thread_block block = this_thread_block();
thread_block_tile<32> tile32 = tiled_partition<32>(block);
thread_block_tile<4, thread_block> tile4 = tiled_partition<4>(block);
```

以下代码将在所有计算能力的设备上上创建大小为 $128$ 的线程块分片组。在计算能力 8.0 或更高版本上可以省略 `block_tile_memory`。
```cuda
/// The following code will create tiles of size 128 on all Compute Capabilities.
/// block_tile_memory can be omitted on Compute Capability 8.0 or higher.
__global__ void kernel(...) {
    // reserve shared memory for thread_block_tile usage,
    //   specify that block size will be at most 256 threads.
    __shared__ block_tile_memory<256> shared;
    thread_block thb = this_thread_block(shared);

    // Create tiles with 128 threads.
    auto tile = tiled_partition<128>(thb);

    // ...
}
```

##### 4.2.1.1 束内同步代码模式

在引入协作组之前，开发人员也会使用到 warp 内同步的场景，并且也编写了相关的代码，代码中通常对 warp 的大小做了隐含的假设，并围绕这个数字（$32$）进行编码，而现在需要显式地指定这一点。简单来说，就是以往编写代码时都知道默认 `warpSize=32`，现在使用协作组了，你要明确指定这个大小。

```cuda
__global__ void cooperative_kernel(...) {
    // obtain default "current thread block" group
    thread_block my_block = this_thread_block();

    // subdivide into 32-thread, tiled subgroups
    // Tiled subgroups evenly partition a parent group into
    // adjacent sets of threads - in this case each one warp in size
    auto my_tile = tiled_partition<32>(my_block);

    // This operation will be performed by only the
    // first 32-thread tile of each block
    if (my_tile.meta_group_rank() == 0) {
        // ...
        my_tile.sync();
    }
}
```

通常会以 $32$ 为单位进行分组，因为正好和 `warpSize` 对应；如上 `my_tile.sync()` 其实就相当于 `__syncwarp()`。

##### 4.2.1.2 单个线程的组

可以从 `this_thread` 函数中获取表示当前线程的组：

```cuda
thread_block_tile<1> this_thread();
```

以下 memcpy_async API 使用 `thread_group` 将 `int` 元素从源地址复制到目标地址：

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

cooperative_groups::memcpy_async(cooperative_groups::this_thread(), dest, src, sizeof(int));
```

使用 `this_thread` 执行异步数据拷贝的更多详细示例可以参阅 CUDA C++ Programming Guide。

#### 4.2.2 合并组（Coalesced Groups） 

在 CUDA 的 SIMT 架构中，在硬件级别，多处理器以 $32$ 个线程为一组执行操作指令，这一组线程称为 warp。如果应用程序代码中存在依赖于数据的条件分支，使得 warp 内的线程发散，则 warp 串行执行每个分支，并禁用不在该路径上的线程。在路径上保持激活状态的线程称为**合并线程**（coalesced）。协作组具有发现和创建包含所有合并线程的组的功能。

```cuda
class coalesced_group;
```

官方提供了 `coalesced_threads()` 方法来创建合并线程组句柄。在被调用的时间点返回激活线程的集合，但不保证在后续执行过程中这些合并线程始合并。

通过以下方式创建：
```cuda
coalesced_group active = coalesced_threads();
```

公有成员函数：

- `void sync() const`：同步组内线程。
- `unsigned long long num_threads() const`：当前组中总的线程数量。
- `unsigned long long thread_rank() const`：线程在组内的标号，区间为 `[0, num_threads)`。
- `unsigned long long meta_group_size() const`：返回对父组进行划分后，单个父组创建的组（分片）数。如果该组是通过查询激活线程创建的，例如 `coalesced_threads()`，则 `meta_group_size()` 的值将为 $1$。
- `unsigned long long meta_group_rank() const`：分片（tile）在父组中的标号，区间为 `[0, meta_group_size)`。如果该组是通过查询激活线程创建的，例如 `coalesced_threads()`，则 `meta_group_rank()` 的值始终为 $0$。
- `T shfl(T var, unsigned int src_rank) const`：参阅[束内洗牌函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484780&idx=1&sn=0697baa88af4edd2169ce2ba73c4076e&chksm=e92780d5de5009c328a0664ffae141abbbc45b5ad24ac1b97333cd307578cc40cb5ba2896b2e&token=1222151528&lang=zh_CN#rd)。
- `T shfl_up(T var, int delta) const`：参阅[束内洗牌函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484780&idx=1&sn=0697baa88af4edd2169ce2ba73c4076e&chksm=e92780d5de5009c328a0664ffae141abbbc45b5ad24ac1b97333cd307578cc40cb5ba2896b2e&token=1222151528&lang=zh_CN#rd)。
- `T shfl_down(T var, int delta) const`：参阅[束内洗牌函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484780&idx=1&sn=0697baa88af4edd2169ce2ba73c4076e&chksm=e92780d5de5009c328a0664ffae141abbbc45b5ad24ac1b97333cd307578cc40cb5ba2896b2e&token=1222151528&lang=zh_CN#rd)。
- `T any(int predicate) const`：参阅[束内表决函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484792&idx=2&sn=f915492932cde930d1e015bb3c9bebab&chksm=e92780c1de5009d7461e6a15988f5b539cb5c74035dd514ebd7f6d723408a9bbdb4bf274d690&token=1222151528&lang=zh_CN#rd)。
- `T all(int predicate) const`：参阅[束内表决函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484792&idx=2&sn=f915492932cde930d1e015bb3c9bebab&chksm=e92780c1de5009d7461e6a15988f5b539cb5c74035dd514ebd7f6d723408a9bbdb4bf274d690&token=1222151528&lang=zh_CN#rd)。
- `T ballot(int predicate) const`：参阅[束内表决函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484792&idx=2&sn=f915492932cde930d1e015bb3c9bebab&chksm=e92780c1de5009d7461e6a15988f5b539cb5c74035dd514ebd7f6d723408a9bbdb4bf274d690&token=1222151528&lang=zh_CN#rd)。
- `unsigned int match_any(T val) const`：参阅[束内匹配函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484792&idx=1&sn=6db3d624eefed9d4276b2a149e9fa1b8&chksm=e92780c1de5009d77cb3556825ed9cbc35e2bd95aa094e631f4aa7545de90dd44d3fcde1b1e7&token=1222151528&lang=zh_CN#rd)。
- `unsigned int match_all(T val, int &pred) const`：参阅[束内匹配函数](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484792&idx=1&sn=6db3d624eefed9d4276b2a149e9fa1b8&chksm=e92780c1de5009d77cb3556825ed9cbc35e2bd95aa094e631f4aa7545de90dd44d3fcde1b1e7&token=1222151528&lang=zh_CN#rd)。


旧版成员函数（别名）：

- `unsigned long long size() const`：当前组中总的线程数量（等价于 `num_threads()`）。

注意：`shfl`、`shfl_up`、`shfl_down` 和 `shfl_xor` 当使用 C++11 或更高版本编译时，函数接受任何类型的对象。这意味着可以对非整数类型进行 shuffle 操作，只要它们满足以下约束：

- 符合简单可复制的条件，即 `is_trivially_copyable<T>::value == true`；
- `sizeof(T) <= 32`。

示例代码：
```cuda
/// Consider a situation whereby there is a branch in the
/// code in which only the 2nd, 4th and 8th threads in each warp are
/// active. The coalesced_threads() call, placed in that branch, will create (for each
/// warp) a group, active, that has three threads (with
/// ranks 0-2 inclusive).
__global__ void kernel(int *globalInput) {
    // Lets say globalInput says that threads 2, 4, 8 should handle the data
    if (threadIdx.x == *globalInput) {
        coalesced_group active = coalesced_threads();
        // active contains 0-2 inclusive
        active.sync();
    }
  }
```

##### 4.2.2.1发现模式

通常在 warp 分支发散的场景下，开发人员有时需要使用当前激活的线程集合。比如下面这个“在 warp 中聚合跨线程的原子增量”的例子：

```cuda
{
    unsigned int writemask = __activemask();
    unsigned int total = __popc(writemask);
    unsigned int prefix = __popc(writemask & __lanemask_lt());
    // Find the lowest-numbered active lane
    int elected_lane = __ffs(writemask) - 1;
    int base_offset = 0;
    if (prefix == 0) {
        base_offset = atomicAdd(p, total);
    }
    base_offset = __shfl_sync(writemask, base_offset, elected_lane);
    int thread_offset = prefix + base_offset;
    return thread_offset;
}
```

上面的代码逻辑发生在 warp 分支发散场景的其中一个分支路径下，其计算内容就是将来自 warp 中的多个线程的原子操作组合成单个原子的过程。想象一个场景，我们要把一个数组中的满足某个条件（比如 $4$ 的倍数）的元素筛选出来转存到另一个数组中，此时我们需要维护一个计数器 `p` 用来存储有多少个元素满足条件，在 Kernel 中需要进行一次条件判断，符合条件的线程需要对计数器 `p` 进行一个原子操作更新数量的同时顺便获取一下本线程的元素在新数组中的位置，这时候就需要进行很多次原子操作，效率较低。以上代码可以将多次原子操作缩减为 $1$ 次，具体逻辑介绍如下。

首先在众多激活线程中定义一个领导线程，可以使用 `__activemask()` 原语函数，这个函数会返回一个 $32$ 位无符号整数表示当前激活线程的掩码。然后调用 `__ffs(writemask)` 返回无符号整型参数 `writemask` 中第一个（最低有效）非 $0$ 位置，最低有效位是位置 $1$。如果 `writemask` 为 $0$，`__ffs(writemask)` 将返回 $0$。把这个最低有效位对应的线程说白了就是标号最小的激活线程，这里把它定义为领导线程。

对于每个激活线程，计数器 `p` 都应该递增 $1$，所以 warp 内的总增量等于激活线程的数量，上面的代码使用 `__popc(writemask)` 求出激活线程的数量，这个原语函数可以返回无符号整型参数 `writemask` 中非零 bit 的数量。这里如果使用协作组的话，直接合并组对象调用 `size()` 函数即可。

在领导线程中执行原子操作。首先要找到领导线程，通过原语函数 `__lanemask_lt()` 返回比当前线程标号小的线程的掩码，然后与 `writemask` 进行与运算，如果为 $0$ 就说明当前线程是领导线程。在领导线程内进行原子操作，一次性将总增量加在计数器上。这里如果使用协作组的话，直接对合并组对象调用 `thread_rank()` 判断是否为 $0$，就能找到领导线程。

把领导线程中的新数组的位置广播到其他激活线程中，然后各个激活线程分别加上自身在激活线程集合中的偏移量就得到新数组中对应的位置，这一步直接使用束内洗牌函数即可，使用协作组也是一样的逻辑。

以上所有的逻辑如果使用协作组重写，代码可以精简很多，可读性大大提升，下面给出重写后的代码：

```cuda
{
    cg::coalesced_group g = cg::coalesced_threads();
    int prev;
    if (g.thread_rank() == 0) {
        prev = atomicAdd(p, g.num_threads());
    }
    prev = g.thread_rank() + g.shfl(prev, 0);
    return prev;
}
```

## 5 协作组的划分 
### 5.1 tiled_partition 

```cuda
template <unsigned int Size, typename ParentT>
thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g);

thread_group tiled_partition(const thread_group& parent, unsigned int tilesz);
```

`tiled_pa​​rtition` 方法是一种集体操作，可以将父组划分为一维、行优先的子组。总共将创建 `((size(parent)/tilesz)` 个子组，因此父组大小必须能被 `Size` 整除。目前支持的父组是 `thread_block` 或 `thread_block_tile`。

该函数被调用时可能会导致调用线程等待，直到父组的所有成员（调用线程）都完成了调用操作才能继续往下执行。前面介绍过，该函数支持 `Size` 大小跟 GPU 设备的计算能力相关，在计算能力 7.5 及以下的设备中最多只能设置为不大于 $32$ 的 $2$ 的非负整数幂，并且 `cg::size(parent)` 必须大于 `Size` 参数。`cooperative_groups::experimental` 命名空间中的实验版本可以支持 $64、128、256、512$ 大小的子组，但是计算能力 7.5 及以下的设备中需要一些额外的其他操作，具体参见[线程块分片](#421-线程块分片thread-block-tile)小节。

**使用要求**：针对子组大小大于 $32$ 的场景，要求设备计算能力不低于 5.0，且使用至少 C++11 进行编译。

下面的代码创建了一个含有 $32$ 个线程的子组：
```cuda
/// The following code will create a 32-thread tile
thread_block block = this_thread_block();
thread_block_tile<32> tile32 = tiled_partition<32>(block);
```

这些子组还可以被分成更小的组，每个组的大小为 4 个线程：

```cuda
auto tile4 = tiled_partition<4>(tile32);
// or using a general group
// thread_group tile4 = tiled_partition(tile32, 4);
```

例如，如果在 Kernel 中包含以下代码行：

```cuda
if (tile4.thread_rank()==0) printf("Hello from tile4 rank 0\n");
```

然后该语句将由 block 中的每四个线程打印：每个 `tile4` 组中 标号为 $0$ 的线程（即每个 `tile4` 线程组中的第一个线程），对应于标号为 $0、4、8、12$ 等的那些线程（即 `threadIdx.x = 0, 4, 8, 12, ...`）。

```cuda
Hello from tile4 rank 0: 0
Hello from tile4 rank 0: 4
Hello from tile4 rank 0: 8
Hello from tile4 rank 0: 12
...
```

### 5.2 labeled_partition 

```cuda
template <typename Label>
coalesced_group labeled_partition(const coalesced_group& g, Label label);

template <unsigned int Size, typename Label>
coalesced_group labeled_partition(const thread_block_tile<Size>& g, Label label);
```

`labeled_pa​​rtition` 方法是一种集体操作，可以将父组划分为线程合并的一维子组。该函数会将 `label` 值相同的线程划分到同一组中。该函数被调用时可能会导致调用线程等待，直到父组的所有成员（调用线程）都完成了调用操作才能继续往下执行。`Label` 可以是任何整数类型。

此功能目前仍在评估中，将来可能会略有变化。


**使用要求**：设备计算能力不低于 7.0，且使用至少 C++11 进行编译。

### 5.3 binary_partition 

```cuda
coalesced_group binary_partition(const coalesced_group& g, bool pred);

template <unsigned int Size>
coalesced_group binary_partition(const thread_block_tile<Size>& g, bool pred);
```

`binary_partition` 方法是一种集体操作，它将父组划分为线程合并的一维子组。该函数会将 `label` 值相同的线程划分到同一组中，是 `labeled_pa​​rtition` 的一种特殊形式，其中 `label` 值只能是 $0$ 或 $1$。该函数被调用时可能会导致调用线程等待，直到父组的所有成员（调用线程）都完成了调用操作才能继续往下执行。

此功能目前仍在评估中，将来可能会略有变化。

**使用要求**：设备计算能力不低于 7.0，且使用至少 C++11 进行编译。

下面的代码将 $32$ 个线程大小的组划分为奇数组和偶数组：

```cuda
/// This example divides a 32-sized tile into a group with odd
/// numbers and a group with even numbers
_global__ void oddEven(int *inputArr) {
    auto block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    // inputArr contains random integers
    int elem = inputArr[block.thread_rank()];
    // after this, tile32 is split into 2 groups,
    // a subtile where elem&1 is true and one where its false
    auto subtile = cg::binary_partition(tile32, (elem & 1));
}
```

## 6 协作组集体操作 

协作组库提供了一系列可以由一组线程执行的集体操作。这些操作需要组中所有线程的参与才能完成操作。除非在函数的参数描述中明确允许不同的值，否则组中的所有线程执行集体操作时函数参数必须相同。否则，调用的行为是未定义的。

### 6.1 同步 
#### 6.1.1 barrier_arrive 和 barrier_wait 

```cuda
T::arrival_token T::barrier_arrive();
void T::barrier_wait(T::arrival_token&&);
```

`barrier_arrive` 和 `barrier_wait` 成员函数提供了类似于 `cuda::barrier` 的同步 API。协作组自动初始化组屏障，但对于到达和等待操作，由于这些操作的集体性质，还有一些额外的限制：

- 协作组中的所有线程必须在每个阶段（phase）到达并等待一次屏障。
- 当协作组调用 `barrier_arrive` 时，该组的任何集体操作或另一个 `barrier_arrive` 操作的结果都是未定义的，直到调用用 `barrier_wait` 观察到屏障阶段完成为止。
- 在 `barrier_wait` 上被阻塞的线程可能会在其他线程调用 `barrier_wait` 之前从同步中释放，但只能在本协作组中所有线程调用 `barrier_arrive` 之后。
- 组类型 `T` 可以是任何隐式组。
- 这种机制使得线程在到达之后和等待同步释放之前进行一些独立的工作，从而可以隐藏一些同步延迟。
- `barrier_arrive` 返回一个 `arrival_token` 对象，该对象必须传递到相应的 `barrier_wait` 中。`token` 只能配套使用，不能用于另一个 `barrier_wait` 调用。

总的来说，相比 `cuda::barrier` 还需要显式手动地维护一个预期到达数计数器，使用协作组同步 API 可以更方便地进行实现异步屏障。

下面的代码使用 `barrier_arrive` 和 `barrier_wait` 进行集群中共享内存初始化的同步操作。

```cuda
#include <cooperative_groups.h>

using namespace cooperative_groups;

void __device__ init_shared_data(const thread_block& block, int *data);
void __device__ local_processing(const thread_block& block);
void __device__ process_shared_data(const thread_block& block, int *data);

__global__ void cluster_kernel() {
    extern __shared__ int array[];
    auto cluster = this_cluster();
    auto block   = this_thread_block();

    // Use this thread block to initialize some shared state
    init_shared_data(block, &array[0]);

    auto token = cluster.barrier_arrive(); // Let other blocks know this block is running and data was initialized

    // Do some local processing to hide the synchronization latency
    local_processing(block);

    // Map data in shared memory from the next block in the cluster
    int *dsmem = cluster.map_shared_rank(&array[0], (cluster.block_rank() + 1) % cluster.num_blocks());

    // Make sure all other blocks in the cluster are running and initialized shared data before accessing dsmem
    cluster.barrier_wait(std::move(token));

    // Consume data in distributed shared memory
    process_shared_data(block, dsmem);
    cluster.sync();
}
```

#### 6.1.2 sync 

```cuda
static void T::sync();

template <typename T>
void sync(T& group);
```

`sync` 函数用于同步协作组中的线程，相当于 `T.barrier_wait(T.barrier_arrive())`。`T` 可以是任何现有的组类型，目前所有的组类型都支持同步操作。如果组对象是 `grid_group` 或 `multi_grid_group` 类型，则必须使用适当的协作启动 API 来启动 Kernel。

### 6.2 数据传输 
#### 6.2.1 memcpy_async 
协作组中的 `memcpy_async` 是一个组范围的集体内存拷贝操作，利用硬件加速效果支持从全局内存到共享内存的非阻塞内存事务。对于一个协作组，`memcpy_async` 将通过单个管道阶段移动指定数量的字节或输入类型的元素。此外，为了在使用 `memcpy_async` API 时获得最佳性能，共享内存和全局内存都需要对齐 $16$ 个字节。重要的是要注意，虽然在一般情况下这是一个内存拷贝，但只有当源地址是全局内存、目标地址是共享内存并且两者都可以用 $16$、$8$ 或 $4$ 字节对齐寻址时，它才是异步的。异步拷贝的数据只能在调用 `wait` 或 `wait_prior` 后读取，这表明相应阶段已完成将数据移动到共享内存。

必须等待所有未完成的请求可能会失去一些灵活性（但会变得简单）。为了有效地重叠数据传输和执行，重要的是能够在等待第 N 阶段的操作请求时启动第 N+1 阶段的 `memcpy_async` 请求。为此，请使用 `memcpy_async` 并使用 `wait_prior` API 等待异步拷贝。详细信息请参阅[barrier_arrive 和 barrier_wait](#611-barrier_arrive-和-barrier_wait) 小节。

用法一：

```cuda
template <typename TyGroup, typename TyElem, typename TyShape>
void memcpy_async(
    const TyGroup &group,
    TyElem *__restrict__ _dst,
    const TyElem *__restrict__ _src,
    const TyShape &shape
);
```

完成一个 `shape` 大小的数据拷贝。

用法二：

```cuda
template <typename TyGroup, typename TyElem, typename TyDstLayout, typename TySrcLayout>
void memcpy_async(
    const TyGroup &group,
    TyElem *__restrict__ dst,
    const TyDstLayout &dstLayout,
    const TyElem *__restrict__ src,
    const TySrcLayout &srcLayout
);
```

执行 `min(dstLayout, srcLayout)` 元素的拷贝。如果内存布局的类型为 `cuda::aligned_size_t<N>`，则两者必须指定相同的对齐方式。

**勘误表**

CUDA 11.1 中引入的具有 `src` 和 `dst` 输入布局的 `memcpy_async` API 期望布局以元素而不是字节形式提供。元素类型是从 `TyElem` 推断出来的，大小为 `sizeof(TyElem)`。如果使用 `cuda::aligned_size_t<N>` 类型作为布局，指定的元素个数乘以 `sizeof(TyElem)` 必须是 N 的倍数，建议使用 `std::byte` 或 `char` 作为元素类型。

如果拷贝的指定形状或布局是 `cuda::aligned_size_t<N>` 类型，则对齐将保证至少为 `min(16, N)`。在这种情况下，`dst` 和 `src` 指针都需要与 N 个字节对齐，并且复制的字节数需要是 N 的倍数。

**使用要求**：设备计算能力不低于 7.0，且使用至少 C++11 进行编译。使用时需要包含 `cooperative_groups/memcpy_async.h` 头文件。

下面的代码将 `elementsPerThreadBlock` 大小的数据从全局内存传输到大小有限的共享内存中进行操作。

```cuda
/// This example streams elementsPerThreadBlock worth of data from global memory
/// into a limited sized shared memory (elementsInShared) block to operate on.
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void kernel(int* global_data) {
    cg::thread_block tb = cg::this_thread_block();
    const size_t elementsPerThreadBlock = 16 * 1024;
    const size_t elementsInShared = 128;
    __shared__ int local_smem[elementsInShared];

    size_t copy_count;
    size_t index = 0;
    while (index < elementsPerThreadBlock) {
        cg::memcpy_async(tb, local_smem, elementsInShared, global_data + index, elementsPerThreadBlock - index);
        copy_count = min(elementsInShared, elementsPerThreadBlock - index);
        cg::wait(tb);
        // Work with local_smem
        index += copy_count;
    }
}
```

####  6.2.2 wait 和 wait_prior 

```cuda
template <typename TyGroup>
void wait(TyGroup & group);

template <unsigned int NumStages, typename TyGroup>
void wair_prior(TyGroup & group);
```

`wait` 和 `wait_prior` 集体操作用于等待 `memcpy_async` 执行完毕。线程调用 `wait` 后进入阻塞，直到所有先前的异步拷贝都执行完毕。而 `wait_prior` 可以允许最新的 `NumStages` 个拷贝未完成，只要先前的所有拷贝执行完毕即可解除阻塞。因此，在请求了 N 个总拷贝后，`wait_prior` 会一直等到第 `N-NumStages` 个拷贝执行完毕，此时最后 `NumStages` 个拷贝可能仍在进行中。 `wait` 和 `wait_prior` 集体操作都会同步指定的协作组。

**使用要求**：设备计算能力最低 5.0，异步计算能力 8.0，C++11。使用时需要包含 `cooperative_groups/memcpy_async.h` 头文件。

下面的代码将 `elementsPerThreadBlock` 大小的数据从全局内存传输到大小有限的共享内存中进行操作。
下面的代码将 `elementsPerThreadBlock` 大小的数据从全局内存传输到大小有限的共享内存中，分两个阶段对其进行操作。随着阶段 N 的开始，我们可以等待并在阶段 N-1 上操作。

```cuda
/// This example streams elementsPerThreadBlock worth of data from global memory
/// into a limited sized shared memory (elementsInShared) block to operate on in
/// multiple (two) stages. As stage N is kicked off, we can wait on and operate on stage N-1.
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void kernel(int* global_data) {
    cg::thread_block tb = cg::this_thread_block();
    const size_t elementsPerThreadBlock = 16 * 1024 + 64;
    const size_t elementsInShared = 128;
    __align__(16) __shared__ int local_smem[2][elementsInShared];
    int stage = 0;
    // First kick off an extra request
    size_t copy_count = elementsInShared;
    size_t index = copy_count;
    cg::memcpy_async(tb, local_smem[stage], elementsInShared, global_data, elementsPerThreadBlock - index);
    while (index < elementsPerThreadBlock) {
        // Now we kick off the next request...
        cg::memcpy_async(tb, local_smem[stage ^ 1], elementsInShared, global_data + index, elementsPerThreadBlock - index);
        // ... but we wait on the one before it
        cg::wait_prior<1>(tb);

        // Its now available and we can work with local_smem[stage] here
        // (...)
        //

        // Calculate the amount fo data that was actually copied, for the next iteration.
        copy_count = min(elementsInShared, elementsPerThreadBlock - index);
        index += copy_count;

        // A cg::sync(tb) might be needed here depending on whether
        // the work done with local_smem[stage] can release threads to race ahead or not
        // Wrap to the next stage
        stage ^= 1;
    }
    cg::wait(tb);
    // The last local_smem[stage] can be handled here
}
```

### 6.3 数据操作 
#### 6.3.1 reduce 

```cuda
template <typename TyGroup, typename TyArg, typename TyOp>
auto reduce(const TyGroup& group, TyArg&& val, TyOp&& op) -> decltype(op(val, val));
```

`reduce` 函数对传入的协作组中每个线程的指定变量执行归约操作。在计算能力 8.0 及更高的设备上利用硬件加速效果进行算术运算 add、min 或 max 操作以及逻辑运算 AND、OR、或 XOR 操作，在老一代硬件上提供软件支持。另外需要说明的是，只有 $4$ 字节数据类型可以获得硬件加速。`reduce` 函数参数说明如下：

- `group`：有效的协作组类型是 `coalesced_group` 和 `thread_block_tile`。
- `val`：满足以下要求的任何类型：
  - 符合简单可复制的条件，即 `is_trivially_copyable<T>::value == true`；
  - 对于合并组（coalesced\_group）和大小不超过 $32$ 的分片组（thread\_block\_tile）而言，`sizeof(T) <= 32`；对于更大的分片组，`sizeof(T) <= 8`；
  - 能够基于给定函数对象 `op` 进行算术或逻辑运算。
  
- `op`：有效的函数对象，可以对整数类型提供硬件加速，如 `plus()`、`less()`、`greater()`、`bit_and()`、`bit_xor()`、`bit_or()`。这些函数对象必须可以构造，因此这些对象还需要待运算的数据类型作为模板参数，例如 `plus<int>()`。`reduce` 还支持可以使用括号运算符直接调用的 lambda 函数和其他函数对象。


异步规约

```cuda
template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
void reduce_update_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);

template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
void reduce_store_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);

template <typename TyGroup, typename TyArg, typename TyOp>
void reduce_store_async(const TyGroup& group, TyArg* ptr, TyArg&& val, TyOp&& op);
```

以上 `*_async` 型 `reduce` 的变体 API 会异步地计算规约结果，并由其中一个参与线程将规约结果存储或更新到指定的目标，而不是在每个线程中返回。为了观察这些异步调用的效果，需要对调用线程组或包含它们的更大的协作组进行同步。

- 对于原子存储或更新的变体 API，`atomic` 参数可以是 cuda C++ 标准库中提供的 `cuda::atomic` 或 `cuda::atomic_ref` 对象。此变体 API 仅在 CUDA C++ 标准库支持的平台和设备上可用。规约结果会根据指定的 `op` 自动更新到 `atomic`，例如。在指定 `cg::plus()` 的情况下，规约结果被原子地添加到 `atomic` 中。`atomic` 对象中的元素类型必须和 `TyArg` 相匹配。`atomic` 对象中的作用域必须包括 `group` 中的所有线程，如果多个组同时使用同一个 `atomic` 对象，则作用域必须包括使用它的所有组中的所有线程。
- 对于存储的变体 API，规约的结果将被弱存储到 `dst` 指针中。


**使用要求**：设备计算能力最低 5.0，硬件加速要求计算能力 8.0 以上，C++11。使用时需要包含 `cooperative_groups/reduce.h` 头文件。

下面的代码展示了整数向量的近似标准偏差计算过程：

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

/// Calculate approximate standard deviation of integers in vec
__device__ int std_dev(const cg::thread_block_tile<32>& tile, int *vec, int length) {
    int thread_sum = 0;

    // calculate average first
    for (int i = tile.thread_rank(); i < length; i += tile.num_threads()) {
        thread_sum += vec[i];
    }
    // cg::plus<int> allows cg::reduce() to know it can use hardware acceleration for addition
    int avg = cg::reduce(tile, thread_sum, cg::plus<int>()) / length;

    int thread_diffs_sum = 0;
    for (int i = tile.thread_rank(); i < length; i += tile.num_threads()) {
        int diff = vec[i] - avg;
        thread_diffs_sum += diff * diff;
    }

    // temporarily use floats to calculate the square root
    float diff_sum = static_cast<float>(cg::reduce(tile, thread_diffs_sum, cg::plus<int>())) / length;

    return static_cast<int>(sqrtf(diff_sum));
}
```

上面的代码在一个含有 $32$ 个线程的 `block_tile` 内完成了长度为 `length` 的整数向量的近似标准差计算。公式如下：

$$
  \sqrt{\frac{\sum_{i = 1}^{n} (x_i - \bar{x})^2 }{n}}
$$

第一步先计算向量的近似平均值 $\bar{x}$，组内每个线程使用 `tile.num_threads()` 步长将向量中的元素循环累加到每个线程的 `thread_sum` 中，然后调用 `reduce` 函数计算组内 `thread_sum` 的规约求和，除以 `length` 得到平均值 `avg`（即，$\bar{x}$）。第二步计算近似标准差，组内每个线程使用 `tile.num_threads()` 步长将元素与 `avg` 差值的平方循环累加到每个线程的 `thread_diffs_sum` 中，然后调用 `reduce` 函数计算组内 `thread_diffs_sum` 的规约求和，除以 `length` 得到平均值 `diff_sum`，最后开平方得到近似标准差。

下面的代码展示了 block 范围的规约计算过程：

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg=cooperative_groups;

/// The following example accepts input in *A and outputs a result into *sum
/// It spreads the data equally within the block
__device__ void block_reduce(const int* A, int count, cuda::atomic<int, cuda::thread_scope_block>& total_sum) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int thread_sum = 0;

    // Stride loop over all values, each thread accumulates its part of the array.
    for (int i = block.thread_rank(); i < count; i += block.size()) {
        thread_sum += A[i];
    }

    // reduce thread sums across the tile, add the result to the atomic
    // cg::plus<int> allows cg::reduce() to know it can use hardware acceleration for addition
  cg::reduce_update_async(tile, total_sum, thread_sum, cg::plus<int>());

  // synchronize the block, to ensure all async reductions are ready
    block.sync();
}
```

上面的代码通过 warp 内规约结合原子操作实现了 block 范围内的规约计算。将 block 划分为多个含有 $32$ 个线程的 `block_tile`（即， warp），调用 `reduce_update_async` 函数将 `block_tile` 内的 `thread_sum` 更新到 `atomic` 对象 `total_sum` 中，注意这时 block 中所有的 `block_tile` 都会执行规约和更新操作。最后对 block 进行同步，确保 block 范围内的异步规约操作都执行完毕，这里要注意，因为 `atomic` 对象 `total_sum` 是被 block 内的所有 `block_tile` 对象同时使用的，所以同步的时候要使用 `block.sync()`，而不能使用 `tile.sync()`。

#### 6.3.2 Reduce Operators 

下面是一些可以作为 `reduce` 函数的 `op` 参数对象进行一些基本规约操作的函数对象：
```cuda
namespace cooperative_groups {
    template <typename Ty>
    struct cg::plus;

    template <typename Ty>
    struct cg::less;

    template <typename Ty>
    struct cg::greater;

    template <typename Ty>
    struct cg::bit_and;

    template <typename Ty>
    struct cg::bit_xor;

    template <typename Ty>
    struct cg::bit_or;
}
```

功能说明：

- `cg::plus`：接受两个值并使用 `operator+` 返回两者的总和。
- `cg::less`：接受两个值并使用 `operator<` 返回较小的值。与 STL 不同之处在于返回的是较小的值而不是布尔值。
- `cg::greater`：接受两个值并使用 `operator<` 返回较大的值。与 STL 不同之处在于返回的是较大的值而不是布尔值。
- `cg::bit_and`: 接受两个值并返回 `operator&` 的结果。
- `cg::bit_xor`: 接受两个值并返回 `operator^` 的结果。
- `cg::bit_or`: 接受两个值并返回 `operator|` 的结果。


示例：

```cuda
{
    // cg::plus<int> is specialized within cg::reduce and calls __reduce_add_sync(...) on CC 8.0+
    cg::reduce(tile, (int)val, cg::plus<int>());

    // cg::plus<float> fails to match with an accelerator and instead performs a standard shuffle based reduction
    cg::reduce(tile, (float)val, cg::plus<float>());

    // While individual components of a vector are supported, reduce will not use hardware intrinsics for the following
    // It will also be necessary to define a corresponding operator for vector and any custom types that may be used
    int4 vec = {...};
    cg::reduce(tile, vec, cg::plus<int4>())

    // Finally lambdas and other function objects cannot be inspected for dispatch
    // and will instead perform shuffle based reductions using the provided function object.
    cg::reduce(tile, (int)val, [](int l, int r) -> int {return l + r;});
}
```

#### 6.3.3 inclusive_scan 和 exclusive_scan 

```cuda
template <typename TyGroup, typename TyVal, typename TyFn>
auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyVal>
TyVal inclusive_scan(const TyGroup& group, TyVal&& val);

template <typename TyGroup, typename TyVal, typename TyFn>
auto exclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyVal>
TyVal exclusive_scan(const TyGroup& group, TyVal&& val);
```

`inclusive_scan` 和 `exclusive_scan` 对传入的协作组中的每个线程中的指定数据执行扫描操作。对于 `exclusive_scan`（非包含扫描）每个线程的扫描结果是对线程标号（`thread_rank`）低于本线程的数据的规约结果；对于 `inclusive_scan`（包含扫描）扫描结果中还要包括本线程的数据。函数参数说明如下：

- `group`：有效的协作组类型是 `coalesced_group` 和 `thread_block_tile`。
- `val`：满足以下要求的任何类型：
  - 符合简单可复制的条件，即 `is_trivially_copyable<T>::value == true`；
  - 对于合并组（coalesced\_group）和大小不超过 $32$ 的分片组（thread\_block\_tile）而言，`sizeof(T) <= 32`；对于更大的分片组，`sizeof(T) <= 8`；
  - 能够基于给定函数对象 `op` 进行算术或逻辑运算。
  
- `op`：有效的函数对象，可以对整数类型提供硬件加速，如 `plus()`、`less()`、`greater()`、`bit_and()`、`bit_xor()`、`bit_or()`，见[Reduce Operators](#632-reduce-operators) 小节。这些函数对象必须可以构造，因此这些对象还需要待运算的数据类型作为模板参数，例如 `plus<int>()`。`inclusive_scan` 和 `exclusive_scan` 还支持可以使用括号运算符直接调用的 lambda 函数和其他函数对象。


扫描更新

```cuda
template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
auto inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyAtomic, typename TyVal>
TyVal inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);

template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
auto exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyAtomic, typename TyVal>
TyVal exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);
```

以上 `*_scan_update` 型变体 API 中有一个附加参数 `atomic`，可以是 cuda C++ 标准库中提供的 `cuda::atomic` 或 `cuda::atomic_ref` 对象。这些变体 API 仅在 CUDA C++ 标准库支持的平台和设备上可用。`atomic` 对象的初始值（执行扫描之前就有的值）将会作为规约初始值参与扫描计算，并将结果返回到协作组内的线程。`atomic` 对象中的元素类型必须和 `TyVal` 相匹配。`atomic` 对象中的作用域必须包括 `group` 中的所有线程，如果多个组同时使用同一个 `atomic` 对象，则作用域必须包括使用它的所有组中的所有线程。

下面的伪代码说明了 `*_scan_update` 型变体 API 的计算原理：

```cuda
/*
inclusive_scan_update behaves as the following block,
except both reduce and inclusive_scan is calculated simultaneously.
auto total = reduce(group, val, op);
TyVal old;
if (group.thread_rank() == selected_thread) {
    atomicaly {
        old = atomic.load();
        atomic.store(op(old, total));
    }
}
old = group.shfl(old, selected_thread);
return op(inclusive_scan(group, val, op), old);
*/
```

**使用要求**：设备计算能力最低 5.0，C++11。使用时需要包含 `cooperative_groups/scan.h` 头文件。

下面的代码展示了 `inclusive_scan` 的基本用法：
```cuda
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    auto thread_block = cg::this_thread_block();
    auto tile = cg::tiled_partition<8>(thread_block);
    unsigned int val = cg::inclusive_scan(tile, tile.thread_rank());
    printf("%u: %u\n", tile.thread_rank(), val);
}

/*  prints for each group:
    0: 0
    1: 1
    2: 3
    3: 6
    4: 10
    5: 15
    6: 21
    7: 28
*/
```

下面的代码展示了使用 `exclusive_scan` 进行流压缩（stream compaction）的计算过程：

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

// put data from input into output only if it passes test_fn predicate
template<typename Group, typename Data, typename TyFn>
__device__ int stream_compaction(Group &g, Data *input, int count, TyFn&& test_fn, Data *output) {
    int per_thread = count / g.num_threads();
    int thread_start = min(g.thread_rank() * per_thread, count);
    int my_count = min(per_thread, count - thread_start);

    // get all passing items from my part of the input
    //  into a contagious part of the array and count them.
    int i = thread_start;
    while (i < my_count + thread_start) {
        if (test_fn(input[i])) {
            i++;
        }
        else {
            my_count--;
            input[i] = input[my_count + thread_start];
        }
    }

    // scan over counts from each thread to calculate my starting
    //  index in the output
    int my_idx = cg::exclusive_scan(g, my_count);

    for (i = 0; i < my_count; ++i) {
        output[my_idx + i] = input[thread_start + i];
    }
    // return the total number of items in the output
    return g.shfl(my_idx + my_count, g.num_threads() - 1);
}
```

所谓流压缩（stream compaction），实际就是把输入数组中满足某个条件（`test_fn`）的元素存储到输出数组的过程，经过压缩后，输出数组的长度一般远小于输入数组，达到一种过滤（压缩）的效果。在上面的代码中，首先计算了协作组中每个线程需要处理的数据量 `per_thread`，并确定当前线程处理元素的起始索引 `thread_start`，通过循环计算把满足 `test_fn(input[i])` 的元素放在前面并返回满足条件的元素数量 `my_count`，此时索引处于 `[thread_start, thread_start + my_count)` 区间的元素均是满足过滤条件的。然后调用 `exclusive_scan` 对 `my_count` 进行扫描，获取当前线程中这些满足条件的元素在输出数组中的起始索引 `my_idx`，基于起始索引将 `input` 中 `[thread_start, thread_start + my_count)` 区间的元素拷贝到 `output` 数组中。最后使用束内洗牌函数将最后一个线程中计算的整个协作组范围内处理的满足过滤条件的总元素数量返回给所有调用线程。

使用 `exclusive_scan_update` 进行动态缓冲区空间分配示例代码：

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

// Buffer partitioning is static to make the example easier to follow,
// but any arbitrary dynamic allocation scheme can be implemented by replacing this function.
__device__ int calculate_buffer_space_needed(cg::thread_block_tile<32>& tile) {
    return tile.thread_rank() % 2 + 1;
}

__device__ int my_thread_data(int i) {
    return i;
}

__global__ void kernel() {
    __shared__ extern int buffer[];
    __shared__ cuda::atomic<int, cuda::thread_scope_block> buffer_used;

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    buffer_used = 0;
    block.sync();

    // each thread calculates buffer size it needs
    int buf_needed = calculate_buffer_space_needed(tile);

    // scan over the needs of each thread, result for each thread is an offset
    // of that thread’s part of the buffer. buffer_used is atomically updated with
    // the sum of all thread's inputs, to correctly offset other tile’s allocations
    int buf_offset =
        cg::exclusive_scan_update(tile, buffer_used, buf_needed);

    // each thread fills its own part of the buffer with thread specific data
    for (int i = 0 ; i < buf_needed ; ++i) {
        buffer[buf_offset + i] = my_thread_data(i);
    }

    block.sync();
    // buffer_used now holds total amount of memory allocated
    // buffer is {0, 0, 1, 0, 0, 1 ...};
}
```

动态缓冲区空间分配的过程正好与流压缩相反，在每个线程中计算需要的缓冲区大小 `buffer_needed`，调用 `exclusive_scan_update` 进行扫描，注意这里有个原子参数 `buffer_used` 通过这个原子参数，使得扫描的范围不再局限于一个分片组（`tile`）而是整个 block，因为每个分片组的扫描结果都是基于 `buffer_used` 的，因此扫描的结果实际就是调用线程在整个缓冲区中的起始索引 `buf_offset `。在调用线程内，根据起始索引和本线程的缓冲区大小，循环将线程对应的数据填入总的缓冲区即可。最终如果要观测填充后的缓冲区数据，可以使用 `block.sync()` 进行同步，确保 block 内的线程均已完成缓冲区填充。

### 6.4 执行控制 
#### 6.4.1 invoke_one 和 invoke_one_broadcast 

```cuda
template<typename Group, typename Fn, typename... Args>
void invoke_one(const Group& group, Fn&& fn, Args&&... args);

template<typename Group, typename Fn, typename... Args>
auto invoke_one_broadcast(const Group& group, Fn&& fn, Args&&... args) -> decltype(fn(args...));
```

`invoke_one` 是一个集体操作，从传入的协作组中选择任意一个线程，使用传入的参数 `args` 调用函数 `fn`。`invoke_one_broadcast` 在此基础上还会把函数 `fn` 的返回值也广播到协作组的每个线程。

在协作组中被选中线程调用函数 `fn` 前后，协作组会与该线程同步。这意味着在函数 `fn` 内不允许在协作组内部进行通信，但可以与协作组外部的线程通信。线程选择机制不能保证是确定性的。

另外，在计算能力 9.0 或更高版本的设备上，当使用显式组类型（参见[显式组](#42-显式组) 小节）调用时，选择线程可能还会有硬件加速效果。`invoke_one` 和 `invoke_one_broadcast` 函数的参数说明如下：

- `group`：所有的协作组类型都可以调用 `invoke_one`； 合并组（`coalesced_group`）和分片组（`thread_block_tile`）可以调用 `invoke_one_broadcast`；
- `fn`：函数，或者重载了 `operate()` 的对象；
- `args`：`fn` 的入参，注意要相互匹配；
- 针对 `invoke_one_broadcast` 函数，`fn` 的返回值类型还需要满足如下要求：
  - 符合简单可复制的条件，即 `is_trivially_copyable<T>::value == true`；
  - 对于合并组（coalesced_group）和大小不超过 $32$ 的分片组（thread_block_tile）而言，`sizeof(T) <= 32`；对于更大的分片组，`sizeof(T) <= 8`；
  


**使用要求**：设备计算能力最低 5.0，硬件加速要求计算能力 9.0 以上，C++11。

针对[合并组](#422-合并组coalesced-groups) 小节中的“在 warp 中聚合跨线程的原子增量”的例子，还可以使用 `invoke_one_broadcast` 重写，代码如下：

```cuda
#include <cooperative_groups.h>
#include <cuda/atomic>
namespace cg = cooperative_groups;

template<cuda::thread_scope Scope>
__device__ unsigned int atomicAddOneRelaxed(cuda::atomic<unsigned int, Scope>& atomic) {
    auto g = cg::coalesced_threads();
    auto prev = cg::invoke_one_broadcast(g, [&] () {
        return atomic.fetch_add(g.num_threads(), cuda::memory_order_relaxed);
    });
    return prev + g.thread_rank();
}
```

## 7 线程网格同步 
在引入协作组之前，CUDA 编程模型中 block 之间的同步只能发生在 Kernel 执行完成的时候，也就是说在 Kernel 边界处进行同步。但在同步的同时，Kernel 也执行完毕了，这些 block 以及执行上下文也都随之失效，这带来了一些潜在的性能影响。

例如，在某些场景中，应用程序中具有大量的小 Kernel，每个 Kernel 代表任务流水线中的一个阶段。这些小 Kernel 的存在可以确保在一个流水线阶段上运行的 block 在下一流水线阶段上运行的 block 开始使用数据之前产生数据，这其实就是一种隐式的跨 block 同步。在这种情况下，提供全局线程间 block 同步的能力可以使应用程序把这些小 Kernel 融合为一个大 Kernel，从而将 block 以及一些计算数据持久化，当给定阶段完成时，这些 block 能够在设备上同步。

要在 Kernel 中实现 grid 间同步，可以使用 `grid.sync()` 函数：
```cuda
grid_group grid = this_grid();
grid.sync();
```

并且在启动内核时，需要使用CUDA 运行时启动 API  `cudaLaunchCooperativeKernel` 或相关 CUDA 驱动程序启动，而不能简单使用 `<<<...>>>` 执行配置语法。

### 7.1 示例 

为了保证 block 在 GPU 上的共同驻留，需要仔细考虑启动的 block 数量。 例如，可以按如下方式启动与 SM 一样多的 block：

```cuda
int dev = 0;
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
// initialize, then launch
cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount, numThreads, args);
```

或者，您可以通过使用**占用计算器**（occupancy calculator）计算每个 SM 可以同时容纳多少 block 来最大化并行程度，如下所示：

```cuda
/// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
int numBlocksPerSm = 0;
  // Number of threads my_kernel will be launched with
int numThreads = 128;
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, my_kernel, numThreads, 0);
// launch
void *kernelArgs[] = { /* add kernel args */ };
dim3 dimBlock(numThreads, 1, 1);
dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
cudaLaunchCooperativeKernel((void*)my_kernel, dimGrid, dimBlock, kernelArgs);
```

最好先通过查询设备属性 `cudaDevAttrCooperativeLaunch` 来确保设备是否支持协作启动：
```cuda
int dev = 0;
int supportsCoopLaunch = 0;
cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
```

如果设备 $0$ 支持协作启动，则 `supportsCoopLaunch` 将被设置为 $1$。协作启动仅在计算能力为 6.0 及更高版本的 GPU 设备上被支持。此外，程序对运行平台的要求如下：

- 没有 MPS 的 Linux 平台；
- 具有 MPS 的 Linux 平台以及具有 7.0 或更高计算能力的设备；
- 最新的 Windows 平台


## 8 多设备同步 
除了线程网格内同步以外，协作组还提供了更大作用域的同步机制，即跨设备同步，对于多 GPU 设备场景的 CUDA 程序，如果要实现跨多个设备的同步，需要使用 `cudaLaunchCooperativeKernelMultiDevice` CUDA API 来启动 Kernel。这个 API 与现有的 CUDA API 有很大不同，它允许单个主机线程在多个设备上启动 Kernel。除了包含 `cudaLaunchCooperativeKernel` 所具有的约束和保证之外，此 API 还具有其他语义，具体如下：

- 此 API 将确保启动是原子的，即如果 API 调用成功，则将在所有指定设备上启动给定数量的 block。
- 通过此 API 在所有设备上启动的 Kernel 必须相同。驱动程序在这方面没有进行明确的检查，需要由应用程序来确保这一点。
- `cudaLaunchParams` 中指定的任意两个 Kernel 不能在一个设备上启动。
- 本次启动所针对的所有设备都必须具有相同的计算能力，主要版本和次要版本都要相同。
- 每个网格的 block 大小、网格大小和共享内存量在所有设备上必须相同。注意，这意味着每个设备可以启动的最大 block 数量将受到 SM 数量最少的设备的限制。
- 任何 `__device__`、`__constant__` 或 `__managed__` 设备全局变量都在每个设备上独立实例化，应用程序负责确保这些变量被初始化并正确使用。


弃用通知：`cudaLaunchCooperativeKernelMultiDevice` 已在 CUDA 11.3 中针对所有设备弃用。在多设备共轭梯度样本中可以找到替代方法的示例。

通过 `cuCtxEnablePeerAccess` 或 `cudaDeviceEnablePeerAccess` 为所有参与设备启用对等访问，可实现多设备同步的最佳性能。

使用 `cudaLaunchCooperativeKernelMultiDevice` 进行多设备启动时，启动参数应当使用结构数组（每个设备一个）来定义，示例如下：

```cuda
cudaDeviceProp deviceProp;
cudaGetDeviceCount(&numGpus);

// Per device launch parameters
cudaLaunchParams *launchParams = (cudaLaunchParams*)malloc(sizeof(cudaLaunchParams) * numGpus);
cudaStream_t *streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * numGpus);

// The kernel arguments are copied over during launch
// Its also possible to have individual copies of kernel arguments per device, but
// the signature and name of the function/kernel must be the same.
void *kernelArgs[] = { /* Add kernel arguments */ };

for (int i = 0; i < numGpus; i++) {
    cudaSetDevice(i);
    // Per device stream, but its also possible to use the default NULL stream of each device
    cudaStreamCreate(&streams[i]);
    // Loop over other devices and cudaDeviceEnablePeerAccess to get a faster barrier implementation
}
// Since all devices must be of the same compute capability and have the same launch configuration
// it is sufficient to query device 0 here
cudaGetDeviceProperties(&deviceProp[i], 0);
dim3 dimBlock(numThreads, 1, 1);
dim3 dimGrid(deviceProp.multiProcessorCount, 1, 1);
for (int i = 0; i < numGpus; i++) {
    launchParamsList[i].func = (void*)my_kernel;
    launchParamsList[i].gridDim = dimGrid;
    launchParamsList[i].blockDim = dimBlock;
    launchParamsList[i].sharedMem = 0;
    launchParamsList[i].stream = streams[i];
    launchParamsList[i].args = kernelArgs;
}
cudaLaunchCooperativeKernelMultiDevice(launchParams, numGpus);
```

此外，与线程网格范围的同步一样，设备范围的同步可使用如下代码：

```cuda
multi_grid_group multi_grid = this_multi_grid();
multi_grid.sync();
```

但是，在编译时需要将编译选项 `-rdc=true` 传递给 nvcc 来单独编译代码。

最好先通过查询设备属性 `cudaDevAttrCooperativeMultiDeviceLaunch` 来确保设备支持多设备协作启动：

```cuda
int dev = 0;
int supportsMdCoopLaunch = 0;
cudaDeviceGetAttribute(&supportsMdCoopLaunch, cudaDevAttrCooperativeMultiDeviceLaunch, dev);
```

如果设备 $0$ 支持多设备协作启动，则 `supportsMdCoopLaunch` 将被设置为 $1$。多设备协作启动仅在计算能力为 6.0 及更高版本的 GPU 设备上被支持。此外，程序对运行平台的要求如下：

- 没有 MPS 的 Linux 平台；
- 最新的 Windows 平台，并且设备处于 TCC 模式。

有关更多信息，请参阅 `cudaLaunchCooperativeKernelMultiDevice` API 文档。
