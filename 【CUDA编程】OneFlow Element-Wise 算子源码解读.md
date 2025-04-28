#! https://zhuanlan.zhihu.com/p/646990764

# 【CUDA编程】OneFlow Element-Wise 算子源码解读

**写在前面**：这篇文章是笔者阅读 oneflow 的 cuda 源码后的知识点总结，读了大神开源代码才知道学习的道路还很漫长。。。人学始知道，不学非自然。

## 1 Elementwise 操作
Elementwise 操作，即逐元素操作，是指对 Tensor 中的每个元素应用一个函数变换，得到最终输出结果。在深度学习里，有很多算子属于 Elementwise 算子范畴，比如常用的激活函数（如ReLU、GELU、Sigmoid）、ScalarMultiply（对 Tensor 每个元素都乘上一个标量）等操作。  
以最常用的 Sigmoid 函数为例，通常我们会给出如下的核函数实现：  

```cuda
/**
 * @brief 单片内核sigmoidKernel
 * 
 * @param in 输入tensor
 * @param out 输出tensor
 * @param N tensor长度
 * @return __global__ 
 */
__global__ void sigmoidKernel(const float* in, float* out, const int N) 
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = 1.0 / (1.0 + expf(-in[tid]));
    }
}
```
以上这种写法虽然可读性很强，但扩展性较差，而且存在明显的性能问题。因此本文将基于 OneFlow 开源的 Element-Wise CUDA 算子源码来解释如何写一个高性能的 Element-Wise CUDA 算子。这里给出源码链接，有兴趣的读者可以下载下来学习：https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/elementwise.cuh

## 2 源码解读
### 2.1 给 Element-Wise 操作设置合理的 GridSize 和 BlockSize
#### 2.1.1 设置 BlockSize
要选择合适的 BlockSize 首先应该获取其可供选择的取值范围。  
（1）最大值限制    
首先在 CUDA 中对 BlockSize 的最大值有明确的限制，即 BlockSize 最大可以取 1024。  
（2）线程束限制（取线程束大小整数倍）  
我们知道，在同一个 block 中连续的 32 个 thread 组成一个 **warp**，这 32 个 thread 每次执行同一条指令（指令级别的并行），因此，BlockSize 最好设置为 32 的整数倍，通常我们直接取 $2^n$。想象一下如果 BlockSize 为 32 的整数倍加一（比如65）会发生什么？这种情况下，最后一个 warp 中将只有一个有效线程，但是它占用的硬件资源和其他 warp 是一样的，也就是说最后一个 warp 的资源有效利用率只有 1/32，在实际应用中要避免这种情况。  
（3）SM 占有率限制  
SM，即流多处理器，是 GPU 的基本组成单元，一个 GPU 由多个 SM 构成。SM 为同一个 block 中的线程提供通信和同步等所需的硬件资源，跨 SM 不支持线程间的通信，所以一个 block 中的所有线程都是执行在同一个 SM 上的。SM 允许多于一个 block 在其上并发执行，如果一个 SM 空闲的资源满足一个 block 的执行，那么这个 block 就可以被立即调度到该 SM 上执行，具体的硬件资源一般包括寄存器、shared memory、以及各种调度相关的资源。有些情况下，一个 SM 中驻留的线程数目有可能达不到理想的最大值，这时候我们说 **SM 占有率**小于100%。这里我们给出两个 SM 相关的指标：

- 单个 SM 中最多线程块个数 $N_b$：开普勒架构和图灵架构为 16，麦克斯韦架构、帕斯卡架构、伏特架构为 32。
- 单个 SM 中最多线程个数 $N_t$：图灵架构为 1024，其他架构（从开普勒到伏特）为 2048。

显然，当 BlockSize 小于 $\frac{N_t}{N_b}$ 或不能整除 $N_t$ 时，SM 占有率是不可能达到 100% 的。前者即使 SM 驻留线程块个数达到最大，线程总数也达不到最大值；后者单个 SM 上线程块个数取不到最大。在开普勒架构中 $\frac{N_t}{N_b} = 128$，其他架构中 $\frac{N_t}{N_b}=64$，考虑到硬件适配性我们取 $BlockSize \geq 128$。  
（4）寄存器数量限制  
在CUDA 官方文档 Compute Capabilities（https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities）中提到了：

- 主流架构里，每个 Block 最大寄存器数量是 64 K
- 每个线程所能使用的最大寄存器数量是 255 个

为了保证每个 thread 都有最大寄存器数量可供使用，那每个 block 最多能启动 `64 * 1024 / 255 = 256` 个线程（往 2 的倍数取整）。  
综上所述，最终这里设定了一个常量 `constexpr int kBlockSize = 256`，当然 128 也是满足要求的。

#### 2.1.2 设置 GridSize 
确定了 BlockSize 之后便可以进一步确定 GridSize，也就是确定总的线程数量。对于一般的 elementwise kernel 来说，总的线程数量应不大于总的 element 数量，也就是一个线程至少处理一个 element，同时 GridSize 也有上限，目前在主流架构上都是 $2^{31} - 1$，对于绝大多数情况都是够用的。  
（1）单片内核  
创建线程在 GPU 上是一个开销很低的操作，所以很多情况下为每个 element 创建一个线程（我们称之为单片内核）是可行的，比如本文开头我们实现的核函数 `sigmoidKernel` 就是这样。但是在某些特殊情况下，比如每个线程都包含一个公共的操作（比如读取同一块全局内存），那么线程数的增多，也代表着这个读取操作的开销变大，如果减小线程数量并循环处理，那么这个读取操作的开销就会降低。  
（2）tail effect  
如何做到减小线程数量并循环处理呢？我们可以想象，GPU 一次可以调度 SM 数量 * 每个 SM 最大 block 数个 block，由于每个 block 的计算量相等，所以所有 SM 应几乎同时完成这些 block 的计算，然后处理下一批，这其中的每一批被称之为一个 **wave**。想象如果 GridSize 恰好比一个 wave 多出一个 block，由于 stream 上的下个 kernel 要等这个 kernel 完全执行完成后才能开始执行，所以第一个 wave 完成后，GPU 上将只有一个 block 在执行，GPU 的实际利用率会很低，这种情况被称之为 **tail effect**。  
我们应尽量避免这种情况，将 grid_size 设置为精确的一个 wave 可能也无法避免 tail effect，因为 GPU 可能不是被当前 stream 独占的，常见的如 NCCL 执行时会占用一些 SM。所以无特殊情况可以将 grid_size 设置为数量足够多的整数个 wave，也就是循环次数多一些。  
总结一下：

- 线程块最小个数为 1。
- 线程块个数应从 `单片内核对应的线程块个数` 和 `wave 数目 * GPU 一次可以调度 SM 数量 * 每个 SM 最大 block 数` 中取较小值，这里 wave 数目设置为 32。宏观上看，元素个数少用前者，否则用后者。

具体代码实现如下：
```cuda
constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;

/**
* @brief Get the Num of Blocks
*
* @param n, num of elements
* @param num_blocks
* @return cudaError_t
*/
inline cudaError_t GetNumBlocks(int64_t n, int *num_blocks)
{
    int dev; // which device is currently being used.
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess)
        { return err; }
    }
    int sm_count; // Number of multiprocessors on the device，即流多处理器的数量
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev); // information about the device.
        if (err != cudaSuccess)
        { return err; }
    }
    int tpm; // Maximum resident threads per multiprocessor，即每个流多处理器的最大驻留线程数
    {
        cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess)
        { return err; }
    }
    *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize, sm_count * tpm / kBlockSize * kNumWaves));
    return cudaSuccess;
}
```
### 2.2 使用向量化的数据提升带宽
大部分 Element wise 算子的计算逻辑较为简单，性能瓶颈主要是在带宽利用上。英伟达官方博客[CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)提到，使用向量化操作能够提升读写的带宽，而 CUDA 里也提供了一系列数据类型来支持向量化操作，如 `float2、float4`，就是将 2 个或 4 个 `float` 数据作为一个整体。实际应用中，算子肯定需要支持不同数据类型（如 `float、half、double`），如果采用 cuda 内置的向量化操作，就需要给每个算子写多个版本，代码冗余过大，这里 oneflow 自定义了一个 `Pack` 数据结构，用于灵活支持不同数据类型的向量化操作。

#### 2.2.1 定义PackType
首先利用了 C++ 内存对齐新特性，定义了一个大小为 `sizeof(T) * pack_size`，对齐要求为 `sizeof(T) * pack_size` 的数据类型 `PackType`，代表向量化后的数据类型。
```cuda
template <typename T, int pack_size>
struct GetPackType
{
    using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template <typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;
```
这里有一个坑，笔者一开始编译的时候会报编译错误，error 信息如下：
```
error: static assertion failed with "You've instantiated std::aligned_storage<Len, Align> with an extended alignment (in other words, Align > alignof(max_align_t)). Before VS 2017 15.8, the member type would non-conformingly have an alignment of only alignof(max_align_t). VS 2017 15.8 was fixed to handle this correctly, but the 
fix inherently changes layout and breaks binary compatibility (*only* for uses of aligned_storage with extended alignments). Please define either (1) _ENABLE_EXTENDED_ALIGNED_STORAGE to acknowledge that you understand this message and that you actually want a type with an extended alignment, or (2) _DISABLE_EXTENDED_ALIGNED_STORAGE to silence this message and get the old non-conformant behavior."
```
我们在 Pack 数据的时候，Pack 后的类型设定的对齐值一般会大于基本对齐值（笔者环境下是 8，可以用 `alignof(std::max_align_t)` 获取），这种做法称为扩展对齐。在 VS 2017 15.8 之前这里有个 BUG，扩展对齐场景下成员类型的对齐方式仅为 `alignof(max_align_t)`，也就是基本对齐值。VS 2017 15.8 修复了这个 BUG，但该修复固有地改变了布局并破坏了二进制兼容性带来了新BUG。所以这里的报错主要是给开发者提醒用的，如果知道这个信息并坚持使用新版的扩展对齐，需要在编译指令加上 `-D _ENABLE_EXTENDED_ALIGNED_STORAGE`；如果只是想忽略该报错并使用旧版本，那么在编译时加上 `-D _DISABLE_EXTENDED_ALIGNED_STORAGE`，当然这样的话，在扩展对齐时，无论 `std::aligned_storage<std::size_t Len,  std::size_t Align = /*default-alignment*/>` 的 Align 参数传多大的值最终该类型的对齐大小均为 `std::max_align_t`，笔者亲测传 `16、32` 都没用，最终 `alignof(PackType<T, pack_size>)` 都是 8，另外还会导致 `sizeof(PackType<T, pack_size>)`、 `sizeof(T) * pack_size`以及 `alignof(PackType<T, pack_size>)` 三者不相等，天坑，慎避。。。综上所述，我们在编译的时候要加 `-D _ENABLE_EXTENDED_ALIGNED_STORAGE`，编译的时候要加 `-D _ENABLE_EXTENDED_ALIGNED_STORAGE`，要加 `-D _ENABLE_EXTENDED_ALIGNED_STORAGE`！！！ 

#### 2.2.2 定义 Pack 和 Packed
然后 oneflow 实现了一个 union 类型 `Pack`，它内部定义了 `PackType<T, pack_size> storage` 来占用空间，与 `storage` 共享内存的，还有 `T elem[pack_size]`。这样方便后续的 Elementwise 操作：在后续计算里，会对 elem 数组中的每个元素都应用 functor，得到输出结果。
```cuda
/**
* @brief 判断类型T的内存长度是否符合预期
*
* @tparam T
* @tparam pack_size
*/
template <typename T, int pack_size>
union Pack
{
    static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "Memory size does not meet expectations");
    __device__ Pack()
    {
        // do nothing
    }
    PackType<T, pack_size> storage; // 占位用的
    T elem[pack_size];
};
```
在后面的调用链中可以发现，这里的联合体 `Pack` 只是起到一个编译期断言的作用，判断类型T的内存长度是否符合预期。如果不符合，就无法完成 Pack。  
接下来 oneflow 定义了真正用于执行数据 Pack 的数据结构 `Packed`。
```cuda
template <typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed
{
    __device__ Packed()
    {
        // do nothing
    }
    union
    {
        T elem[pack_size]; // 这里联合体只有一个成员，应该是为了方便后期扩展
    };
};
```
这里有 3 点值得注意：
- 利用修饰符 `alignas` 将 `Packed` 类型的内存对齐方式设置为 `sizeof(T) * pack_size`，也就是 `pack_size` 个数据类型 `T` 所占的的字节数。这里要注意 `alignas` 要求对齐值必须要是 2 的自然数次幂，也就意味着 `pack_size` 必须取 2 的自然数次幂。
- 用 `__device__` 修饰无参构造函数的目的是让 `Packed` 能够在设备端被实例化（后面的 `ApplyPack` 函数中），因为设备端无法调用主机端的函数。
- `union { T elem[pack_size]; };` 是 `Packed` 的数据存储部分，就是一个简单的数组。

#### 2.2.3 计算 PackSize
oneflow 源码中给出了计算 `pack_size` 的工具函数。
```cuda
constexpr int kMaxPackBytes = 128 / 8; //  CUDA 最多支持 128 个 bit 的访问粒度
constexpr int kMaxPackSize = 8;        // half 类型占 2 个字节，也就是 16 个 bit，所以最大可以 Pack 的数量为 128 / 16 = 8

constexpr int Min(int a, int b) 
{ 
    return a < b ? a : b; 
}

template <typename T>
constexpr int PackSize()
{
    return Min(kMaxPackBytes / sizeof(T), kMaxPackSize);
}

template <typename T, typename U, typename... Args>
constexpr int PackSize()
{
    return Min(PackSize<T>(), PackSize<U, Args...>());
}
```
CUDA 里最大支持 128 bit 的 pack 大小，而在浮点数据类型中，最小的类型（half）大小为 16 bit，最多能把 128 / 16=8 个 half 数据 pack 到一起，因此 oneflow 设置了两个编译期常量，`kMaxPackBytes` 表示 pack 最大字节数，`kMaxPackSize` 表示 pack 数据的最大个数。  
有读者看到代码后可能会问：为什么每个函数的返回值前面都有一个关键字 `constexpr` ？ `constexpr` 是 C++ 11 标准新添加的关键字，目的是解决 `const` 关键字的双重语义问题，即如果要表示这个变量只读，推荐用 `const`；**如果表示这是一个编译期常量，用 `constexpr`**。这里的  `pack_size` 是后续程序用来初始化容器的，必须要设置为常量，为这几个函返回值加 `constexpr` 修饰后，编译器在编译期就可以将结果计算出来。  
核心代码是两个 `PackSize` 函数，利用了 C++ 11 的新特性——**可变参数模板函数**。初看的时候可能一头雾水，一个 `PackSize` 函数就已经可以求出类型 T 的 `pack_size` 了，第二个 `PackSize` 函数有什么用？其实这里是为了后续对  Element-Wise 操作做一个扩展，使其可以适配多元 Element-Wise 操作。两个 `PackSize` 函数组合起来利用重载和递归的机制求出任意元数据组合的整体 `pack_size`。按如下实现也是可以的：

```cuda
template <int MaxPackSize>
constexpr int PackSize()
{
	return  MaxPackSize;
}

template <int MaxPackSize, typename T, typename... Args>
constexpr int PackSize()
{
	return Min(kMaxPackBytes / sizeof(T), PackSize<MaxPackSize, Args...>());
}
```
### 2.3 针对 half2 数据类型优化
half 即 FP16 是 cuda 7.5 引入的一种占 16 个 bit 的数据类型，也称为**半精度浮点**，要求设备计算能力在 5.3及以上。为了提升 half 类型基础运算的性能，通常通过 half2 数据类型来进行半精度浮点计算。这里要注意，half2 并不是简单的两个 half 组成的数组，其定义是一个 `unsigned int`，所以不能向其它数据类型一样直接使用 `+、-、*、/` 运算符完成运算。针对 half、half2 的基础运算，CUDA 官方推出了一系列特殊指令，封装在 `cuda_fp16.h` 头文件里，如 `__hadd2` 可以实现两个 half2 数据的加法，`__hmul2` 可以实现两个 half2 数据的乘法，`__halves2half2` 可以将两个半精度值转换为 half2 数据类型，以及 half2 和 float 之间的转换等等。  
考虑到 half2 计算的特殊性，oneflow 给 `ApplyPack` 函数特化了一个版本，通过调用 `functor` 的 `apply2` 函数，来调用 half2 相关特殊指令，接口如下：

```cuda
/**
* @brief 对一个 pack 内的元素做循环，对 elem 数组中的每个元素调用 functor ，得到输出结果并返回
* OneFlow 给 ApplyPack 函数特化了一个版本，通过调用 functor 的 apply2 函数，来调用 half2 相关特殊指令
* 
* @tparam pack_size 
* @tparam FunctorT 
* @tparam R 
* @tparam IN 
* @param functor 
* @param in 
* @return __device__ 
*/
template <int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == true && pack_size % 2 == 0, Packed<R, pack_size>>::type
ApplyPack(const FunctorT &functor, const Packed<IN, pack_size>... in)
{
    Packed<R, pack_size> ret;
    #pragma unroll
    for (int j = 0; j < pack_size; j += 2)
    {
        functor.Apply2(ret.elem + j, (in.elem + j)...);
    }
    return ret;
}
```
先来看 `ApplyPack` 函数的返回值 `typename std::enable_if<HasApply2<FunctorT>::value == true && pack_size % 2 == 0, Packed<R, pack_size>>::type`，利用 `std::enable_if` 来限制当 `HasApply2<FunctorT>::value == true` 时模板有效，也就是说只有满足 `HasApply2<FunctorT>::value == true`，在编译期才会匹配上这个 `ApplyPack` 函数，否则可能匹配到其他重载的 `ApplyPack` 函数，oneflow 在源码中也提供了其他情况下的 `ApplyPack` 函数。  
这里的 `HasApply2<FunctorT>::value == true` 是指 `FunctorT` 中含有函数名为 `Apply2` 的成员函数，也就是说在编译期可以通过 `HasApply2` 判断 `FunctorT` 是否含 `Apply2` 函数。具体实现看源码：
```cuda
/**
* @brief 该模板的作用是在编译期判断类型 T 是否具有成员方法 Apply2
*    
* @tparam T
*/
template <typename T>
class HasApply2
{
    // 定义两个数据类型：one(占一个字节) 和 two(占2个字节)，用于成员方法的返回值类型
    typedef char one;
    struct two
    {
        char x[2];
    };
    // 声明两个test成员函数模板，利用了函数重载以及其调用顺序，这里两个重载test方法无需定义
    template <typename C>
    static one test(decltype(&C::Apply2));  // decltype用来获取参数的类型，表明这个函数的参数类型是decltype(&C::Apply2)，这里省略了形参所以不好理解
    template <typename C>
    static two test(...);   // 省略了参数，表示可变长参数

    public:
    /**
    * @brief 这里联合体只有一个成员，后续方便扩展
    *        value为true表示T中有成员函数Apply2，为false表示没有
    *        当T中有成员函数Apply2，test<T>(0)调用的是"static one test(decltype(&C::Apply2));"所以其返回值类型为one；否则
    * 将调用"static two test(...);"返回值为two
    */
    enum
    {
        value = sizeof(test<T>(0)) == sizeof(char) 
    };
};
```
关于在编译期如何判断类型中是否含有特定成员函数的方法，笔者在代码注释中做了解释，这里就不细讲了，笔者在另一篇文章有详细介绍，有兴趣的读者可以移步：[C++中如何在编译期获取指定类型中是否有某个成员函数](https://mp.weixin.qq.com/s/5xADji2NNTCYWjpJcvtYLQ)  
综上所述，**如果 Element-Wise 操作的元素类型是 half2，必须在 `functor` 中定义  `HasApply2` 函数**。  

### 2.4 调用链

通读 `oneflow/core/cuda/elementwise.cuh` 源代码可以发现，源码中给出了一元、二元、三元的 Element-Wise 操作接口：`Unary、Binary、Ternary`，其基本调用关系如下：

```cuda
Unary/Binary/Ternary
  -> xxxWithFactory
     -> GenericLauncher<...>::Launch
       -> LaunchKernel
         -> ApplyGeneric(CUDA Kernel)
```
前两层代码逻辑比较简单，就是一个简单工厂模式，主要是方便扩展 `functor`，有兴趣的读者可以了解一下设计模式，这里不再赘述，我们从 `GenericLauncher` 开始讲起。

#### 2.4.1 GenericLauncher

```cuda
template <typename FactoryT, typename R, typename... IN>
struct GenericLauncher
{
    static cudaError_t Launch(FactoryT factory, int64_t n, R *r, const IN *...in,
    cudaStream_t stream)
    {
        constexpr int max_pack_size = PackSize<R, IN...>();
        if (IsAlignedForPack<max_pack_size, R, IN...>(r, in...))
        {
        	return LaunchKernel<max_pack_size, FactoryT, R, IN...>(factory, n, r, in..., stream);
        }
        else
        {
        	return LaunchKernel<1, FactoryT, R, IN...>(factory, n, r, in..., stream);
        }
    }
};
```

 `GenericLauncher` 中只有一个静态成员方法 `Launch`，通过 `PackSize` 函数获取 `max_pack_size`（注意这个 `max_pack_size` 的值在编译期就已经确定），然后调用 `IsAlignedForPack` 判断 `max_pack_size` 是否满足内存对其要求。如果满足就使用这个 `max_pack_size` 来 pack 数据，否则 `pack_size` 取 1，也是就不使用向量化的数据了。  

 `IsAlignedForPack` 函数和 `PackSize` 函数一样都是可变参数模板函数，利用重载和递归的机制分别计算每一个输入参数的内存地址是否能整除 pack 后的内存大小 `sizeof(Pack<T, pack_size>)` ，只有全部输入参数都能满足内存对齐要求，最终结果才为真。简单来说就是所有参数 pack 后内存都能对齐，才执行 pack，只要有一个参数不符合，大家都不 pack 了。

```cuda
template <size_t pack_size>
bool IsAlignedForPack()
{
	return true;
}

template <size_t pack_size, typename T, typename... Args>
/**
* @brief 判断类型 T 在 pack 后是否内存对齐
* 
* @param ptr 
* @param others 
* @return true 
* @return false 
*/
bool IsAlignedForPack(const T *ptr, const Args *...others)
{
    // 判断ptr地址是否能够整除 pack 后的 T 的大小，reinterpret_cast<uintptr_t>(ptr)将指针转换为一个 unsigned __int64
    return reinterpret_cast<uintptr_t>(ptr) % sizeof(Pack<T, pack_size>) == 0 && IsAlignedForPack<pack_size, Args...>(others...);
}
```

#### 2.4.2 LaunchKernel

`LaunchKernel` 用来启动核函数，在函数体内部计算了 pack 数目、tail 数目和 numBlocks，并将 pack 数据和 tail 数据分别传入核函数 `ApplyGeneric`，逻辑比较简单，详细请看笔者的代码注释。

```cuda
template <size_t pack_size, typename FactoryT, typename R, typename... IN>
/**
* @brief 启动核函数
* 
* @param factory 
* @param n 元素个数
* @param r 
* @param in 
* @param stream 
* @return cudaError_t 
*/
cudaError_t LaunchKernel(FactoryT factory, int64_t n, R *r, const IN *...in, cudaStream_t stream)
{
    const int64_t n_pack = n / pack_size; // 根据元素个数和pack_size，计算pack数目，比如1026 / 4 = 256。
    const int64_t tail_offset = n_pack * pack_size; // 如果存在不被整除的情况，我们计算使用pack的偏移量：256*4
    const int64_t n_tail = n - tail_offset; // 元素数目-偏移量 = 剩下的元素个数-> 1026-1024 = 2
    int num_blocks;
    {
    	cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
        if (err != cudaSuccess)
        {
        	return err;
    	}
	}
	ApplyGeneric<pack_size, FactoryT, R, IN...><<<num_blocks, kBlockSize, 0, stream>>>(
factory, n_pack, reinterpret_cast<Packed<R, pack_size> *>(r),
(reinterpret_cast<const Packed<IN, pack_size> *>(in))..., n_tail, r + tail_offset,
(in + tail_offset)...);
	return cudaPeekAtLastError();
}
```

#### 2.4.3 ApplyGeneric

`ApplyGeneric` 这个 CUDA Kernel 中所做的主要工作是：
- 根据参数创建一个 `functor`，注意这里要求 `factory` 对象重载了 `()` 运算符。
- 进入循环，针对打包（pack）后的数据，调用 `ApplyPack` 函数，每调用一次 `ApplyPack`，就处理一个 `pack` 后的数据
- 当元素个数不能被 `pack_size` 整除时（即 `n_tail > 0`），需要让前几个线程直接调用 `functor` 单独处理下尾部剩余元素

```cuda
/**
* @brief 
*    __launch_bounds__(kBlockSize):为编译器指定每个线程块的最大线程数，优化寄存器使用
* @tparam pack_size 
* @tparam FactoryT 
* @tparam R 
* @tparam IN 
*/
template <int pack_size, typename FactoryT, typename R, typename... IN>
/**
* @brief Construct a new Apply Generic object
* 
* @param factory 
* @param n_pack num of packed data
* @param pack_r 
* @param pack_in 
* @param n_tail num of elements which are unpacked
* @param tail_r 
* @param tail_in 
*/
__global__ void __launch_bounds__(kBlockSize)
ApplyGeneric(FactoryT factory, int64_t n_pack, Packed<R, pack_size> *pack_r,
const Packed<IN, pack_size> *...pack_in, int64_t n_tail, R *tail_r,
const IN *...tail_in)
{
    auto functor = factory();
    const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
    for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x)
    {
    	pack_r[i] = ApplyPack<pack_size, decltype(functor), R, IN...>(functor, (pack_in[i])...);
    }
    if (global_tid < n_tail)
    {
    	tail_r[global_tid] = functor((tail_in[global_tid])...);
    }
}
```

另外 `ApplyGeneric` 中还使用了 `Grid-Stride Loops` ，在 Kernel 中每一个线程可能处理了多个 pack 后的数据，即：`for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x)` 。 这个步长 `blockDim.x * gridDim.x` 表示的是 CUDA 线程网格中的线程总数，假设线程网格中有 1280 个线程，线程 0 将计算元素 0、1280、2560 等。什么情况下会处理多个 pack 后的数据？回顾 `numBlocks` 的计算方法可以发现，当元素个数较多， `numBlocks = wave 数目 * GPU 一次可以调度 SM 数量 * 每个 SM 最大 block 数` 时，`n_pack` 远大于线程总数，此时一个线程将处理多个 pack 后的数据。关于 `Grid-Stride Loops` 的细节，有兴趣的读者可以参考：[CUDA编程入门之 Grid-Stride Loops](https://zhuanlan.zhihu.com/p/571320529 )

#### 2.4.4 ApplyPack

`ApplyPack` 函数对一个 pack 内的元素做了循环处理，对 `elem` 数组中的每个元素调用 `functor`，逻辑比较简单不再赘述。

```cuda
template <int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == false || pack_size % 2 != 0,
Packed<R, pack_size>>::type
ApplyPack(const FunctorT &functor, const Packed<IN, pack_size>... in)
{
	Packed<R, pack_size> ret;
    #pragma unroll
    for (int j = 0; j < pack_size; ++j)
    {
    	ret.elem[j] = functor((in.elem[j])...);
    }
    return ret;
}
```

## 3 调用方式
OneFlow 在 elementwise.cuh 文件中分别针对一元，二元，三元运算的 Element-Wise 操作提供了模板函数。在包含头文件之后我们可以使用 `cuda::elementwise::Unary/Binary/Ternary` 这几个模板函数来针对我们自己定义的 Element-Wise 操作进行计算。注意，这里说的一元、二元、三元指的是这个 Element-Wise 操作有几个输入 Tensor，当然如果有更多输入参数的需求，可以自行根据模板提供的接口和样例编写代码即可。下面以 `sigmoid` 操作为例，介绍一下调用方法，根据 `sigmoid` 计算公式可知，这是一个一元变换，就是把实数域的 `x` 映射到 `(0,1)` 之间，常用于神经网络的激活函数或者将数值映射成概率。  
$$
Sigmoid(x) = \frac{1}{1+e^{-x}} 
$$
根据 `sigmoid` 计算公式，我们可以定义一个模板类 `SigmoidFunctor`，根据 `ApplyGeneric` 函数中 `auto functor = factory();` 可知， `SigmoidFunctor` 要重载小括号运算符，并在重载函数中实现 `sigmoid` 计算逻辑，具体实现如下：  
```cuda
template<typename T>
struct SigmoidFunctor
{
    __device__ __host__ __forceinline__ T operator()(T x) const
    {
        return T(1.0) / (T(1.0) + expf(-x));
    }
};
```
注意重载运算符函数要加 `__device__` 修饰，因为该函数是在设备端调用的。然后我们就可以使用 `cuda::elementwise::Unary` 函数来完成 `sigmoid` 计算了，实例代码如下：
```cuda
cudaStream_t stream;
CHECK(cudaStreamCreate(&stream));
CHECK(Unary(SigmoidFunctor<float>(), N, d_out, d_in, stream));
CHECK(cudaStreamSynchronize(stream));
CHECK(cudaStreamDestroy(stream));
```
其中 `N` 是 element 个数；`d_out` 是设备端的输出；`d_in` 是设备端的输入；`stream` 是当前流，这里也提供了一个 cuda 流的创建销毁示例。

## 4 小结

至此，OneFlow 的 CUDA Elementwise 模板的源码解读完毕，最后再来总结下这套模板的优势：

- 性能够高，通过构造向量化数据充分利用了 GPU 带宽，这也是这套高性能模板的主要优化内容。
- 开发效率高，利用简单工厂模式对外提供了完善的接口，开发人员可以不用过分关注 CUDA 逻辑及相关优化手段，也无需更改模板代码，只需要编写计算逻辑即可。
- 可扩展性强，目前这套模板支持了一元，二元，三元操作。需要支持更多输入时，只需要仿照编写对应的工厂即可。
- 模板源码中使用了不少 C++ 新特性，尤其泛型编程用的出神入化，值得学习借鉴。