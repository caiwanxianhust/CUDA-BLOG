# 【CUDA编程】剖析一下cute库中基于Ampere架构的GEMM示例

**写在前面**：笔者前面有几篇文章介绍了一些异步拷贝和矩阵乘法相关的 PTX 指令，同时也给出了这些指令在 cute 库中对应的封装。为了更深入地了解并使用 cute 库，本文将以 cute 库中基于Ampere架构的 GEMM 示例代码为样例，逐步介绍这些 GEMM 的优略策略如何通过 cute 库来落地实现。在此笔者有一句忠告，如果你想快速掌握 cutlass、cute 库，至少要对相关的 PTX 指令有所了解，要不然即便 cuda 学得再好，学起这俩库来依然举步维艰。

## 1 任务背景

本文将以 cute 库中基于Ampere架构的 GEMM 示例代码为样例，cutlass 版本为 3.9.0，具体代码路径如下，读者可以去 github 上自行下载。

> cutlass/examples/cute/tutorial/sgemm_sm80.cu

该源文件中提供了一套基于 Ampere 架构的 GEMM 高性能计算方案，包括 TN GEMM、NT GEMM 和 TN HGEMM 三种，本文将以 TN HGEMM 场景为例，对源码进行剖析。

所谓 TN HGEMM，顾名思义，TN 说的就是矩阵乘加操作的 A\B 矩阵是否转置，T 代表转置（Transpose），N 代表不转置，也就是说矩阵 A 是以转置形式输入，矩阵 B 以非转置形式输入。对于矩阵内存排序而言，cute 中默认为 LayoutLeft（即列主序），那么矩阵 A 的 layout 可以表示为 `(m, k):(lda, 1)`，矩阵 B 的 layout 则表示为 `(n, k):(ldb, 1)`。代码中 `mnk` 分别取值为 `5120`、`5120` 和 `4096`，换句话说，对于不熟悉 cute layout 的读者可以认为矩阵 A、B 的输入形状分别为行主序下的 `[5120, 4096]`、`[5120, 4096]`，这也是我们做 Attention 等计算任务时的常态。

HGEMM 表示当前 GEMM 的数据类型为 `half`，自然而然地，我们应该想到 PTX 中对应该任务的一个指令：

> mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16

以上指令中，`m16n8k16` 对应 mma 指令的矩阵形状，当然源代码所用的指令不一定是这个形状；`row.col` 正好对应 TN，即 `A[m, k]` 矩阵为行主序，`B[k, n]` 为列主序；`f16.f16.f16.f16` 表示输入输出矩阵的元素都是 `half` 类型。

话不多说，我们来看一下源代码的调用链：

```cp
main -> gemm -> gemm_tn -> gemm_device(Kernel)
```

`main` 函数中对本次计算任务的一些基础参数进行初始化，如 `mnk`、`transA`、`transB`、数据类型以及矩阵元素填充等，最后调用 `gemm` 函数。

`gemm` 函数中针对 `transA`、`transB` 参数进行任务分流，如果是 `TN` 乘则调用 `gemm_tn` 函数。

`gemm_tn` 函数中基于 cute 库的 layout、copy、mma 等抽象对 gemm 任务中的一些变量进行定义，并调用 Kernel。

`gemm_device` 函数中包含了 HGEMM 的核心逻辑，如矩阵分块、数据拷贝、流水线、mma 等。

总的来说，绝大部分计算逻辑都集中在 `gemm_tn` 函数和 `gemm_device` 函数中，下面我们将逐行对这两个函数中的代码进行剖析，在这之前，假设各位读者都了解 cute 中的一些基本抽象概念，在这里推荐各位阅读知乎大佬 reed 的相关文章。

## 2 `gemm_tn` 函数

### 2.1 Layout 定义

#### 2.1.1 Layout 参数设置

首先定义了本次 HGEMM 任务的矩阵 Layout 相关的参数，如描述矩阵形状参数 `mnk` 的 `prob_shape` 变量，以及描述矩阵 stride 的 `dA`、`dB`、`dC` 变量。要注意的是，三个矩阵形状分别表示为 `[m, k]`、`[n, k]`，`[m, n]`，所以在 TN 乘的背景下内存排序分别为行主序、行主序、列主序，具体如下：

```cpp
// Define shapes (dynamic)
auto M = int(m);
auto N = int(n);
auto K = int(k);
auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

// Define TN strides (mixed)
auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)
```

然后进行分片矩阵的 Layout 定义，指定 thread block 层级分片矩阵的形状参数分别为 `128`、`128`、`64`，并定义流水线缓冲区参数 `bP` 为 `3`。

```cpp
// Define CTA tile sizes (static)
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<64>{};
auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
auto bP = Int<3>{};  // Pipeline
```

#### 2.1.2 swizzle 机制

笔者在前一篇文章（[【CUDA编程】Flash Attention CUDA 实现思路（第二版）](https://mp.weixin.qq.com/s/4OtFXAcngZhLCSfWNiPrbg)）中有介绍过，当计算 mma 时通常会从 shared memory load 矩阵分片到寄存器中，这中间通常会有 bank conflict，为了避免 bank conflict，需要打乱矩阵元素在 shared memory 中的排布，一般有两种方式：padding 和 swizzle。由于 padding 会增加 shared memory 需求量，所以通常会选择 swizzle 策略，这里源代码中基于也选择了 swizzle。由于元素类型为 `half`，`bK` 为 `64`，所以 swizzle 参数分别为 `B=M=S=3`。这里对参数选择或 bank conflict 有疑虑的读者不妨阅读一下笔者的上一篇文章的第 3 节。

```cpp
 auto swizzle_atom = composition(Swizzle<3, 3, 3>{},
                                    Layout<Shape<_8, Shape<_8, _8>>,
                                           Stride<_8, Stride<_1, _64>>>{});

auto sA_layout = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
auto sB_layout = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
auto sC_layout = make_layout(make_shape(bM, bN));
```

源码中首先定义了一个 swizzle 原子的 Layout，我们不妨分成两部分来看，swizzle 前的 Layout 为 `(8, (8, 8)):(8, 1, 64)`，这个 Layout 打印出来如下所示：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5pegfwH67hFksnlLcRrEpicUiaUZicVaqVG1gXTbXibfBFw5CoUCXR23QUylae5m9HSjf025Qazx99JJQ/640?wx_fmt=png&amp;from=appmsg)

无论是从 global memory load 到 shared memory，还是从 shared memory load 到寄存器，都是按 `128bit` 一次 load，所以连续的 `8` 个元素作为一个整体来进行 load，根据 Layout 可知（参考红色箭头），在一次 `m8n8` 的 LDSM 指令中，实际加载的元素是上述矩阵的一整行（`0-7`、`64-71`、`128-135`、···，参考蓝色矩形框），正好全部分属在第 `0-3` bank，存在 bank conflict。加入 Layout 后，`swizzle_atom` 内存排布变成如下所示：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5pegfwH67hFksnlLcRrEpicUWEuW91OtvDJ8R3XRneia1l6HdlOyVgZtsG4WWH0DRU9iaC5P4yYQhJ2g/640?wx_fmt=png&amp;from=appmsg)

为了更加直观地展示，我们不妨把连续地 `8` 个元素当作一个整体用一个方块表示，则 swizzle 前后示意图如下：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5pegfwH67hFksnlLcRrEpicUYIWUGzibSVF6CibjL0MY8liaHerDQEayxLMX8PJv6N6laVsDYTTZ8ia2gQ/640?wx_fmt=png&amp;from=appmsg)

图中 `0` 所在的方格（含连续的 `8` 个元素）属于 `0-3` bank，`1` 所在的方格（含连续的 `8` 个元素）属于 `4-7` bank，依次类推。可以发现，经过 swizzle 以后，之前连续加载的一整行元素（`m8n8` 矩阵）分属到了 `32` 个 bank，bank conflict 被消除。值得注意的是，swizzle 在 shared memory 的读写两个环节都需要应用，这样才能保证写入寄存器的矩阵与原本 global memory 中的矩阵分片相同，即**两次 swizzle 变换后，矩阵还是那个矩阵，只是加载过程中的 bank conflict 被通过 swizzle 变换给消除了**。

共享内存变量 `sA_layout` 与 `sB_layout` 通过 `tile_to_shape` 将 `swizzle_atom` 的内存布局应用到整个矩阵形状。

### 2.2 copy 方式

在 GEMM 计算任务中，矩阵元素先是从 global memory 移动到 shared memory，然后再从 shared memory 移动到寄存器，这中间存在两个 copy 过程，源码中分别定义了相应的 copy 对象。

#### 2.2.1 从 global memory 移动到 shared memory

先来看一下从 global memory 移动到 shared memory 的过程，源码中使用 `SM80_CP_ASYNC_CACHEALWAYS` 进行数据拷贝，该结构体封装的是一个 Ampere 架构引入的异步拷贝 PTX 指令，具体如下：

```cpp
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEALWAYS
{
    ...

    CUTE_HOST_DEVICE static void copy(TS const& gmem_src, TD& smem_dst)
    {
        ...
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
                         :: "r"(smem_int_ptr),
                         "l"(gmem_ptr),
                         "n"(sizeof(TS)));
        ...
    }
};
```

该指令是一条非阻塞指令，启动了一个异步拷贝操作，将数据从 global memory 直接拷贝到 shared memory 中，而不必经过寄存器，这在 Ampere 架构之前是无法做到的，一定程度上削减了寄存器使用量，除此之外该指令配合对应的 Async‑group 机制可以灵活控制数据的异步拷贝过程，为多级流水线策略提供基础。关于这个 cp.async 指令的用法，这里不再赘述，有兴趣的读者可以移步笔者的另一篇文章：[【CUDA编程】异步拷贝指令 cp.async 的介绍](https://mp.weixin.qq.com/s/nATI7NBRLOVPL0dCAKqrPg)，我们先来看一下源码：

```cpp
TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  
                                    Layout<Shape< _1,_8>>{});         

TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                  Layout<Shape<_16,_8>,Stride<_8,_1>>{},  
                                  Layout<Shape< _1,_8>>{});               
```

`Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>` 表示基于上述 cp.async 指令定义了一个拷贝原子操作，拷贝元素是 `half` 类型，一次拷贝 `128 bit`（`8` 个元素）。除了 `Copy_Atom` 以外，`make_tiled_copy` 函数还有两个参数：`ThrLayout` 和 `ValLayout`。`ThrLayout` 定义了 thread block 内线程的排布方式，`16` 行 `8` 列且行主序排布；`ValLayout` 定义了单个线程一次拷贝的元素数量，这个值与前面 `Copy_Atom` 中的参数 `uint128_t` 相关，设置为 `Layout<Shape< _1,_8>>{}`  表示一次拷贝形状为 `(1, 8)` 的一批元素，所以 `ValLayout` 形状也与矩阵在  global memory 中的内存排序有关，行主序的情况下一行元素是连续的，所以才能设置为 `(1, 8)`，如果设置为 `(8, 1)` 则会报错。结合 `ThrLayout` 和 `ValLayout` 整体来看，一个 thread block 一次完成了一个形如 `16 * 1` 行 `8 * 8` 列的矩阵分片的拷贝，示意图如下：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5o7BVh8boWDzH2qJfZMK8bTONg31GMaTonlfYVsPI4dqMYG4gIqOzvZt1sbwVPWwrjZDmmvibOWOoQ/640?wx_fmt=png&amp;from=appmsg)

关于参数设置，根据笔者的经验，在确定 `Copy_Atom` 之后，已经确定一次拷贝 `128 bit`（`8` 个元素），再结合原始矩阵行主序的背景，所以 `ValLayout` 设置为 `Layout<Shape<_1,_8>>{}`，然后为了方便沿列方向重复拷贝，所以要将 tile 的列数（一行元素的数量）与 `bK` 一致，这样的话每行则需要 `64 / 8 = 8` 个线程，如果 `block_size` 设置为 `128`，则每列将有 `128 / 8 = 16` 个线程，就得到了 `ThrLayout`。 

#### 2.2.2 从 shared memory 移动到寄存器

再来说从 shared memory 移动到寄存器的过程，这个操作的目的是接下来的 mma 计算，所以 `TiledCopy` 对象要通过 `make_tiled_copy_A\B` 函数生成，参数分别是 `Copy_Atom` 和 `TileMMA` 对象。

先来看一下 `TileMMA` 对象，通过 `make_tiled_mma` 函数生成，包括 `MMA_Atom`、`ThrLayout`、`Permutations` 三个参数对象。

```cpp
TiledMMA mmaC = make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{},
    Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
    Tile<_32,_32,_16>{});      // 32x32x16 Tiled MMA for LDSM
```

`MMA_Atom` 对象使用的是 `SM80_16x8x8_F16F16F16F16_TN`，这个结构体封装的是一个 Ampere 架构引入的 mma 指令，具体如下：

```cpp
struct SM80_16x8x8_F16F16F16F16_TN
{
	...
	CUTE_HOST_DEVICE static void
     fma(uint32_t      & d0, uint32_t      & d1,
         uint32_t const& a0, uint32_t const& a1,
         uint32_t const& b0,
         uint32_t const& c0, uint32_t const& c1)
  	{
...
		asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1},"
            "{%2, %3},"
            "{%4},"
            "{%5, %6};\n"
            : "=r"(d0), "=r"(d1)
            :  "r"(a0),  "r"(a1),
            "r"(b0),
            "r"(c0),  "r"(c1));
...
  	}
};
```

该指令通过一个 warp 内的所有线程共同完成形状为 `m16n8k8` 的半精度矩阵乘加，矩阵的元素存储在每个线程的寄存器中。由于是一个 warp 级的集体操作，因此 `ThrLayout` 对象表述的也是 warp 的排布而非 thread，前面说过 `block_size` 设置为 `128`，包含 4 个 warp，源码中采用 `(2, 2)` 的排布形式。第 3 个参数从 cutlass 3.4 以后由 `ValLayoutMNK` 换成了 `Permutations`，新的参数功能更加强大，笔者对这个新的参数了解也不是很深，建议有兴趣的读者研究下面这个[issue](https://github.com/NVIDIA/cutlass/discussions/1345)。源码中也只是对这个参数的简单应用，通过设置为 `Tile<_32,_32,_16>`，在 block 内线程数量不变的前提下，通过寄存器重复将 `MMA_Atom` 计算的矩阵形状由原本的 `m16n8k8` 拓展到 `m32n32k16`。

> https://github.com/NVIDIA/cutlass/discussions/1345

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rMU9H4JBEdlToFVVbATfyd3iavmx7rfPdNvLTicqNCwfGLKFNIIImC7a5rzFfMM5icIYbg9zcxKFsdA/640?wx_fmt=png&amp;from=appmsg)

`TiledMMA` 对象规定了一个 mma 计算过程中矩阵元素与各线程持有的寄存器对应关系，在进行 mma 计算之前，需要将矩阵元素从 shared memory 拷贝到寄存器中，并且按照 `TiledMMA` 对象描述的那样去填充 A、B 矩阵对应的各线程的寄存器。源码中使用 `SM75_U32x4_LDSM_N`，定义了两个 `Copy_Atom`。

```cpp
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;
```

`SM75_U32x4_LDSM_N` 结构体封装的是 Turing 架构引入的 PTX 指令，具体如下：

```cpp
struct SM75_U32x4_LDSM_N
{
   ...
   
   CUTE_HOST_DEVICE static void
   copy(uint128_t const& smem_src,
    uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
    {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
        asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
    }
};
```

 该指令从 shared memory 中一次加载 `4` 个 `m8n8` 矩阵到寄存器，也是一个 warp 级的集体操作，示意图如下：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5ouIrFVthDAC3KbZOZm43fqa0hmnZmfOIW9micH2ZxZFLKO07olorwZq5CUYTachGxqkHQP935kI7g/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1)

有了 `Copy_Atom` 和 `TileMMA` 对象后，调用 `make_tiled_copy_A\B` 函数就可以生成 `TiledCopy` 对象，源码中这个过程放在 Kernel 中进行，这里我们不妨提前打印出来看下：

`copyA` 的布局

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rMU9H4JBEdlToFVVbATfydd8yGmt3YdOLum5VHcCoxVyQeG9GpwC39QW0eia37pxA4lvgQHvicXWDA/640?wx_fmt=png&amp;from=appmsg)

`Tiled_A` 矩阵的形状是 `(32, 16)`，而 `s2r_atom_A` 一次可以拷贝 4 个 `m8n8` 矩阵（即形状为 `m16n16`），所以需要两个 warp，即 warp 0 和 warp 1 负责 `Tiled_A` 矩阵的拷贝，左图是源数据在 shared memory 中的布局以及 LDSM 指令负责拷贝的线程 ID，右图是各线程寄存器持有的矩阵元素布局。

`copyB` 布局

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rMU9H4JBEdlToFVVbATfydqPGLof9B8l5T3PTypkk3km4XMPXEZ0ehTiaWhnKSgATRwX2RibLzw5rA/640?wx_fmt=png&amp;from=appmsg)

类似地，`Tiled_B` 矩阵的形状是 `(32, 16)`，而 `s2r_atom_B` 一次可以拷贝 4 个 `m8n8` 矩阵（即形状为 `m16n16`），所以需要两个 warp，即 warp 0 和 warp 2 负责 `Tiled_B` 矩阵的拷贝，左图是源数据在 shared memory 中的布局以及 LDSM 指令负责拷贝的线程 ID，右图是各线程寄存器持有的矩阵元素布局。

### 2.3 启动 Kernel

由于 Kernel 中使用到了动态分配的 shared memory，所以需要提前计算好共享内存数量大小 `smem_size`，源码中封装了一个 `SharedStorage` 结构体，我们把源码都贴出来看下：

```cpp
template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage
{
  	cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  	cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

int smem_size = int(sizeof(SharedStorage<cute::half_t, cute::half_t, decltype(sA), decltype(sB)>));
```

从源码中大致可以看出，`smem_size` 等于 `sA` 和 `sB` 的 Layout 的 cosize 之和。cosize 表示布局函数的陪域（codomain）的大小，不一定等于范围（range）。它等同于 `A(size(A) - 1) + 1`，即布局根据 stride 计算的实际占用空间。这个值考虑了布局中可能的冗余或填充，因此可能大于布局的shape表示的维度。

举个例子，假设一维 tensor A 的 Layout 为 `5:3`，则 `size_v<LayoutA>` 就是元素个数 `5`，`cosize_v<LayoutA>` 则需要根据 A 的值域来看，把 A 列出来：

```cpp
0, 3, 6, 9, 12
```

`cosize_v<LayoutA>` 就是：从值域的最小值 `0` 到值域的最大值 `12`，算上中间被漏掉的数值，总共有 `13` 个整数，所以 `cosize_v<LayoutA>` 等于 `13`，公式 `A(size(A) - 1) + 1` 可以解释为 `LayoutA` 的最大值加 `1`。总的来说，cosize 值大于等于 size 值，当 Layout 值域连续时两者相等，当 Layout 值域不连续时，cosize 值大于 size 值。

`smem_size` 确定后，源码中调用了 cuda runtime api `cudaFuncSetAttribute` 设置了最大动态共享内存容量，同时针对核函数 `gemm_device` 将 Unified Cache 中 shared memory 的比例设置为 `100%`（从 Volta 架构开始，L1 cache、Texture Cache 和 shared memory 三者在物理上被统一成了一个数据缓存，把 shared memory 的比例设置为 `100%` ，则 L1 cache 和 Texture Cache 占比将为 `0`）。

```cpp
int smem_size = int(sizeof(SharedStorage<cute::half_t, cute::half_t, decltype(sA), decltype(sB)>));
dim3 dimBlock(size(mmaC));
dim3 dimGrid(size(ceil_div(M, bM)),
             size(ceil_div(N, bN)));

auto kernel_fptr = gemm_device<
    decltype(prob_shape), decltype(cta_tiler),
cute::half_t, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
cute::half_t, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
cute::half_t, decltype(dC), decltype(sC), decltype(mmaC),
decltype(alpha), decltype(beta)>;

// Set L1 to be SMEM only
cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>
    (prob_shape, cta_tiler,
     A, dA, sA, copyA, s2r_atom_A,
     B, dB, sB, copyB, s2r_atom_B,
     C, dC, sC, mmaC,
     alpha, beta);
}
```

## 3 gemm_device 函数

### 3.1 形状参数静态 assert

在 Kernel 的开头部分主要做一些针对各种形状参数的静态断言，用以校验前面 Layout 的设置是否正确且相互匹配，此过程在编译期进行，代码逻辑比较简单，读者自行阅读注释即可。

```cpp
// Preconditions
CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

static_assert(is_static<ASmemLayout>::value);
static_assert(is_static<BSmemLayout>::value);
static_assert(is_static<CSmemLayout>::value);

CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN
```

### 3.2 定义 global memory 和 shared memory 中的矩阵 Tensor

```cpp
// Represent the full tensors
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

// Get the appropriate blocks for this thread block
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

// Shared memory buffers
extern __shared__ char shared_memory[];
using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)
```

定义代表 A\B\C 3 个矩阵整体的 Tensor，从参数可以看出组成 Tensor 的两个要素：指针和布局 Layout（包括 Shape、Stride）。

使用 `local_tile` 函数定义 block 级别的 Tensor，`gA` 的形状是 `(BLK_M,BLK_K,k)`，其中 `k` 是矩阵分片沿 `K` 方向滑动的次数，`k = K/BLK_K`，`gB` 的形状是 `(BLK_N,BLK_K,k)`，`gC` 的形状是 `(BLK_M,BLK_N)`。

定义 shared memory 中的 A\B 矩阵 Tensor，源码中借助了 `SharedStorage` 类来定义，这个类的源码在 2.1.1 中已经介绍过了，包含 `A` 和 `B` 两个成员，分别占用 `sizeof(TA) * cosize_v<SmemLayoutA>` 和 `sizeof(TB) * cosize_v<SmemLayoutB>` 的内存空间，根据 `begin()` 函数返回的指针以及 Layout 定义 `sA` 和 `sB` Tensor。

### 3.3 g2s 拷贝对象的线程级划分

在 `gemm_tn` 中定义了 `TiledCopy` 对象，用于完成矩阵元素从 global memory 到 shared memory 的拷贝，在 Kernel 中将基于 `TiledCopy` 对象进行更进一步的任务划分，即通过 `get_slice` 函数，将拷贝任务划分到线程层级。`tAgA` 表示对 global memory 中的 Tensor `gA` 的线程级划分，`tAsA` 表示对 shared memory 中的 Tensor `sA` 的线程级划分，拷贝过程就是将 `tAgA` 中的元素移动到 `tAsA` 中。

```cpp
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);          // (CPY,CPY_M,CPY_K,k)
Tensor tAsA = thr_copy_a.partition_D(sA);          // (CPY,CPY_M,CPY_K,PIPE)

ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
Tensor tBgB = thr_copy_b.partition_S(gB);          // (CPY,CPY_N,CPY_K,k)
Tensor tBsB = thr_copy_b.partition_D(sB);          // (CPY,CPY_N,CPY_K,PIPE)
```

我们知道 `gA` 的形状为 `(BLK_M,BLK_K,k)`， 即 `(128,64,64)`，`k` 表示沿 `K` 方向拷贝 `64` 次，根据 2.2.1 中的 `TiledCopy` 规则：每个线程每次拷贝 `128 bit`，包含连续 `8` 个 `half` 元素，因此 `CPY` 的 `size` 为 `8`；一个 thread block 一次完成了一个形如 `16 * 1` 行 `8 * 8` 列的矩阵分片的拷贝，沿 `M` 方向 `128` 行需要拷贝 `8` 次，因此 `CPY_M` 的 `size` 为 `8`，沿 `K` 方向每次拷贝一整行，所以 `CPY_K` 的 `size` 为 `1`。打印出 `tAgA` 和 `tAsA` 的布局如下：

```cpp
tAgA : gmem_ptr[16b](0x505600000) o ((_8,_1),_8,_1,64):((_1,_0),65536,_0,_64)
tAsA : smem_ptr[16b](0x7f531d000000) o ((_8,_1),_8,_1,(_1,_3)):((_1,_0),_1024,_0,(_0,_8192))
```

同理，`tBgB` 和 `tBsB` 的布局如下：

```cpp
tBgB : gmem_ptr[16b](0x507e00000) o ((_8,_1),_8,_1,64):((_1,_0),65536,_0,_64)
tBsB : smem_ptr[16b](0x7f531d00c000) o ((_8,_1),_8,_1,(_1,_3)):((_1,_0),_1024,_0,(_0,_8192))
```

### 3.4 从 global memory 预加载（prefetch）到 shared memory 



#### 3.4.1 流水线基础

预加载（prefetch）的核心就是两项技术：流水线、异步拷贝指令。关于异步拷贝指令（即 cp.async 指令），笔者在之前的一篇文章有过介绍，这里就不再重复了，有兴趣的读者可以前往阅读。流水线与其说是一种技术，不如说是一种并行化提升效率的手段，顾名思义，相当于工厂车间中将一项复杂任务拆分为多个环节（如取材、生产、打包、装运等），每个阶段由不同的人负责，当第一批产品正在打包时，负责生产的人可以同时进行第二批产品的生产，依次类推各个环节可以并行工作，从而提升工作效率。

> [【CUDA编程】异步拷贝指令 cp.async 的介绍](https://mp.weixin.qq.com/s/nATI7NBRLOVPL0dCAKqrPg)

对应到计算机系统中，流水线技术是提升指令并行的有效手段，以经典的 RISC 流水线为例，一条指令的执行被分为 5 个阶段：取指（IF）、译码（ID）、执行（EX）、访存（MEM）、写回（WB）等，每个阶段需要的事件记为 $t_i$。若不使用流水线技术，则对于多个指令的处理则会采取如下方式进行：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5pUKsTlxEvBaL8FBibSNz0Zzb2giaYavcBHFWBgYJsAGw3CvrsJYe4XMaYQzXYVUZPKNNJE9KrEMYnw/640?wx_fmt=png&amp;from=appmsg)

从图上可以看出，若不使用流水线技术，则 3 个指令共 15 个阶段只能顺序执行，总共耗时 $3\sum t_i$，推广到 n 个指令则总耗时为 $n \sum t_i$。而如果采用流水线技术，则对于多个指令的处理则会采取如下方式进行：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5pUKsTlxEvBaL8FBibSNz0ZznibicrMeZ8cuBryFAibgeAoodSoZDzfzGWp1ASicgaS5Ph2BDRb5QHNSCw/640?wx_fmt=png&amp;from=appmsg)

从图上可以看出，在流水线结构下，第一条指令在取指完成后进入译码阶段，这时候第二条指令则可以进入取指阶段，后续的指令阶段也是依此类推重叠进行。采用流水线技术后 3 个指令执行耗时大大降低，具体降低了多少呢？

我们把单个执行中耗时最长的阶段所需的时间记为流水线周期 $T$，则流水线计算公式为：
$$
\sum t_i + (n - 1)T
$$
即，`1 条指令执行时间  + （指令条数  - 1）* 流水线周期`。

根据公式，使用流水线技术后，加速比为：
$$
\frac {n \sum t_i} {\sum t_i + (n - 1)T}
$$
从公式可以看出，流水线阶段数越多，即任务分的越细，理论上加速比越高；当指令数量（任务数）远大于阶段数时，加速比趋近于 $\frac {\sum t_i}{T}$，一般约等于阶段数 $k$。

#### 3.4.2 GEMM 流水线

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5pUKsTlxEvBaL8FBibSNz0Zz5gjnX9SXhDia8dibwmPZicwC3icgxxWTIj4xCqrpGiamQuYjzOIObZqEDYg/640?wx_fmt=png&amp;from=appmsg)

回顾一下基于矩阵分片的 GEMM 计算过程，通过在 A、B 矩阵的 K 轴上循环读取形如 `(BM, BK)`、`(BK, BN)` 的 Tile，二者相乘再累加得到 C 矩阵中的 `(BM, BN)` Tile 的结果。基于流水线思想，我们不妨把这个过程也划分为多个阶段：

- A、B Tile 从 global memory 加载到 shared memory（LDGSTS = LoaD Global STore Shared memory）
- A、B Tile 从 shared memory 加载到寄存器（LDSM = LoaD Shared Matrix）
- 矩阵乘加运算（MMA = Matrix Multiply Accumulate）

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5pUKsTlxEvBaL8FBibSNz0Zz3MeTUexGqDKNuojY6W7HvS7iagYBgRKoZiacYrHS0q5LmWysiaic817JmQ/640?wx_fmt=png&amp;from=appmsg)

通过将 Tile 的计算划分为 3 个阶段，应用流水线思想，就可以进一步提升 GEMM 效率。LDGSTS 的目的地址在 shared memory，LDSM 的源地址也在 shared memory，要实现这两个阶段并行，就需要在 shared memory 中设计多缓冲区，源代码中有 3 个缓冲区。

```cpp
// 缓冲区（或 stage）数量
auto K_PIPE_MAX = size<3>(tAsA);

// Total count of tiles
int k_tile_count = size<3>(tAgA);
// Current tile index in gmem to read from
int k_tile_next = 0;

// Start async loads for all pipes but the last
CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }
```

从代码中来看，`K_PIPE_MAX` 是 `3`，在 for 循环中提交了两次拷贝操作，将 A、B 矩阵的前两个 tile 分别加载到 shared memroy 的前两个缓冲区，如下图所示。由于 `TiledCopy` 对象是一个异步拷贝对象，所以调用 `copy` 函数后，再调用 `cp_async_fence()` 函数提交本次拷贝操作，该函数封装的是一个 `cp.async.commit_group` 指令，该指令通过 cp.async-group 机制提交了一组异步操作。对于先后提交的多个拷贝操作，其完成顺序取决于提交顺序，也就是说，第一个 tile 一定比第二个 tile 先完成异步拷贝。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5o90AfkHYPnOG38SafTwYEsNAXFv7ziaKAmRdpphwCjIANZu0QWAJckzBBnyQibrSHW1tJnDQD3OHFg/640?wx_fmt=png&amp;from=appmsg)

### 3.5 mma 对象的线程级划分

在 `gemm_tn` 中定义了 `TiledMMA` 对象，用于完成矩阵乘加操作，在 Kernel 中将基于 `TiledMMA` 对象进行更进一步的任务划分，即通过 `get_slice` 函数，划分到线程层级。`tCgC` 表示按照 `thr_mma` 中 C 矩阵的布局对 global memory 中的 Tensor `gC` 的线程级划分，`tCrA` 是根据 shared memory 中的 Tensor `sA` 的形状结合 `thr_mma` 中 A 矩阵的布局新定义的一个 Tensor，这个 Tensor 是线程级的存放在寄存器中，`tCrB` 同理。`tCrC` 是根据 `tCgC` 的形状新定义的一个存放在寄存器中的线程级 Tensor，表示参与 mma 操作的矩阵 C。下面我们分别来看一下这些 Tensor 的布局。

```cpp
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC);                           // (MMA,MMA_M,MMA_N)

// Allocate registers for pipelining
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));           // (MMA,MMA_M,MMA_K)
Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));           // (MMA,MMA_N,MMA_K)
// Allocate the accumulators -- same size as the projected data
Tensor tCrC = thr_mma.make_fragment_C(tCgC);                     // (MMA,MMA_M,MMA_N)

CUTE_STATIC_ASSERT_V((  shape(tCrC) == take<0,3>(shape(tCgC)))); // (MMA,MMA_M,MMA_N)
CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));          // MMA_M
CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));          // MMA_N

// Clear the accumulators，mma 前将 accumulators 置零
clear(tCrC);
```

我们知道 `gC` 的形状为 `(BLK_M,BLK_N)`， 即 `(128,128)`，根据 2.2.2 中的 `TiledMMA` 中 C 矩阵的布局对 global memory 中的 Tensor `gC` 的线程级划分，得到 `tCgC` 的布局 `(MMA,MMA_M,MMA_N)`。其中 `MMA` 是在 `MMA_Atom` 层面的布局，因此只与 `MMA_Atom` 有关，源码中 `MMA_Atom` 基于 `SM80_16x8x8_F16F16F16F16_TN` 实现，根据 `SM80_16x8x8_F16F16F16F16_TN` 的 PTX 指令（布局如下图）可以知道，对于矩阵 C，每个线程持有的矩阵元素布局为 `(2, 2)`，即 `tCgC` 的布局 `(MMA,MMA_M,MMA_N)` 中 `MMA` 值为 `(2, 2)`。同理，`tCrA` 的布局 `(MMA,MMA_M,MMA_K)` 中 `MMA` 值为 `(2, 2)`，`tCrB` 的布局 `(MMA,MMA_N,MMA_K)` 中 `MMA` 值为 `2`，`tCrC` 的布局 `(MMA,MMA_M,MMA_N)` 中 `MMA` 值为 `(2, 2)`。 **总的来说，布局中 `MMA` 的值取决于具体的 `MMA_Atom` 指令。**

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rgfrpibdmAibEicKd5RKHp6fUp9mpDwTDAtah3331d8zQlc0Z2KaNN75g9p8bFAB5ccZMqKKw1YPvHQ/640?wx_fmt=png&amp;from=appmsg)

下面我们再来看 `MMA_M`、`MMA_N`、`MMA_K`，这三个值描述的是站在 thread block 层面，基于 `MMA_Atom` 完成  `BMXBNXBK` 乘法的所需要的各个方向的重复次数。已知 `(BLK_M, BLK_N, BLK_K)` 分别为 `(128,128,64)`，`MMA_Atom` 是 `(16,8,8)`，这是一个 warp 的计算任务，一个 thread block 有 `4` 个 warp，`(2, 2)` 排列，所以不考虑重复计算的情况下，一个 thread block 在一次 `MMA_Atom` 计算时可以完成 `(32, 16, 8)` 的矩阵乘加操作，如下图所示。那么在 M\N\K 三个方向分别需要重复的次数 `MMA_M`、`MMA_N`、`MMA_K` 就是 `128/32=4`、`128/16=8`、`64/8=8`。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rgfrpibdmAibEicKd5RKHp6fUUpbw0nauLdup9q166g0DU2PAiao09cuVcWNIM1CibqmVpylSXVdOhPtg/640?wx_fmt=png&amp;from=appmsg)

```cpp
 mC : gmem_ptr[16b](0x50a600000) o (5120,5120):(_1,5120)
 gC : gmem_ptr[16b](0x50a600000) o (_128,_128):(_1,5120)

tCgC : gmem_ptr[16b](0x50a600000) o ((_2,_2),_4,(_2,_4)):((5120,_8),_32,(81920,163840))
tCrA : ptr[16b](0x7f7adafff9f0) o ((_2,_2),_4,(_2,_2,_2)):((_1,_2),_4,(_16,_32,_64))
tCrB : ptr[16b](0x7f7adafffaf0) o (_2,_8,(_2,_2,_2)):(_1,_2,(_16,_32,_64))
tCrC : ptr[16b](0x7f7adafffbf0) o ((_2,_2),_4,(_2,_4)):((_1,_2),_4,(_16,_32))
```

### 3.6 s2r Copy Atom retiling

前面 3.3 节中定义了 g2s 拷贝对象的线程级划分，数据拷贝到 shared memory 后，在进行 mma 计算之前，还需要将数据拷贝到寄存器中，因此还需要根据 MMA 和 s2r_Copy_Atom 对 Tensor 进行线程级划分。这里使用一个 `make_tiled_copy_A\B` 函数创建 TiledCopy 对象，然后基于 `TiledCopy` 对象进行更进一步的任务划分，即通过 `get_slice` 函数，将拷贝任务划分到线程层级。这里的 TiledCopy 对象对应的布局示意图，就是2.2 节中的 copyA 和 copyB 的示意图。

`tXsA` 表示对 shared memory 中的 Tensor `sA` 的线程级划分，`tXrA` 是对 `tCrA` 调用 `retile_D` 后生成，定义数据从共享内存到寄存器的目标排列规则（如分块大小、维度顺序），指导硬件执行ldmatrix等加载指令，本质上只是重新组织一下线程布局，满足 s2r copy 的操作。拷贝过程就是将 `tXsA` 中的元素移动到 `tCrA` 中。

```cpp
  //
  // Copy Atom retiling
  //

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)
```

下面我们来看一下，`CPY`、`MMA_M`、`MMA_N`、`MMA_K`、`PIPE` 这几个值的含义。`CPY` 由 s2r_Copy_Atom 决定，在 2.2 节中有介绍，s2r_Copy_Atom 使用的是 `SM75_U32x4_LDSM_N`，与之对应的PTX 指令为 `ldmatrix.sync.aligned.x4.m8n8.shared.b16`，这个指令单个线程持有 `8` 个元素，所以 `CPY` 是 `8`。`MMA_M`、`MMA_N`、`MMA_K` 分别是站在 thread block 维度，在 M\N\K 三个方向的重复次数，已知 TiledCopy 的形状为 `Tile<_32,_32,_16>`，而 `(BLK_M, BLK_N, BLK_K)` 分别为 `(128,128,64)`，所以 `MMA_M`、`MMA_N`、`MMA_K` 都是 `4`。

```cpp
tXsA : smem_ptr[16b](0x7f7adc000000) o ((_8,_1),_4,(_2,_2),(_1,_3)):((_1,_0),_2048,(144,288),(_0,_8192))
tXrA : ptr[16b](0x7f7adafff9f0) o (((_4,_2),_1),_4,_4):(((_1,_16),_0),_4,_32)
tXsB : smem_ptr[16b](0x7f7adc00c000) o ((_8,_1),_4,(_2,_2),(_1,_3)):((_1,_0),_2048,(144,288),(_0,_8192))
tXrB : ptr[16b](0x7f7adafffaf0) o (((_4,_2),_1),_4,_4):(((_1,_16),_0),_4,_32)
```



