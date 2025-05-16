# 【CUDA编程】异步拷贝指令 cp.async 的介绍

**写在前面**：笔者前段时间学习 cuda 的 cute 库，被各种高度抽象的概念蹂躏的死去活来。诚然 cute 库的提出，给广大 GPU 并行计算开发人员提供了一种新的编程方式，其中 layout、tensor、copy、mma 等概念的引入，一定程度上让开发者免于复杂场景的索引计算、计算任务规划等繁琐工作，但是由于 cute 的官方文档过于简略，且这个赛道过于小众，使得入坑难度直线上升。也正是由于官方文档的缺失，使得笔者只好进一步研习源码，发现 cute 源码中的很多功能实现直接基于 PTX 指令，说实话笔者之前对这一块内容不甚了解，通常基于 CUDA C++ 就已经足以完成功能实现，很少涉及汇编层级的底层指令，正好借着这个机会学习一下。本文要介绍的是跟 cute 库的 copy 抽象相关的 异步拷贝指令 cp.async，将结合 PTX 文档对其功能和使用方法进行介绍。

## 1 异步拷贝的概念

提到异步拷贝，对于 CUDA Runtime 比较了解的读者可能首先想到的是 `cudaMemcpyAsync` 函数，这个 API 是基于流有序内存分配的概念，用于在设备内存中异步地拷贝数据，避免阻塞当前线程。通常是结合 CUDA 流用来将数据从主机内存拷贝到设备内存，或者从设备内存拷贝到主机内存。这是一个主机端的函数，不是本文要介绍的内容，本文要介绍的是在设备端异步拷贝的场景，即从全局内存（global memory）拷贝数据到共享内存（shared memory）的场景。

为什么需要异步拷贝呢？因为通常在大规模数据并行计算中，我们的计算任务规划通常是结合流水线机制的，通俗地说，会使用数据预取（或者说多缓冲、muti-stage 等等）等策略，在当前阶段加载下一阶段计算所需的数据，同时执行当前阶段的计算，重叠数据加载与计算操作，进一步提升性能。典型的双缓冲结合异步拷贝的场景（伪代码）如下：

```cpp
for (int stage = 0; stage < num_stages; ++stage) {
    // 阶段1: 异步加载下一批数据到共享内存
    cp_async_copy(global_ptr + stage*block_size, shared_buf[stage % 2]);
    cp_async_fence();

    // 阶段2: 计算当前批数据（共享内存中的前一组）
    if (stage > 0) {
        compute(shared_buf[(stage-1) % 2]);
    }

    // 阶段3: 等待当前批数据加载完成
    cp_async_wait<0>();
    __syncthreads();
}
```

上面的伪代码中，将计算任务分为多个 stage，在当前 stage 异步加载下一批计算所需的数据到共享内存，通过 `cp_async_fence()` 提交异步加载操作，然后计算当前批数据，注意在计算当前批数据的时候下一批数据的加载可能尚未完成，也就是说加载结果当前线程尚不能获取，然后通过 `cp_async_wait<0>()` 等待异步操作加载完成，此时加载结果对当前线程已经可见，随后插入一个同步 `__syncthreads()` ，在同步点以后（即下一个 stage）其他线程也对当前线程异步拷贝的数据可见。

## 2 Ampere 架构上的异步拷贝指令 cp.async

英伟达在 2020 年 5 月发布的 Ampere GPU 架构中引入了新的异步拷贝（Async Copy）指令，可将数据直接从全局内存异步加载到 SM 共享内存中。在 Ampere 架构之前从全局内存加载数据到共享内存必须要从 global memory 经过 L2 缓存到达 L1 缓存然后到达寄存器，然后再从寄存器到达 shared memory。新的异步拷贝指令一方面可以有选择地通过绕开 L1 缓存，避免数据在寄存器文件中的往返传输，节省了 SM 内部带宽，并且还避免了为正在传输的数据分配寄存器文件存储空间，对于大量的、连续的、从全局内存到共享内存的拷贝操作，使用异步拷贝可明显提高内存拷贝性能。

除了异步拷贝指令 cp.async 以外同时还提供了对应地拷贝完成机制，后面会进行介绍。新的异步拷贝指令 cp.async 语法如下：

```ptx
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
						[dst], [src], cp-size{, src-size}{, cache-policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
						[dst], [src], 16{, src-size}{, cache-policy} ;
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
						[dst], [src], cp-size{, ignore-src}{, cache-policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
						[dst], [src], 16{, ignore-src}{, cache-policy} ;
						
.level::cache_hint = { .L2::cache_hint }
.level::prefetch_size = { .L2::64B, .L2::128B, .L2::256B }
cp-size = { 4, 8, 16 }
```

cp.async 是一条非阻塞指令，启动了一个异步拷贝操作，将源地址操作数 src 指定的位置的数据拷贝到目标地址操作数 dst 指定的位置。操作数 src 是全局内存空间中的地址，dst 是共享内存空间地地址。

先来逐步看一下指令的限制符：

- `cp.async` 表示 cp 指令将异步启动内存拷贝操作，并且在指令发出后将控制权返回给当前线程，不会阻塞到拷贝完成。执行线程可以使用 async-group 对应的完成机制或者基于 mbarrier 的完成机制来等待异步拷贝操作完成，除此之外没有提供其他机制保证拷贝操作执行完成。
- `.cg` 即 cache global，表示仅在全局级别的 L2 缓存中缓存数据，绕过 L1 缓存（SM 级）；`.ca` 即 cache all，表示在所有级别的缓存中缓存数据，包括 L1 缓存。该限定符设置的作用仅与拷贝性能相关，不影响拷贝结果。`.cg` 与 `.ca` 分别代表下面第 2、3 幅图中的场景。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5oMjROOibyg6NWXJNLnTZCC0EmgVqNaZ5FYjKiazj5JqJicSWr7dtNMCjdOFQSE9NRrLSbTAiaJiaTpOcA/640?wx_fmt=png&amp;from=appmsg)

在 cute 库中，针对 `.cg` 与 `.ca` 分别封装了几个 copy Operation 对象，比如以 caching at global level 为代表的 `SM80_CP_ASYNC_CACHEGLOBAL` 和以 caching at all levels 为代表的 `SM80_CP_ASYNC_CACHEALWAYS`，其源码如下（重点关注 copy 方法）：

```cpp
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEGLOBAL
{
    ...

    CUTE_HOST_DEVICE static void copy(TS const& gmem_src, TD& smem_dst)
    {
       ...
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                         :: "r"(smem_int_ptr),
                         "l"(gmem_ptr),
                         "n"(sizeof(TS)));
        ...
     }
};

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

根据缓存策略可知，当数据重用性较低时，比如仅一次性访问的临时数据（流式访问），此时应当使用 `.cg` ，绕过 L1 缓存直接加载，减少对 L1 的无效占用；而当数据重用性较高时，比如需频繁加载的模型权重矩阵，此时应当使用 `.ca`，通过 L1 缓存一定程度上持久化访问，降低延迟。

- `.shared` 表示拷贝的目标地址在共享内存空间中，这里如果指定为 `shared{::cta}`，则目标地址必须在集群内执行 CTA 的共享内存中，否则行为未定义，这里是针对线程块集群和分布式共享内存的场景，本文不做讨论。
- `.global` 表示拷贝的源地址在全局内存空间中。
- `.level::cache_hint` 仅支持 `.global` 状态空间和地址指向 `.global` 状态空间的通用寻址，从指令语法可以看出只有一个枚举值 `.L2::cache_hint`。
- `.level::prefetch_size` 限定符是一个提示限定符，只影响性能，不影响拷贝结果。用于将指定大小的额外数据预取到相应的缓存，子限定符 `prefetch_size` 可以设置为 64B 、 128B 或 256B，从而允许预取大小分别为 64 字节、 128 字节或 256 字节。指令语法中也给出了枚举值，cute 源码中的 copy Operation 对象都采用了 `L2::128B`。
- `[src]` 和 `[dst]` 分别指定异步拷贝操作的源地址和目的地址。前面说过，操作数 `src` 指定为全局状态空间中的位置，`dst` 指定为共享状态空间中的位置。要注意的是，指向共享内存空间的地址 `dst` 不能直接传原始指针，需要使用内置 `__cvta_generic_to_shared` 函数转化为 `uint32_t` 类型再入参。
- `cp-size` 是一个整型常量，它指定要拷贝到目标 `dst` 的数据大小（以字节为单位）。`cp-size` 只能是 4 、8 和 16。
- `src-size` 是一个可选参数，表示从 `src` 拷贝到 `dst` 的数据大小（以字节为单位），并且必须小于 `cp-size`。在这种情况下，目标 `dst` 中剩余的字节将被填充为零。指定大于 `cp-size` 的 `src-size` 将导致未定义行为。
- `ignore-src` 是一个可选参数，指定是否完全忽略源位置 `src` 的数据。如果忽略源数据，则将 0 拷贝到目标 `dst`。如果未指定参数 `ignore-src`，则默认为 `False`。
- `cache-policy` 指定在内存访问期间可能使用的缓存剔除策略，该参数只是一个性能提示，并不一定会被采纳，也不会改变内存一致性行为。如果指定了参数 `cache-policy`，则必须指定限定符 `.level::cache_hint`，文档中针对该参数并未给出枚举值，笔者也没见别人用过，由于 `.level::cache_hint` 只有一个枚举值 `.L2::cache_hint`，故猜测这个参数可能与 L2 缓存的访问策略有关，有兴趣的读者可以阅读笔者的另一篇文章 [【CUDA编程】CUDA L2 缓存介绍](https://mp.weixin.qq.com/s/OCXv9B9HseUC_vCdh3ef7g)。

了解了指令参数的含义，我们再来理解 cute 库的 copy Operation 对象，可以看出 `SM80_CP_ASYNC_CACHEGLOBAL` 和 `SM80_CP_ASYNC_CACHEALWAYS` 除了是否访问 L1 缓存以外，其他功能都是类似的，都是从全局内存地址 `gmem_ptr` 拷贝 `sizeof(TS)` 个字节的数据到共享内存地址 `smem_dst`，并且指定在 L2 缓存中缓存 128 字节。

## 3 基于 async-group 的异步拷贝完成机制

PTX 文档指出，线程必须显式等待异步拷贝操作的完成才能访问操作的结果。一旦异步拷贝操作被启动，在异步操作完成之前对源地址进行写操作或修改张量描述符，或从目标地址读取数据，会表现出未定义行为。PTX 中提供了两种异步拷贝操作完成机制：Async‑group 机制和基于 mbarrier 的机制，使用任意一种均可，cute 库中使用的是第一种，笔者这里也只介绍第一种。

当使用 cp.async-group 完成机制时，执行线程使用一个 commit 操作指定一组异步拷贝操作，称为 **cp.async-group**，并使用一个 wait 操作跟踪该 cp.async-group 的完成信号。

执行线程每发出一个提交操作同时就创建一个 cp.async-group，也就是说这个  cp.async-group 是 per-thread 的，其中包含由执行线程在创建 cp.async-group 之前发起的所有异步拷贝操作，但不包括提交操作之后的任何异步拷贝操作。每个已提交的异步操作只属于单个 cp.async-group，也就是说在两次提交操作之间的异步拷贝操作只归属于后面的 cp.async-group。

当 cp.async-group 完成时，属于该组的所有异步拷贝操作都已完成，发起异步操作的执行线程可以读取异步操作的结果。执行线程提交的所有 cp.async-groups 总是按照提交的顺序完成，一个 cp.async-groups 内的异步操作之间没有顺序，也就是说 cp.async-groups 之间的完成顺序取决于提交的顺序，cp.async-groups 之内的操作完成顺序是未定义的。

基于 cp.async-groups 的完成机制的典型模式如下：

- 发起异步操作。
-  将异步操作组合到一个 cp.async-groups 中，使用一个 commit 操作。
- 使用 wait 操作等待异步组的完成。
- 一旦 cp.async-groups 完成，就可以访问该 cp.async-groups 中所有异步操作的结果。

现在我们先不看具体 PTX 指令，先看一下 cute 库中封装的函数，不妨来对照一下：

```cpp
copy(gs_tiled_copy, tAgA, tAsA);
cp_async_fence();
...
cp_async_wait<0>(); 
__syncthreads();
```

在 cute 库提供的官方示例中我们经常可以看到类似上述的代码结构，首先 `copy` 函数封装的就是上节介绍的 `cp.async` 指令，相当于一个异步拷贝操作；然后调用 `cp_async_fence()` 函数，fence，顾名思义这里是一个内存栅栏，我们不妨看一下其封装的底层代码：

```cpp
CUTE_HOST_DEVICE
void
cp_async_fence()
{
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  	asm volatile("cp.async.commit_group;\n" ::);
#endif
}
```

可以看出其封装了一个 `cp.async.commit_group` 指令，这就是我们刚刚介绍的执行线程的 commit 操作，通过该操作创建了一个 cp.async-groups，其中包含了上面的 copy 操作，该指令比较简单，也没有什么参数，不再赘述。

异步操作 commit 之后，通常会执行一些与该数据无关的一些计算操作，然后在某个需要用到拷贝数据的节点执行 wait 操作等待拷贝完成。这里调用了 `cp_async_wait<0>()` 函数，我们也来看一下底层代码：

```cpp
template <int N>
CUTE_HOST_DEVICE
void
cp_async_wait()
{
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    if constexpr (N == 0) {
        asm volatile("cp.async.wait_all;\n" ::);
    } else {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
    }
#endif
}

template <int N>
CUTE_HOST_DEVICE
void
cp_async_wait(Int<N>)
{
    return cp_async_wait<N>();
}
```

可以看出，针对入参 `N` 的不同取值，分别使用了两个指令：`cp.async.wait_all` 和 `cp.async.wait_group N`。

`cp.async.wait_group` 指令使得执行线程进入等待状态，直到只有 `N` 或更少的最新 cp.async-group 还未完成，并且所有先前由执行线程提交的 cp.async-group 都已完成。也就是说，`cp.async.wait_group` 指令的参数 `N` 指定了最接近 cp.async.wait_group 调用的、可以处于未完成状态的 cp.async-group 的数量。如果 `N` 为 0，则执行线程应等待所有之前的 cp.async-group 完成。

```ptx
cp.async.ca.shared.global [shrd0], [gbl0], 4;
cp.async.ca.shared.global [shrd1], [gbl1], 16;
cp.async.commit_group;  // End of group 0

cp.async.cg.shared.global [shrd2], [gbl2], 8;
cp.async.cg.shared.global [shrd3], [gbl3], 16;
cp.async.commit_group;  // End of group 1

cp.async.cg.shared.global [shrd4], [gbl4], 8;
cp.async.cg.shared.global [shrd5], [gbl5], 16;
cp.async.commit_group;  // End of group 2

cp.async.wait_group 1;  // waits for group 0 and group 1 to complete
```

对文档中的示例来说，示例中有三个 `cp.async-group`，`cp.async.wait_group` 指令的参数 `N` 等于 1，因此只需要等待前两个 `cp.async-group` 完成，最后一个 `cp.async-group` 可以未完成。

`cp.async.wait_all` 相当于 `cp.async.commit_group` 和 `cp.async.wait_group 0`的功能合并。即 `cp.async.wait_all` 本身自带一个 commit 操作，如果要使用 `cp.async.wait_all` 那么可以省略掉一次 commit ，例如下面这样：

```ptx
cp.async.ca.shared.global [shrd0], [gbl0], 4;
cp.async.ca.shared.global [shrd1], [gbl1], 16;
cp.async.wait_all;  
```

此时 cp.async.wait_all 会等待之前启动的所有 cp.async 完成。

对于执行线程而言，只有 cp.async  操作完成后（即，当 `cp.async.wait_all` 完成或在 cp.async 所属的 `cp.async-group` 上完成 `cp.async.wait_group`），cp.async 操作执行的写入数据才对执行线程可见，此时为了对 block 内其他线程可见，需要在 wait 操作后加一个同步点 `__syncthreads()`，让其他线程也阻塞在 wait 操作后，在 `__syncthreads()` 后，异步拷贝操作对 block 内线程可见。

## 4 小结

本文从 cute 库中的 copy Operation 对象入手，介绍了 cute 库源码的封装逻辑，并对相应的 ptx 指令用法进行介绍，具体如下：

- cp.async 是一条非阻塞指令，启动了一个异步拷贝操作，将源地址操作数 src 指定的位置的数据拷贝到目标地址操作数 dst 指定的位置。
- PTX 中提供了两种异步拷贝操作完成机制：Async‑group 机制和基于 mbarrier 的机制，使用任意一种均可。
- `cp.async.commit_group` 指令创建了一个 cp.async-group，其中包含由执行线程在创建 cp.async-group 之前发起的所有异步拷贝操作，但不包括提交操作之后的任何异步拷贝操作。
- 执行线程使用 `cp.async.wait_all` 和 `cp.async.wait_group N` 指令等待异步拷贝操作完成，注意， `__syncthreads()` 不能取代 wait 指令，这是两码事不能混淆。



