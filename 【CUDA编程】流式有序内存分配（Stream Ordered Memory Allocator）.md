#! https://zhuanlan.zhihu.com/p/677268397
![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/GJUG0H1sS5pl8AXxibhJicH2BjscfVC6ScaYR9uXUjdbxogE70TGf6k9C7zaoiaPlORXnaicAujUY4fib7shQ6KRNww/640?wx_fmt=jpeg&amp;from=appmsg)

# 【CUDA编程】流式有序内存分配（Stream Ordered Memory Allocator）

**写在前面**：本文是笔者手稿的第 11 章的内容，主要介绍了一种新的设备内存分配机制，即流式有序内存分配。如有错漏之处，请读者们务必指出，感谢！

## 1 介绍 
大多数 CUDA 开发人员应该都知道如何使用 `cudaMalloc` 和 `cudaFree` API 来分配和释放 GPU 可访问内存。但这些 API 长期以来一直存在一个问题，就是它们并非是流式有序的（Stream Ordered），且使用 `cudaMalloc` 和 `cudaFree` 管理内存分配会同步所有正在执行的 CUDA 流，极大地影响了运行时性能。在本章中，我们将介绍新的 API 函数 `cudaMallocAsync` 和 `cudaFreeAsync`，它们使内存分配和释放从同步整个设备的全局作用域操作转换为流式有序操作。

**流式有序内存分配**（Stream Ordered Memory Allocator）使应用程序能够将内存分配和释放与处于相同 CUDA 流中的其他任务按照它们发出的顺序执行。

对于大多数应用程序来说，流式有序内存分配的引入，使得内存管理方面更加简单高效。对于使用自定义内存分配的应用程序和库，采用流有序内存分配可以使多个库共享由驱动程序管理的公共内存池，从而减少多余的内存消耗。此外，驱动程序也会基于实际代码中的内存分配、API 调用等来执行相关底层优化。

## 2 查询是否支持流式有序内存分配 

应用程序在使用流式有序内存分配之前需要确保设备支持该功能，可以通过调用 `cudaDeviceGetAttribute()` 并传入 `cudaDevAttrMemoryPoolsSupported` 属性来查询。

从 CUDA 11.3 开始，可以通过查询 `cudaDevAttrMemoryPoolSupportedHandleTypes` 设备属性来查询设备是否支持 IPC 内存池，旧的驱动程序将返回 `cudaErrorInvalidValue` 错误，因为旧版本驱动程序不知道属性枚举。

```cuda
int driverVersion = 0;
int deviceSupportsMemoryPools = 0;
int poolSupportedHandleTypes = 0;
cudaDriverGetVersion(&driverVersion);
if (driverVersion >= 11020) {
    cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
                            cudaDevAttrMemoryPoolsSupported, device);
}
if (deviceSupportsMemoryPools != 0) {
    // `device` supports the Stream Ordered Memory Allocator
}

if (driverVersion >= 11030) {
    cudaDeviceGetAttribute(&poolSupportedHandleTypes,
              cudaDevAttrMemoryPoolSupportedHandleTypes, device);
}
if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {
    // Pools on the specified device can be created with posix file descriptor-based IPC
}
```

注意，在查询之前先执行驱动程序版本检查可以避免在尚未定义属性的驱动程序上遇到 `cudaErrorInvalidValue` 错误。

## 3 API 介绍（cudaMallocAsync 和 cudaFreeAsync） 
前面介绍过，CUDA 中引入了 `cudaMallocAsync` 和 `cudaFreeAsync` 两个 API，用于实现流式有序内存分配功能。应用程序通过 `cudaMallocAsync` 进行内存分配，通过 `cudaFreeAsync` 释放分配的内存。两个 API 都接受 CUDA 流参数来定义分配内存的生命周期（即何时可用、何时不可用）。`cudaMallocAsync` 返回的指针值是同步确定的，可用于构建未来的工作。需要注意的是，`cudaMallocAsync` 在确定内存分配的位置时会忽略当前设备及 Context，而是根据指定的内存池或提供的 CUDA 流来确定驻留的设备。最简单的使用模式是内存分配、使用和释放都在同一个 CUDA 流中进行。

```cuda
void *ptr;
size_t size = 512;
cudaMallocAsync(&ptr, size, cudaStreamPerThread);
// do work using the allocation
kernel<<<..., cudaStreamPerThread>>>(ptr, ...);
// An asynchronous free can be specified without synchronizing the cpu and GPU
cudaFreeAsync(ptr, cudaStreamPerThread);
```

在调用 `cudaMallocAsync` 函数时会传入一个 CUDA 流作为参数，我们姑且称之为**分配流**（allocating stream）。当在分配流之外的其他流中使用分配的内存指针时，应用程序必须保证对该内存的访问发生在分配流中的内存分配操作之后，否则行为未定义。应用程序可以通过同步分配流或使用 CUDA 事件来同步生产和消费流来保证这一行为。

在调用 `cudaFreeAsync` 函数时会在流中插入一个内存释放操作，我们姑且称这个流为**释放流**（freeing stream）。应用程序必须保证内存释放操作发生在内存分配操作和对该内存的所有使用行为之后，在内存释放操作开始后，对该内存的任何访问行为都是未定义的。针对这个情况，建议使用事件和 CUDA 流同步操作来保证在释放流开始释放操作之前完成其他流上对于该内存的任何访问。

```cuda
cudaMallocAsync(&ptr, size, stream1);
cudaEventRecord(event1, stream1);
//stream2 must wait for the allocation to be ready before accessing
cudaStreamWaitEvent(stream2, event1);
kernel<<<..., stream2>>>(ptr, ...);
cudaEventRecord(event2, stream2);
// stream3 must wait for stream2 to finish accessing the allocation before
// freeing the allocation
cudaStreamWaitEvent(stream3, event2);
cudaFreeAsync(ptr, stream3);
```

前面介绍的内存分配与释放操作对应的 API 都是配对的，就是说 `cudaMallocAsync` 对应 `cudaFreeAsync`，`cudaMalloc` 对应 `cudaFree`。实际上，关于内存的分配与释放操作，也可以使用不配对的 API，具体注意事项如下。

应用程序可以使用 `cudaFreeAsync()` 函数释放由 `cudaMalloc()` 分配的内存。同样地，应用程序必须保证在内存释放操作开始之前对该内存的所有访问行为已经完成。要注意的是，在下一次同步传递到 `cudaFreeAsync()` 的流之前，不会释放基础内存，也就是说如果要确保内存释放后再执行后续代码，应该加入同步操作。

```cuda
cudaMalloc(&ptr, size);
kernel<<<..., stream>>>(ptr, ...);
cudaFreeAsync(ptr, stream);
cudaStreamSynchronize(stream); // The memory for ptr is freed at this point 
```

应用程序可以使用 `cudaFree()` 释放使用 `cudaMallocAsync()` 函数分配的内存。通过 `cudaFree()` API 释放此类内存时，驱动程序假定对该内存的所有访问都已完成，并且不执行进一步的同步。针对这种情况，用户可以使用 `cudaStreamQuery`、`cudaStreamSynchronize`、`cudaEventQuery`、`cudaEventSynchronize`、`cudaDeviceSynchronize` 等 API 来保证相应的异步工作已经完成并且 GPU 不会再访问该内存。

```cuda
cudaMallocAsync(&ptr, size,stream);
kernel<<<..., stream>>>(ptr, ...);
// synchronize is needed to avoid prematurely freeing the memory
cudaStreamSynchronize(stream);
cudaFree(ptr);
```

## 4 内存池和 cudaMemPool\_t 

流式有序内存分配中引入了**内存池**（Memory Pools）的概念，内存池是预先分配的内存的集合，可以在后续的内存分配中重复使用。在 CUDA 中，内存池用 `cudaMemPool_t` 类型的句柄表示，应用程序可以显式创建内存池并使用它。此外，每个设备都有一个**默认内存池**（default memory pool）的概念，可以使用 `cudaDeviceGetDefaultMemPool` 查询其句柄。

CUDA 程序中所有对 `cudaMallocAsync` API 的调用都会使用内存池的资源。通常在没有显式指定内存池的情况下，`cudaMallocAsync` API 使用传入的流所在设备的**当前内存池**（current memory pool）。设备的当前内存池可以使用 `cudaDeviceSetMempool` API 设置并使用 `cudaDeviceGetMempool` API 查询。默认情况下（在没有显式调用 `cudaDeviceSetMempool` API 的情况下），当前内存池是设备的默认内存池。

需要说明的是，`cudaMallocFromPoolAsync` API 和 `cudaMallocAsync` API 的 C++ 的重载版本允许用户直接指定要用于分配的内存池，而无需先将其设置为当前池。

## 5 默认（隐式）内存池 

默认内存池，也称隐式内存池，即无需显式创建而实际存在的内存池，应用程序可以使用 `cudaDeviceGetDefaultMemPool` API 检索设备的默认内存池。在设备默认内存池中进行的内存分配是不可迁移的，这些内存分配始终可以从该设备访问。默认内存池的访问权限可以通过 `cudaMemPoolSetAccess` 进行修改，并通过 `cudaMemPoolGetAccess` 进行查询。设备默认内存池不支持 IPC 功能。

## 6 显式内存池 

应用程序可以通过调用 `cudaMemPoolCreate` API 创建一个内存池，称为**显式内存池**（Explicit Pools）。目前内存池只能用于设备内存的分配，具体内存分配所将驻留的设备需要在属性结构中指定，具体可见以下代码。显式池的主要使用场景是 IPC 功能。

```cuda
// create a pool similar to the implicit pool on device 0
int device = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = device;
poolProps.location.type = cudaMemLocationTypeDevice;

cudaMemPoolCreate(&memPool, &poolProps));
```

## 7 物理页面缓存行为 

默认情况下，流式有序内存分配的底层逻辑会尝试最小化内存池占用的物理内存，从而降低应用程序的内存占用。

当应用程序调用 `cudaMallocAsync` API 时，如果内存池中内存不足，CUDA 驱动程序将调用操作系统以分配更多内存，这种操作系统调用开销是昂贵的。当应用程序调用 `cudaFreeAsync` API 时，CUDA 驱动程序又会将内存返回到池中，然后可在后续 `cudaMallocAsync` 请求中重新使用这部分内存。默认情况下，在事件、流或设备上的下一次同步操作期间，内存池中累积的未使用内存将返回到操作系统，如下面的代码示例所示。

```cuda
cudaMallocAsync(ptr1, size1, stream); // Allocates new memory into the pool
kernel<<<..., stream>>>(ptr);
cudaFreeAsync(ptr1, stream); // Frees memory back to the pool
cudaMallocAsync(ptr2, size2, stream); // Allocates existing memory from the pool
kernel<<<..., stream>>>(ptr2);
cudaFreeAsync(ptr2, stream); // Frees memory back to the pool
cudaDeviceSynchronize(); // Frees unused memory accumulated in the pool back to the OS
// Note: cudaStreamSynchronize(stream) achieves the same effect here 
```

可见，当内存池物理内存资源不足或同步操作期间，会由于物理内存的分配和释放存在一些操作系统调用，为了尽量减少这种系统调用，应用程序必须为每个内存池配置内存占用阈值。应用程序可以通过设置**释放阈值**属性 `cudaMemPoolAttrReleaseThreshold` 执行此操作，释放阈值是内存池在尝试将内存释放回操作系统之前应保留的内存量（以字节为单位）。当内存池持有超过释放阈值大小的内存时，在下一次调用流、事件或设备同步时会将多余的内存释放回操作系统，否则未使用的内存在同步操作前后将保持不变，也就不会产生前述系统调用。

默认情况下，内存池的释放阈值为零。这意味着内存池中使用的内存在每次同步操作期间都会释放回操作系统，将释放阈值设置为 `UINT64_MAX` 将防止 CUDA 驱动程序在每次同步后收缩池的行为。

```cuda
Cuuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
```

内存池的释放阈值只是一个提示，在相同的内存池中还可以隐式释放内存分配，以使新的内存分配能够成功。例如，对 `cudaMalloc` 或 `cuMemCreate` 的调用可能会导致 CUDA 从与同一进程中的设备关联的任何内存池中释放未使用的内存来为新的请求提供服务。这在应用程序使用多个库的情况下尤其有用，其中一些库使用 `cudaMallocAsync`，而另一些库不使用 `cudaMallocAsync`。通过自动释放未使用的池内存，这些库不必相互协调以使各自的内存分配请求成功。

前面介绍的内存池收缩场景都是 CUDA 驱动程序自动将内存从内存池释放并重新分配给不相关的分配请求。有时由于业务诉求，需要显式的进行内存池收缩，比如应用程序为了禁止同步期间的内存池收缩，可能会设置一个很大的释放阈值，导致内存池无法自动收缩，再比如应用程序可能使用不同的接口（如 Vulkan 或 DirectX ）来访问 GPU ，或者可能有多个进程同时使用 GPU，这些场景下在进行新的内存分配请求时 CUDA 驱动程序都不会自动收缩内存池，所以此时缺少一个主动收缩内存池的机制。

在这种情况下，应用程序可以调用 `cudaMemPoolTrimTo` API 显式释放内存池中未使用的内存，通过设置 `minBytesToKeep` 参数保留在后续执行阶段需要的内存量。

```cuda
Cuuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

// application phase needing a lot of memory from the stream ordered allocator
for (i=0; i<10; i++) {
    for (j=0; j<10; j++) {
        cudaMallocAsync(&ptrs[j],size[j], stream);
    }
    kernel<<<...,stream>>>(ptrs,...);
    for (j=0; j<10; j++) {
        cudaFreeAsync(ptrs[j], stream);
    }
}

// Process does not need as much memory for the next phase.
// Synchronize so that the trim operation will know that the allocations are no
// longer in use.
cudaStreamSynchronize(stream);
cudaMemPoolTrimTo(mempool, 0);

// Some other process/allocation mechanism can now use the physical memory
// released by the trimming operation.
```

## 8 资源使用情况统计 

从 CUDA 11.3 开始，内存池 `cudaMemoryPool_t` 中新添加了 $4$ 个属性用来查询内存池的内存使用情况，分别为：`cudaMemPoolAttrReservedMemCurrent`、`cudaMemPoolAttrReservedMemHigh`、`cudaMemPoolAttrUsedMemCurrent` 和 `cudaMemPoolAttrUsedMemHigh`，具体属性含义如下：

- `cudaMemPoolAttrReservedMemCurrent`：内存池中目前占用的总物理 GPU 内存；
- `cudaMemPoolAttrUsedMemCurrent`：内存池中目前已经分配的不可重用的所有内存的总大小；
- `cudaMemPoolAttrReservedMemHigh`：上次重置以来，`cudaMemPoolAttrReservedMemCurrent` 达到的最大值；
- `cudaMemPoolAttrUsedMemHigh`：上次重置以来，`cudaMemPoolAttrUsedMemCurrent` 达到的最大值。


这 $4$ 个指标其实分别就是内存池占用的物理内存、已使用的内存以及两者的峰值，关于峰值属性，可以通过 `cudaMemPoolSetAttribute` API 重置，具体代码如下：

```cuda
// sample helper functions for getting the usage statistics in bulk
struct usageStatistics {
    cuuint64_t reserved;
    cuuint64_t reservedHigh;
    cuuint64_t used;
    cuuint64_t usedHigh;
};

void getUsageStatistics(cudaMemoryPool_t memPool, struct usageStatistics *statistics)
{
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, statistics->reserved);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, statistics->reservedHigh);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, statistics->used);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, statistics->usedHigh);
}


// resetting the watermarks will make them take on the current value.
void resetStatistics(cudaMemoryPool_t memPool)
{
    cuuint64_t value = 0;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &value);
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &value);
}
```

## 9 内存重用策略 

在接收到新的内存分配请求时，CUDA 驱动程序在尝试从操作系统分配更多物理内存之前会优先重用之前通过 `cudaFreeAsync()` 释放的内存，避免进行昂贵的系统调用。通过这种方式，CUDA 驱动程序可以帮助降低应用程序的内存占用，同时提高内存分配性能。

此外，流式有序内存分配机制中还有一些可控的分配策略，应用程序可以通过内存池属性 `cudaMemPoolReuseFollowEventDependencies`、`cudaMemPoolReuseAllowOpportunistic` 和 `cudaMemPoolReuseAllowInternalDependencies` 控制这些策略。注意：新版本的 CUDA 驱动程序可能会对这些重用策略进行更新和删减，具体使用时要注意驱动版本。

### 9.1 流内内存重用 
考虑下面的代码，当 CUDA 流中发生新的内存分配请求时，`ptr2` 在分配时可以重用之前 `ptr1` 的部分或全部内存。

```cuda
cudaMallocAsync(&ptr1, size1, stream);
kernelA<<<..., stream>>>(ptr1);
cudaFreeAsync(ptr1, stream);
cudaMallocAsync(&ptr2, size2, stream);
kernelB<<<..., stream>>>(ptr2); 
```

在这个代码示例中，`ptr2` 是在 `ptr1` 被释放后按流顺序分配的。`ptr2` 在分配时可以重用之前 `ptr1` 的部分或全部内存，而无需任何同步，因为 `kernelA` 和 `kernelB` 在同一个流中启动，流排序语义保证 `kernelB` 在 `kernelA` 完成之前不能开始执行和访问内存，这就是所谓的流式有序内存分配。

### 9.2 跨流内存重用 

CUDA 驱动程序可以通过流中插入的 CUDA 事件跟踪流之间的依赖关系，如以下代码示例所示：

```cuda
cudaMallocAsync(&ptr1, size1, streamA);
kernelA<<<..., streamA>>>(ptr1);
cudaFreeAsync(ptr1, streamA);
cudaEventRecord(event, streamA);
cudaStreamWaitEvent(streamB, event, 0);
cudaMallocAsync(&ptr2, size2, streamB);
kernelB<<<..., streamB>>>(ptr2); 
```

代码中通过 CUDA 事件保证 `ptr2` 是在 `ptr1` 被释放后分配的，因此 CUDA 驱动程序在分配 `ptr2` 时可以重用之前 `ptr1` 的部分或全部内存。实际应用中，`streamA` 和 `streamB` 之间的依赖关系链可以包含任意数量的流，如下面的代码示例所示。

```cuda
cudaMallocAsync(&ptr1, size1, streamA);
kernelA<<<..., streamA>>>(ptr1);
cudaFreeAsync(ptr1, streamA);
cudaEventRecord(event, streamA);
for (int i = 0; i < 100; i++) {
    cudaStreamWaitEvent(streams[i], event, 0);       // streams[] is a previously created array of streams
    cudaEventRecord(event, streams[i]);
}
cudaStreamWaitEvent(streamB, event, 0);
cudaMallocAsync(&ptr2, size2, streamB);
kernelB<<<..., streamB>>>(ptr2); 
```

这种基于事件依赖的内存重用策略是 CUDA 驱动程序自动进行的，如果应用程序需要禁用这种重用策略，可以通过如下代码在内存池的层面实现禁用：

```cuda
int enable = 0;
cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseFollowEventDependencies, &enable);
```

CUDA 驱动程序还可以在没有指定显式依赖项的情况下，启发式地重用内存。考虑下面的代码示例：

```cuda
cudaMallocAsync(&ptr1, size1, streamA);
kernelA<<<..., streamA>>>(ptr1);
cudaFreeAsync(ptr1);
cudaMallocAsync(&ptr2, size2, streamB);
kernelB<<<..., streamB>>>(ptr2);
cudaFreeAsync(ptr2); 
```
  
在此场景中，`streamA` 和 `streamB` 之间没有明确的依赖关系。但是， CUDA 驱动程序知道每个流执行了多远。如果在调用 `streamB` 中的 `cudaMallocAsync` 时，CUDA 驱动程序确定 `kernelA` 已在 GPU 上完成执行，则它在分配 `ptr2` 时可以重用之前 `ptr1` 的部分或全部内存。

退一步说，如果 `kernelA` 尚未完成执行， CUDA 驱动程序可以在两个流之间添加隐式依赖项，以便 `kernelB` 在 `kernelA` 完成之前不会开始执行，从而实现一种隐式的“有序”，以便在分配 `ptr2` 时可以重用之前 `ptr1` 的部分或全部内存。

虽然这些启发式方法可能有助于提高性能或避免内存分配失败，但它们会给应用程序增加不确定性，因此可以在每个池的基础上禁用。应用程序可以按如下方式禁用这些启发式内存重用策略：

```cuda
int enable = 0;
cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseAllowOpportunistic, &enable);
cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseAllowInternalDependencies, &enable); 
```

类似地场景，当一个流与主机端同步时，之前在该流中释放的内存也可以重新用于其他任何流中新的内存分配。说白了，只要“有序”，就可以重用，有序是重用的前提，应用程序可以根据具体情况手动控制某些重用策略是否生效。

## 10 多设备场景的访问控制 

就像通过虚拟内存管理 API 控制的内存分配的访问权限一样，内存池中分配的内存（即使用 `cudaMallocAsync` 分配的内存）的访问权限通过专门的 `cudaMemPoolSetAccess` API 进行控制，而不是  `cudaDeviceEnablePeerAccess` 或 `cuCtxEnablePeerAccess`。默认情况下，使用 `cudaMallocAsync` 分配的内存可以在与指定流关联的设备中进行访问，这种在内存池自身所在设备访问内存的权限是默认存在且无法撤销的。

要想从其他设备访问这块内存，需要启用其他设备对这个内存池的访问权限。同时还要求设备与设备之间具有对等（peer）访问能力，可以通过调用 `cudaDeviceCanAccessPeer` 接口来判断，如果未检查对等功能，则设置访问时可能会失败，并返回 `cudaErrorInvalidDevice` 错误。如果内存池中没有进行内存分配，即使设备不具备对等能力，`cudaMemPoolSetAccess` 调用也可能成功；在这种情况下，该内存池中的下一次内存分配将会失败。

值得注意的是，调用 `cudaMemPoolSetAccess` 会影响内存池中的所有的内存分配，而不仅仅是后续分配的内存（即调用 `cudaMemPoolSetAccess` 后分配的内存）。同样地，调用 `cudaMemPoolGetAccess`  获取内存池访问权限报告也适用于内存池中所有分配的内存，而不仅仅是后续分配的内存。建议不要频繁更改给定设备的内存池的访问权限设置，一旦内存池可以从给定的设备访问，那么该设备应该在内存池的整个生命周期内都可以访问池中分配的内存。下面的代码展示了如何设置访问权限实现从设备 `accessingDevice` 访问设备 `residentDevice` 上内存池 `memPool` 中的内存。

```cuda
// snippet showing usage of cudaMemPoolSetAccess:
cudaError_t setAccessOnDevice(cudaMemPool_t memPool, int residentDevice,
              int accessingDevice) {
    cudaMemAccessDesc accessDesc = {};
    accessDesc.location.type = cudaMemLocationTypeDevice;
    accessDesc.location.id = accessingDevice;
    accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

    int canAccess = 0;
    cudaError_t error = cudaDeviceCanAccessPeer(&canAccess, accessingDevice,
              residentDevice);
    if (error != cudaSuccess) {
        return error;
    } else if (canAccess == 0) {
        return cudaErrorPeerAccessUnsupported;
    }

    // Make the address accessible
    return cudaMemPoolSetAccess(memPool, &accessDesc, 1);
}
```

此外，如果应用程序需要撤销对内存池所在设备以外的设备的访问权限，可以在调用 `cudaMemPoolSetAccess` 时传入 `cudaMemAccessFlagsProtNone` 标识，注意，这只会撤销其他设备的访问权限，本设备的访问权限是无法撤销的。

## 11 进程间通信（IPC）内存池 

前面介绍过，设备默认内存池不支持 IPC 功能，即分配的内存不能与其他进程共享。应用程序如果要与其他进程共享使用 `cudaMallocAsync` 分配的内存，需要显式创建自己的内存池。支持 IPC 的内存池允许在进程之间轻松、高效和安全地共享 GPU 内存。

基于内存池的实现的 IPC 功能可以分为两个阶段：进程首先需要共享对内存池的访问权限，然后共享来自该内存池的特定内存分配。

### 11.1 创建并共享 IPC 内存池 

共享对内存池的访问权限主要包括 $3$ 部分内容：通过 `cudaMemPoolExportToShareableHandle()` API 检索该池的操作系统本机句柄；使用常规的操作系统本地 IPC 机制将句柄传输到导入进程；通过 `cudaMemPoolImportFromShareableHandle()` API 创建导入进程的内存池。

使用 `cudaMemPoolExportToShareableHandle()` API 前，还需要在创建导出进程的内存池时设置相应的内存池属性，具体参考代码如下：

```cuda
// in exporting process
// create an exportable IPC capable pool on device 0
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = 0;
poolProps.location.type = cudaMemLocationTypeDevice;

// Setting handleTypes to a non zero value will make the pool exportable (IPC capable)
poolProps.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

cudaMemPoolCreate(&memPool, &poolProps));

// FD based handles are integer types
int fdHandle = 0;


// Retrieve an OS native handle to the pool.
// Note that a pointer to the handle memory is passed in here.
cudaMemPoolExportToShareableHandle(&fdHandle,
              memPool,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
              0);

// The handle must be sent to the importing process with the appropriate
// OS specific APIs.
```

对于导出进程，首先创建**导出内存池** `memPool`，在创建内存池之前需要进行相关属性的设置。注意 `allocType` 属性需要设置为 `cudaMemAllocationTypePinned`，表示该内存 non-migratable，即页锁定内存。`handleTypes` 属性需要设置为 `cudaMemHandleTypePosixFileDescriptor`，表示用户打算查询内存池的文件描述符，以便与其他进程共享。

对于导出进程，内存池创建之后，即可通过 `cudaMemPoolExportToShareableHandle()` API 查询表示该池的操作系统本机句柄 `fdHandle`。

对于导入进程，可以通过 UNIX 域 Socket 等特定操作系统 API 获取导出池的操作系统本机句柄，然后根据获得的句柄调用 `cudaMemPoolImportFromShareableHandle()` API 创建**导入内存池** `importedMemPool`。

### 11.2 设置导入内存池的访问权限 

要注意的是，导入内存池不继承导出进程中对导出内存池进行的任何访问权限设置。所以导入内存池最初只能从其所在的设备中访问，导入进程中如果涉及从其他设备访问导入内存池的内存分配，那么需要对导入内存池进行访问权限设置（使`cudaMemPoolSetAccess`）。

如果导入内存池在导入过程中属于不可见的设备，则用户必须使用 `cudaMemPoolSetAccess` API 来启用访问权限。

### 11.3 从导出内存池中创建、共享内存分配 

内存池共享以后，在导出进程中使用 `cudaMallocAsync()` API 从导出内存池中进行的内存分配可以跟导入池的其他进程共享。由于内存池的安全策略是在池级别建立和验证的，操作系统不需要额外为特定的池分配提供安全性，换句话说，导入进程中需要的不透明句柄 `cudaMemPoolPtrExportData` 可以通过任何标准 IPC 机制（例如通过共享内存、管道等）从导出进程发送到导入进程。

虽然内存分配可以在不与分配流同步的情况下导出和导入，但在访问内存分配时，导入进程必须遵循与导出进程相同的规则。即，对内存分配的访问必须发生在分配流中分配操作的流排序之后，分配操作完成之前不允许访问该内存。以下两个代码片段展示了使用 `cudaMemPoolExportPointer()`、`cudaMemPoolImportPointer()` 及 IPC 事件共享内存分配的过程，并通过事件保证在内存分配完成之前在导入过程中不会访问该内存。

```cuda
// preparing an allocation in the exporting process
cudaMemPoolPtrExportData exportData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t readyIpcEventHandle;

// ipc event for coordinating between processes
// cudaEventInterprocess flag makes the event an ipc event
// cudaEventDisableTiming  is set for performance reasons

cudaEventCreate(
        &readyIpcEvent, cudaEventDisableTiming | cudaEventInterprocess)

// allocate from the exporting mem pool
cudaMallocAsync(&ptr, size,exportMemPool, stream);

// event for sharing when the allocation is ready.
cudaEventRecord(readyIpcEvent, stream);
cudaMemPoolExportPointer(&exportData, ptr);
cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);

// Share IPC event and pointer export data with the importing process using
//  any mechanism. Here we copy the data into shared memory
shmem->ptrData = exportData;
shmem->readyIpcEventHandle = readyIpcEventHandle;
// signal consumers data is ready
```

对于导出进程，首先创建了 CUDA 事件 `readyIpcEvent`，用于标定访问内存的流与分配内存的流之间的依赖关系，由于该事件跨进程，所以使用 `cudaIpcGetEventHandle` API 获取事件句柄 `readyIpcEventHandle`。另外导出进程中调用 `cudaMallocAsync` 的重载版本进行内存分配，这里也可以使用 `cudaMallocFromPoolAsync` API，参数都是一样的。从导出池分配内存后，指针就可以与导入进程共享。通过 `cuMemPoolExportPointer` 获取需要在进程中共享的数据，其返回结果 `exportData` 是数据，而不是句柄，可以使用任何 IPC 机制在进程间共享。将 `exportData`、`readyIpcEventHandle` 存入共享内存。

要注意的是，这里的共享内存跟我们前面介绍的 GPU 设备内存中的共享内存不是一个东西，后者是用于 Kernel 执行过程中线程块（block）中线程通信的一块片上内存，而前者则是主机端的一块物理内存，不同的进程将同一块物理内存映射到各自的地址空间，如果某个进程向这块内存写入数据，其它进程也会知道，这是目前最快的一种 IPC 机制。

```cuda
// Importing an allocation
cudaMemPoolPtrExportData *importData = &shmem->prtData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t *readyIpcEventHandle = &shmem->readyIpcEventHandle;

// Need to retrieve the ipc event handle and the export data from the
// exporting process using any mechanism.  Here we are using shmem and just
// need synchronization to make sure the shared memory is filled in.

cudaIpcOpenEventHandle(&readyIpcEvent, readyIpcEventHandle);

// import the allocation. The operation does not block on the allocation being ready.
cudaMemPoolImportPointer(&ptr, importedMemPool, importData);

// Wait for the prior stream operations in the allocating stream to complete before
// using the allocation in the importing process.
cudaStreamWaitEvent(stream, readyIpcEvent);
kernel<<<..., stream>>>(ptr, ...);
```

对于导入进程，首先从共享内存中取出 `exportData`、`readyIpcEventHandle` 两个句柄，通过 `cudaIpcOpenEventHandle` API 基于事件句柄创建 CUDA 事件 `readyIpcEvent`，通过 `cuMemPoolImportPointer` 基于共享内存中的导入数据 `importData` 获取导入进程中指向导入内存的指针 `ptr`。现在，两个进程已经完成对相同内存分配的共享，根据 `readyIpcEvent` 事件依赖关系，待导出进程中的内存分配完成后，导入进程中即可使用该内存。

针对共享内存池中分配的内存，在导出进程中释放内存之前，必须先在导入进程中释放内存。这是为了确保当导入进程仍在访问之前共享的内存分配时，这块内存不会被释放并重新用于另一个 `cudaMallocAsync` 请求，从而可能导致未定义的行为。为了在 IPC 场景下实现安全可靠的内存释放，可以在进程间加入适当的同步或事件依赖等机制确保释放顺序可控，具体可参阅如下代码：

```cuda
// The free must happen in importing process before the exporting process
kernel<<<..., stream>>>(ptr, ...);

// Last access in importing process
cudaFreeAsync(ptr, stream);

// Access not allowed in the importing process after the free
cudaIpcEventRecord(finishedIpcEvent, stream);
```

对于导入进程，确保不会再访问 `ptr` 指向的内存后，直接调用 `cudaFreeAsync` API 释放内存，为了确保导入进程先释放，需要在 `cudaFreeAsync` 后面插入事件 `finishedIpcEvent`，以便导出进程中等待该事件完成后再释放。

```cuda
// Exporting process
// The exporting process needs to coordinate its free with the stream order
// of the importing process’s free.
cudaStreamWaitEvent(stream, finishedIpcEvent);
kernel<<<..., stream>>>(ptrInExportingProcess, ...);

// The free in the importing process doesn’t stop the exporting process
// from using the allocation.
cudFreeAsync(ptrInExportingProcess,stream);
```

对于导出进程，等待 `finishedIpcEvent` 事件确保导入进程中内存已释放，要注意的是，导入进程释放内存并不影响导出进程继续使用该内存，所以即使导入进程已经完成释放，但是导出进程中依然可以使用 `ptrInExportingProcess` 指向的内存。最后，在导出进程中调用 `cudaFreeAsync` API 完成内存释放。

### 11.4 IPC 导出池的限制 

IPC 导出内存池目前不支持将物理内存释放回操作系统，因此，在对导出池调用 `cudaMemPoolTrimTo` API 希望显式释放内存池中未使用的内存时时不会执行任何操作，并且释放阈值 `cudaMemPoolAttrReleaseThreshold` 也会被 CUDA 驱动程序忽略。

注意：此行为由 CUDA 驱动程序控制，而不是由 Runtime 控制，并且可能会在未来的驱动程序更新中发生变化。

### 11.5 IPC 导入池的限制 

不允许从导入池中分配内存，具体来说，导入池不能设置为当前内存池，也不能在 `cudaMallocFromPoolAsync` API 中使用。因此，前面说的内存重用策略等属性对导入池没有意义。

与导出池一样，导入池目前也不支持将物理内存释放回操作系统，因此，使用 `cudaMemPoolTrimTo` API 与 `cudaMemPoolAttrReleaseThreshold` 属性的效果同上。

通过 `cudaMemPoolGetAttribute` API 查询导入池资源使用情况时，查询的是导入到导入进程的内存分配及其相关的物理内存情况。

## 12 同步 API 操作 

CUDA 驱动程序针对流式有序内存分配集成了同步 API，当用户请求 CUDA 驱动程序同步时，驱动程序等待异步工作完成，在返回之前，驱动程序将确定同步完成时哪些内存将被释放。无论这些内存属于哪个流或者应用程序显式禁用了什么分配策略，这些释放掉的物理内存都可以用于后续的新的内存分配。另外，驱动程序还在同步时检查释放阈值 `cudaMemPoolAttrReleaseThreshold`，并尽可能释放任何多余的物理内存。

## 13 附录 
### 13.1 cudaMemcpyAsync 的 Context 设置

在当前的 CUDA 驱动程序中，当异步内存拷贝操作涉及的内存是由 `cudaMallocAsync` 分配时，应该使用指定流的 Context 作为调用线程的当前 Context。但是，对于对等内存异步拷贝 `cudaMemcpyPeerAsync` 来说不是必需的，因为此时 API 中使用的是指定的设备主 Context 而不是当前 Context。

### 13.2 cuPointerGetAttribute 的 Context 设置

当调用 `cudaFreeAsync` 释放给定内存后，再对该指针调用 `cuPointerGetAttribute` 查询指针相关的属性会导致未定义的行为。

### 13.3 cuGraphAddMemsetNode

`cuGraphAddMemsetNode` API 不适用于流式有序内存分配的内存。但是，通过流式有序内存分配机制分配的内存的属性设置可以被流捕获。

### 13.4 查询指针属性

查询指针属性的 `cuPointerGetAttributes` API 适用于流式有序内存分配。由于流式有序内存分配与 Context 无关，因此通过 `CU_POINTER_ATTRIBUTE_CONTEXT` 参数查询 Context 属性将返回成功，只是在 `*data` 中返回 `NULL`。属性 `CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` 可用于确定内存分配的位置：这在使用 `cudaMemcpyPeerAsync` 进行设备间对等内存拷贝时很有用。`CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE` 属性是在 CUDA 11.3 中添加的，可用于调试和在执行 IPC 之前确认该内存分配来自哪个内存池。
