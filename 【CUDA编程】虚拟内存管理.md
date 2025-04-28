#! https://zhuanlan.zhihu.com/p/676597961
![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/GJUG0H1sS5pl8AXxibhJicH2BjscfVC6ScaYR9uXUjdbxogE70TGf6k9C7zaoiaPlORXnaicAujUY4fib7shQ6KRNww/640?wx_fmt=jpeg&amp;from=appmsg)

# 【CUDA编程】虚拟内存管理

**写在前面**：本文是笔者手稿的第 10 章的内容，主要介绍了虚拟内存管理相关的内容。如有错漏之处，请读者们务必指出，感谢！

## 1 介绍 

虚拟内存管理 API 为应用程序提供了一种直接管理 CUDA 中统一虚拟地址空间的方法，该空间用于将物理内存映射到 GPU 可访问的虚拟地址。这些 API 在 CUDA 10.2 中引入，除了管理虚拟地址空间外，还提供了一种与其他进程和图形 API（如 OpenGL 和 Vulkan）进行互操作的新方法，此外还提供了新的内存属性可供用户根据不同需求进行应用程序开发。

通常，调用 CUDA 编程模型中的内存分配 API 后（例如 `cudaMalloc`），会返回一个指向 GPU 内存的内存地址，这个地址可以在任何 CUDA API 中使用，也可以在设备 Kernel 中使用。但是，这样分配的内存无法根据用户的内存需求重新调整大小。有时为了增加内存分配的大小，用户必须重新显式分配一个更大的缓冲区，然后从初始分配的内存中拷贝数据到缓冲区，释放初始分配的内存，然后继续新分配的缓冲区的地址替代初始内存的地址。实际上就是重新分配了一块更大的内存然后把数据拷贝过来，这通常会导致较低的性能和较高的应用程序峰值内存利用率。

本质上就是用户有一个类似 `malloc` 的 API 来分配 GPU 内存，但没有相应的 `realloc` API。虚拟内存管理 API 将地址和内存的概念解耦，允许应用程序分别处理它们。应用程序可以使用虚拟内存管理 API 随时从虚拟地址范围映射或取消映射物理内存。

在通过 `cudaEnablePeerAccess` 启用对等设备内存访问的情况下，已分配和未分配的内存都将映射到目标对等设备，这导致用户无意中支付了将所有 `cudaMalloc` 分配映射到对等设备的运行时成本。然而，在大多数情况下，应用程序在与另一个设备通信时仅共享少量内存，并非所有内存分配都需要映射到所有设备。使用虚拟内存管理 API，应用程序可以专门指定某些内存分配可从目标设备访问。

CUDA 虚拟内存管理 API 向用户提供细粒度控制，以管理应用程序中的 GPU 内存。具体内容如下：

- 将分配在不同设备上的内存放入一个连续的虚拟地址空间范围内。
- 使用平台特定机制执行内存共享的进程间通信。
- 在支持虚拟内存管理 API 的设备上选择新的内存类型。


虚拟内存管理编程模型提供了以下功能用于分配内存：

- 分配物理内存；
- 保留一个虚拟地址空间范围；
- 将分配的内存映射到虚拟地址空间范围；
- 控制映射范围的访问权限。


注意，本节中描述的 API 套件要求操作系统支持 UVA。

## 2 查询设备是否支持虚拟内存管理 

在使用虚拟内存管理 API 之前，应用程序必须确保设备支持 CUDA 虚拟内存管理功能。以下代码示例展示了如何查询设备是否虚拟内存管理：

```cuda
int deviceSupportsVmm;
CUresult result = cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
if (deviceSupportsVmm != 0) {
    // `device` supports Virtual Memory Management
}
```

可以通过调用 `cuDeviceGetAttribute` 查询设备的 `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` 属性，若为 $0$ 则不支持，反之则支持该功能。

## 3 分配物理内存 

使用虚拟内存管理 API 进行内存分配的第一步是创建一块物理内存，为后续内存分配提供支持。应用程序必须使用 `cuMemCreate` API 分配物理内存，此函数仅仅是进行物理内存分配，不会做任何设备或主机端内存映射。函数参数 `CUmemGenericAllocationHandle` 描述了要分配的物理内存的属性，例如分配的内存的位置、分配的内存是否要共享给另一个进程（或其他图形 API），或者要分配的内存的物理属性。另外，用户在使用时必须确保请求分配的内存大小必须与适当的粒度对齐。虚拟内存管理 API 提供了 `cuMemGetAllocationGranularity` 函数，用于查询有关分配粒度要求的信息。以下代码片段展示了使用 `cuMemCreate` 分配物理内存的过程：

```cuda
CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    // Allocate physical memory
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);

    return allocHandle;
}
```

通过 `cuMemCreate` 函数分配的内存可以使用该函数返回的 `CUmemGenericAllocationHandle` 句柄引用。这与 `cudaMalloc` 风格的内存分配不同，后者返回一个指向 GPU 内存的指针，该指针可由在设备上执行的 Kernel 中直接访问。分配的物理内存除了使用 `cuMemGetAllocationPropertiesFromHandle` API 查询内存的属性之外，不能用于其他任何操作。为了使该物理内存可访问，应用程序必须将此内存映射到由 `cuMemAddressReserve` 保留的虚拟地址空间范围，并为其提供适当的访问权限。使用完毕后，应用程序必须通过 `cuMemRelease` API 释放分配的内存。

### 3.1 分配可共享的内存 

这里的可共享的内存指的是应用程序用于进程间通信或图形互操作目的的一块内存，而不是传统 CUDA 内存模型中的共享内存的概念。在使用 `cuMemCreate` 分配内存时，用户可以将 `CUmemAllocationProp::requestedHandleTypes` 设置为特定字段，从而向 CUDA 指示该内存是可用于进程间通信或图形互操作的可共享内存。在 Windows 系统上，当 `CUmemAllocationProp::requestedHandleTypes` 设置为 `CU_MEM_HANDLE_TYPE_WIN32` 时，应用程序还必须在 `CUmemAllocationProp::win32HandleMetaData` 中指定 `LPSECURITYATTRIBUTES` 属性。该安全属性定义了可以将导出的分配转移到其他进程的范围。

CUDA 虚拟内存管理 API 函数不支持传统的进程间通信函数及其内存，相反，其提供了一种利用操作系统特定句柄进行进程间通信的新机制。应用程序可以使用 `cuMemExportToShareableHandle` 获取与内存分配相对应的这些操作系统特定句柄。这样获得的句柄可以通过使用常规的 OS 原生机制进行传输，从而进行进程间通信。接收进程需要使用 `cuMemImportFromShareableHandle` 导入分配。

另外，用户在导出使用 `cuMemCreate` 分配的内存之前，必须确保设备平台支持请求的句柄类型。以下代码片段展示了不同平台下如何查询是否支持句柄类型的过程：

```cuda
int deviceSupportsIpcHandle;
#if defined(__linux__)
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
#else
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, device));
#endif
```

应用程序可以通过如下方式适当设置 `CUmemAllocationProp::requestedHandleTypes`，代码如下：

```cuda
#if defined(__linux__)
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
    prop.win32HandleMetaData = // Windows specific LPSECURITYATTRIBUTES attribute.
#endif
```

更多关于虚拟内存管理的示例代码可以参阅 CUDA samples 中第三部分的 memMapIPCDrv 示例，该示例中将 IPC 与虚拟内存管理分配一起使用。

###  3.2 内存类型 
在 CUDA 10.2 之前，应用程序无法为一些可能支持特殊内存类型的设备分配任何特殊类型的内存。使用 `cuMemCreate` API 时，应用程序可以通过 `CUmemAllocationProp::allocFlags` 额外指定内存类型，从而使用特定的内存的功能。应用程序在分配特殊内存前，必须确保所请求的内存类型在分配设备上得到支持，否则将会报错。

#### 3.2.1 可压缩内存 

**可压缩内存**（Compressible Memory）可用于对一些非结构化的稀疏数据或其他可压缩数据模式的数据进行加速访问。压缩可以节省 DRAM 带宽、L2 读取带宽和 L2 容量，具体压缩效果取决于具体数据。应用程序可以通过将 `CUmemAllocationProp::allocFlags::compressionType` 设置为 `CU_MEM_ALLOCATION_COMP_GENERIC`，在支持计算数据压缩的设备上分配可压缩内存。当然，再分配之前必须通过 `CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED` 查询设备是否支持计算数据压缩，确保程序不会报错。以下代码片段展示了如何通过 `cuDeviceGetAttribute` 查询设备是否支持可压缩内存的过程： 

```cuda
int compressionSupported = 0;
cuDeviceGetAttribute(&compressionSupported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device);
```

在支持计算数据压缩的设备上，应用程序通过如下方式分配可压缩内存：

```cuda
prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
```

由于硬件资源有限等各种原因，实际分配的内存可能没有压缩属性，因此，应用程序需要使用 `cuMemGetAllocationPropertiesFromHandle` 检查分配内存的压缩属性。

```cuda
CUmemAllocationPropPrivate allocationProp = {};
cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);

if (allocationProp.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC)
{
    // Obtained compressible memory allocation
}
```

## 4 保留一个虚拟地址空间范围 

前面介绍过，在虚拟内存管理中，地址和内存的概念是不同的，因此应用程序必须划出一个地址范围，以容纳由 `cuMemCreate` API 分配的内存。保留的地址范围必须至少与用户计划放入其中的所有分配的物理内存大小的总和一样大。

应用程序可以通过将适当的参数传递给 cuMemAddressReserve 来保留虚拟地址范围，获得的地址范围不会有任何与之关联的设备或主机物理内存。保留的虚拟地址范围可以映射到属于系统中任何设备的内存块，从而为应用程序提供一块连续的虚拟地址空间范围。以下代码片段展示了该函数的用法：

```cuda
CUdeviceptr ptr;
// `ptr` holds the returned start of virtual address range reserved.
CUresult result = cuMemAddressReserve(&ptr, size, 0, 0, 0); // alignment = 0 for default alignment
```

当需要进行内存释放时，应用程序应使用 `cuMemAddressFree` 将虚拟地址范围返回给 CUDA，同时必须确保在调用 `cuMemAddressFree` 之前未映射整个虚拟地址范围。也就是说释放前要对整个虚拟地址范围取消映射，可以调用 `cuMemUnmap` API 实现，这些函数在概念上类似于 `mmap`、`munmap`（在 Linux 上）或 `VirtualAlloc`、`VirtualFree`（在 Windows 上）函数，取消映射后，通过 `cuMemRelease` API 使句柄失效。下面的示例代码展示了一个内存释放的过程：

```cuda
cuMemUnmap(ptr, size);
cuMemRelease(allocHandle);
cuMemAddressFree(ptr, size);
```

## 5 虚拟别名（Aliasing） 

虚拟内存管理 API 提供了一种创建多个虚拟内存映射或“代理''到同一块物理内存的方法，即使用不同的虚拟地址多次调用 `cuMemMap` 函数，姑且称之为虚拟别名（Virtual Aliasing）。除非在 PTX ISA 中另有说明，在写入设备操作（网格启动、memcpy、memset 等）完成之前，对分配的物理内存的一个``代理”的写入被认为与同一物理内存的其他代理不一致和不连贯。同样，如果一个线程网格在写入设备操作之前被调度到 GPU 上但在写入设备操作完成后才开始读取内存，代理之间也被认为具有不一致和不连贯的内存特性。

例如，下面的代码片段被认为是未定义的，假设设备指针 `A` 和 `B` 是同一块物理内存的虚拟别名：

```cuda
__global__ void foo(char *A, char *B) {
    *A = 0x1;
    printf("%d\n", *B);    // Undefined behavior!  *B can take on either
// the previous value or some value in-between.
}
```

在 Kernel 中通过设备指针 `A` 修改了数据，紧接着使用 `B` 读取，由于写入设备操作（该 grid）此时还未完成，所以 `B` 读取到的数据是未定义的。如果 `B` 需要读取到修改后的数据，那么可以将读取操作放在另一个 Kernel，通过流或事件控制两个 Kernel 的执行顺序，使得读取操作在写入操作的 grid 完成之后进行，参考如下代码：

```cuda
__global__ void foo1(char *A) {
    *A = 0x1;
}

__global__ void foo2(char *B) {
    printf("%d\n", *B);    // *B == *A == 0x1 assuming foo2 waits for foo1
// to complete before launching
}

cudaMemcpyAsync(B, input, size, stream1);    // Aliases are allowed at
// operation boundaries
foo1<<<1,1,0,stream1>>>(A);                  // allowing foo1 to access A.
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);
foo2<<<1,1,0,stream2>>>(B);
cudaStreamWaitEvent(stream3, event);
cudaMemcpyAsync(output, B, size, stream3);  // Both launches of foo2 and
                                            // cudaMemcpy (which both
                                            // read) wait for foo1 (which writes)
                                            // to complete before proceeding
```

## 6 内存映射 

前面介绍的分配的物理内存和保留的虚拟地址空间分别代表了虚拟内存管理 API 中引入的内存和地址概念。为了使分配的物理内存可用，应用程序必须首先将内存放在虚拟地址空间中，也就是我们说的内存映射。从 `cuMemAddressReserve` 获取的虚拟地址范围和从 `cuMemCreate` 或 `cuMemImportFromShareableHandle` 获取的已分配的物理内存需要通过 `cuMemMap` 相互关联。

用户可以关联来自多个设备的已分配的物理内存并将其驻留在连续的虚拟地址范围内，只要保留的虚拟地址空间足够大即可。

如果应用程序需要解耦已分配的物理内存和虚拟地址范围，必须通过 `cuMemUnmap` 取消映射的地址。

用户可以根据需要多次将内存映射或取消映射到同一地址范围，只要确保不会在已映射的虚拟地址范围上继续创建映射。以下代码片段展示了映射 API 的用法：

```cuda
CUdeviceptr ptr;
// `ptr`: address in the address range previously reserved by cuMemAddressReserve.
// `allocHandle`: CUmemGenericAllocationHandle obtained by a previous call to cuMemCreate.
CUresult result = cuMemMap(ptr, size, 0, allocHandle, 0);
```

## 7 控制访问权限 

虚拟内存管理 API 使应用程序能够通过访问控制机制显式保护其虚拟地址范围。也就是说，在使用 `cuMemUnmap` 将物理内存映射到虚拟地址范围后，该地址还不能直接从设备访问，并且如果被 CUDA Kernel 访问会导致程序崩溃。应用程序必须使用 `cuMemSetAccess` 函数专门选择访问控制，该函数允许或限制特定设备对映射虚拟地址范围的访问。以下代码片段展示了该函数的用法：

```cuda
void setAccessOnDevice(int device, CUdeviceptr ptr, size_t size) {
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Make the address accessible
    cuMemSetAccess(ptr, size, &accessDesc, 1);
}
```

虚拟内存管理提供的访问控制机制允许用户明确哪些物理内存是可以共享给系统上的其他对等设备的。如前所述，`cudaEnablePeerAccess` 强制将所有使用 `cudaMalloc` 已分配和未分配的内存映射到目标对等设备。这在许多情况下很方便，因为用户不必跟踪每个分配内存到系统中每个设备的映射状态，但是这种方法对于应用程序的性能会造成一些影响。通过分配粒度的访问控制，虚拟内存管理提供了一种机制，可以以最小的开销进行对等映射。具体可以参阅 CUDA samples 中的 vectorAddMMAP 示例，该示例展示了多设备场景下使用虚拟内存管理 API 进行内存分配、映射以及控制访问权限的具体过程。
