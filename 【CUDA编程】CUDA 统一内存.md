#! https://zhuanlan.zhihu.com/p/691326839
# 【CUDA编程】CUDA 统一内存
**写在前面**：本文主要介绍了 CUDA 统一内存的相关概念和功能，本文所有的内容全部来自官方文档，再结合笔者的理解进行阐述。如有错漏之处，请读者们务必指出，感谢！考虑到最近联系读者加 CUDA 交流群的读者较多，笔者自建了一个 CUDA 交流群，二维码将在文末附上，有兴趣的读者可以加群交流。

## 1 基本概念 
本章将介绍一种新的内存模型，**统一内存**（Unified Memory），统一内存是 CUDA 6 引入的一种新的内存模型，从 Kepler 架构开始就在硬件上受到支持，但是最初的 GPU 架构上的统一内存的编程功能相对较弱，随着 GPU 架构的不断演进，统一内存的功能也随之大大加强。本章也主要针对 Maxwell 及更高架构上的统一内存特性进行介绍，即要求计算能力不低于 5.0，至于更早的设备，有兴趣的读者可以参考产品白皮书了解。

统一内存是一种新的内存管理机制，而不是具体的硬件内存（如设备内存、主机内存），提供了一个可以由系统中任何处理器（CPU 或 GPU）访问的虚拟内存空间（姑且称之为托管内存空间，Managed Memory Space），并保证内存一致性。这意味着用户不再需要关心数据的具体存放位置，也不需要手动在主机内存和设备内存之间来回传输数据，大大降低了 CUDA 程序开发难度。

但要注意的是，即使使用了统一内存，数据移动仍然会发生，统一内存会尝试通过将数据迁移到正在访问它的处理器内存来提升性能（也就是说，如果 CPU 正在访问数据，则将数据移动到主机内存，如果 GPU 将访问数据，则将数据移动到设备内存）。数据迁移是统一内存最重要的机制，但对应用程序是不可见的，底层系统会尝试将数据放置在最有利于数据访问的位置，同时保证内存一致性。

Pascal 架构之前的设备，不支持按需将托管内存的数据细粒度移动到 GPU 内存上，因此每当启动 Kernel 时，通常必须将所有托管内存转移到 GPU 内存，以避免内存访问出错。Pascal 架构开始引入了一种新的 GPU 页面错误机制，提供了更全面的统一内存功能，结合系统范围的虚拟地址空间，页面错误机制有不少优势：首先，页面错误机制意味着 CUDA 系统软件不需要在每次启动 Kernel 时将所有托管内存映射到到 GPU 内存；如果某个 GPU 上运行的 Kernel 中访问了一个不在该 GPU 内存中的页面，就会出错，从而允许该页面按需自动迁移到该 GPU 内存中，或者，可以将页面映射到 GPU 地址空间，以便通过 PCIe 或 NVLink 互连进行访问（访问映射有时可能比迁移更快）。要注意的是，统一内存是系统范围的，即任一处理器都可以从系统中其他处理器的内存中发生故障并迁移内存页面。

在统一内存之前，还有一种零拷贝内存（页锁定内存的一种），但是零拷贝内存在分配时只会分配在主机内存上，而统一内存的实际物理内存位置却是不固定的，系统会根据实际使用需要将数据自动存放到合适的位置（可以放在主机内存，也可以放在设备内存），相比之下统一内存无论在开发效率和性能上都更具有优势。

使用统一内存进行 CUDA 编程具有如下优势：
- 统一内存可以降低开发难度。统一内存为系统中所有处理器提供了同一个统一的内存池，CPU 或 GPU 都可以通过同一个指针直接访问统一内存中的数据，不再需要手动将数据在主机与设备之间来回传输，无需用户关心数据的实际存储位置。对于多 GPU 系统，统一内存可以在多个 GPU 中直接访问，非常方便。
- 相比传统的手动移动数据，可能会提供更好的性能。统一内存底层会自动将一部分数据放置到离某个处理器更近的位置，这种就近存放的原则，可以提升整体内存带宽。
- 允许 GPU 在使用了统一内存的情况下，进行超量分配，超出 GPU 内存额度的部分可能存放在主机内存上，这也是使用统一内存最大的好处。


## 2 系统对统一内存的支持程度 
CUDA 编程模型中对统一内存的支持程度会因操作系统、GPU 架构和 CUDA 版本等硬件软件环境的不同而有所差异。因此可以将系统对统一内存的支持程度划分为多个等级，用户可以通过 `cudaGetDeviceProperties()` API 查询相关设备属性来确定当前系统对统一内存的支持级别。下面对 CUDA 统一内存的不同支持级别以及检测这些支持级别时所使用的设备属性进行介绍。

- 完全支持统一内存功能（full CUDA Unified Memory support）：即全部的统一内存分配方式、功能特性，包括系统分配和 CUDA 托管内存（Managed Memory）。此时设备属性 `pageableMemoryAccess` 的值为 1，在某些提供硬件加速的系统也会将 `hostNativeAtomicSupported`、`pageableMemoryAccessUsesHostPageTables`、`directManagedMemAccessFromHost` 等属性设置为 1。我们姑且称这种支持程度为**系统级**。
- 完全支持 CUDA 托管内存（Only CUDA Managed Memory has full support）：支持 CUDA 托管内存的全部功能，但不支持系统分配统一内存。此时设备属性 `concurrentManagedAccess` 的值为 1，`pageableMemoryAccess` 的值为 0。我们姑且称这种支持程度为**CUDA 级**。
- 部分支持 CUDA 托管内存（CUDA Managed Memory without full support）：支持访问 CUDA 分配的内存，但不允许并发访问（比如 CPU 和 GPU 同时访问）。此时设备属性 `managedMemory` 的值为 1，`concurrentManagedAccess` 的值为 0。我们姑且称这种支持程度为**准 CUDA 级**。
- 不支持统一内存：此时设备属性 `managedMemory` 值为 0。

关于系统级、CUDA 级和准 CUDA 级，这三个级别是笔者自己定义的，并非官方命名。

除了 `cudaGetDeviceProperties()` API 查询相关设备属性以外，还可以通过 `cudaDeviceGetAttribute()` API 传入具体属性枚举值查询系统对统一内存的支持级别，下面的代码展示了两种查询方式的具体流程，并以 GeForce RTX 2070 SUPER 为例给出查询结果。

```cuda
void detect_unified_memory_level()
{
    int d;
    cudaGetDevice(&d);

    int pma = 0;
    cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, d);
    printf("Full Unified Memory Support: %s\n", pma == 1? "YES" : "NO");
    
    int cma = 0;
    cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, d);
    printf("CUDA Managed Memory with full support: %s\n", cma == 1? "YES" : "NO");

    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("CUDA device properties pageableMemoryAccess: %d\n", prop.pageableMemoryAccess);
    printf("CUDA device properties hostNativeAtomicSupported: %d\n", prop.hostNativeAtomicSupported);
    printf("CUDA device properties pageableMemoryAccessUsesHostPageTables: %d\n", prop.pageableMemoryAccessUsesHostPageTables);
    printf("CUDA device properties directManagedMemAccessFromHost: %d\n", prop.directManagedMemAccessFromHost);
    printf("CUDA device properties concurrentManagedAccess: %d\n", prop.concurrentManagedAccess);
    printf("CUDA device properties pageableMemoryAccess: %d\n", prop.pageableMemoryAccess);
    printf("CUDA device properties managedMemory: %d\n", prop.managedMemory);
    printf("CUDA device properties concurrentManagedAccess: %d\n", prop.concurrentManagedAccess);
    printf("CUDA device properties managedMemory: %d\n", prop.managedMemory);
}

/*
Full Unified Memory Support: NO
CUDA Managed Memory with full support: NO
CUDA device properties pageableMemoryAccess: 0
CUDA device properties hostNativeAtomicSupported: 0
CUDA device properties pageableMemoryAccessUsesHostPageTables: 0
CUDA device properties directManagedMemAccessFromHost: 0
CUDA device properties concurrentManagedAccess: 0
CUDA device properties pageableMemoryAccess: 0
CUDA device properties managedMemory: 1
CUDA device properties concurrentManagedAccess: 0
CUDA device properties managedMemory: 1
*/
```

从查询结果可以看出，GeForce RTX 2070 SUPER 对统一内存的支持程度为：“准 CUDA 级”。

## 3 统一内存的分配方式 
根据系统对 CUDA 统一内存的支持程度，应用程序可以通过以下三种方式分配统一内存：
- 系统 API 分配：在系统级支持统一内存（full CUDA Unified Memory support）的系统上，在主机进程中任何形式分配的内存（也包括栈上的变量）都是统一内存（例如 C 语言的 `malloc()`、C++ 的 `new` 关键字、POSIX 的 `mmap` 等）。
- CUDA Runtime API 分配：通过 `cudaMallocManaged()` API 分配的内存，这个 API 的用法和 `cudaMalloc()` 类似。
- CUDA 中的 `__managed__` 变量：即使用 `__managed__` 内存空间说明符声明的变量，相当于是一个静态的统一内存分配。


### 3.1 系统 API 分配统一内存 

在系统级支持统一内存（full CUDA Unified Memory support）的系统上，所有内存都是统一内存，包括使用系统上所有内存分配 API 分配的内存，比如 C 语言的 `malloc()`、C++ 的 `new` 关键字、POSIX 的 `mmap`，甚至还包括全局变量、栈上的局部变量等。

下面给出一个一般的使用系统 API 分配统一内存的示例代码（注意这个代码只能在完全支持统一内存功能的系统上运行）：
```cuda
__global__ void printme(char *str) {
    printf(str);
}

int main() {
    // Allocate 100 bytes of memory, accessible to both Host and Device code
    char *s = (char*)malloc(100);
    // Physical allocation placed in CPU memory because host accesses "s" first
    strncpy(s, "Hello Unified Memory\n", 99);
    // Here we pass "s" to a kernel without explicitly copying
    printme<<< 1, 1 >>>(s);
    cudaDeviceSynchronize();
    // Free as for normal CUDA allocations
    cudaFree(s); 
    return  0;
}
```

另外需要注意的是，在系统级支持统一内存的系统上，虽然所有的内存都被视为统一内存，所有的变量都可以在 Kernel 中访问，但要注意一点，在 GPU 中访问堆栈变量、文件作用域和全局作用域变量时必须以指针的形式访问，考虑下面这个全局变量的例子：
```cuda
// this variable is declared at global scope
int global_variable;

__global__ void kernel_uncompilable() {
    // this causes a compilation error: global (__host__) variables must not
    // be accessed from __device__ / __global__ code
    printf("%d\n", global_variable);
}

// On systems with pageableMemoryAccess set to 1, we can access the address
// of a global variable. The below kernel takes that address as an argument
__global__ void kernel(int* global_variable_addr) {
    printf("%d\n", *global_variable_addr);
}
int main() {
    kernel<<<1, 1>>>(&global_variable);
    ...
    return 0;
}
```

在上面的示例中，用户需要确保将指向全局变量的指针传递给 Kernel，而不是直接在 Kernel 中访问全局变量 `global_variable`。这是因为没有使用 `__managed__` 说明符修饰的全局变量在默认情况下被声明为 `__host__` 变量，目前大多数编译器不允许在设备代码中直接使用`__host__` 变量。

系统分配的统一内存的实际物理内存填充时间取决于具体 API 和系统设置，可以设置为 **API 分配虚拟内存时直接填充**，也可以设置为**线程第一次访问这块内存时再填充**，也就是说这里涉及一个统一内存**延迟映射**的概念。

实际分配的物理内存的位置与实际访问这块内存的线程所运行的位置有关，具体来说，如果第一次访问这块内存的线程是 GPU 上运行的线程（比如在 Kernel 或设备函数中访问），则这块内存通常会从 GPU 显存上分配，如果第一次访问这块内存的线程是 CPU 上运行的线程（在主机代码中访问），则这块内存通常会从主机内存上分配。注意，这个**最近分配原则**仅限于 Pascal 架构或更高的架构，在 Maxwell 架构下，统一内存分配在 GPU 内存上。

除了上面的物理内存分配原则以外，CUDA 还允许用户通过 
`cudaMemAdvise` 和 `cudaMemPreftchAsync` 两个 API 给系统分配 API 提供一些提示或建议，这可能会给应用程序的性能带来增益，这些 API 的使用方法将在后面的章节介绍。

### 3.2 CUDA Runtime API 分配统一内存 

在 CUDA 级或准 CUDA 级支持统一内存的系统上，用户可以通过 CUDA Runtime API `cudaMallocManaged()` 分配统一内存，函数签名如下：
```cuda
__host__ cudaError_t cudaMallocManaged(void **devPtr, size_t size);
```

`cudaMallocManaged()` 在语法上与 `cudaMalloc()` 相同，分配的统一内存大小为 `size`，并通过指针 `devPtr` 进行访问，内存使用完毕后依然使用 `cudaFree()` API 进行内存释放。

在 CUDA 级支持统一内存的系统上，系统中的所有 CPU 和 GPU 可以同时访问 CUDA 分配的统一内存。将主机上对 `cudaMalloc()` 的调用替换为 `cudaMallocManaged()`，不会影响程序语义，另外，`cudaMallocManaged()` 是一个主机函数，在动态并行场景下，设备代码中无法调用。下面的代码示例演示了使用 `cudaMallocManaged()` API 分配统一内存的场景：
```cuda
__global__ void printme(char *str) {
    printf(str);
}

int main() {
    // Allocate 100 bytes of memory, accessible to both Host and Device code
    char *s;
    cudaMallocManaged(&s, 100);
    // Note direct Host-code use of "s"
    strncpy(s, "Hello Unified Memory\n", 99);
    // Here we pass "s" to a kernel without explicitly copying
    printme<<< 1, 1 >>>(s);
    cudaDeviceSynchronize();
    // Free as for normal CUDA allocations
    cudaFree(s); 
    return  0;
}
```

### 3.3 全局范围的 \_\_managed\_\_ 变量 

前面介绍过，统一内存还可以通过 `__managed__` 关键字修饰全局范围内的变量达到静态分配的效果，这种方式分配的统一内存和通过 `cudaMallocManaged()` API 分配的具有同样的功能。

要注意的是，在系统级支持统一内存的系统上，设备代码（Kernel 或设备函数中）无法直接访问文件范围或全局范围内通过 `__managed__` 关键字修饰的变量，但可以把变量的指针作为参数传递给 Kernel 达到访问该变量的效果，具体代码示例如下。

```cuda
__global__ void write_value(int* ptr, int v) {
    *ptr = v;    // ok
    value = 2;   // error: identifier "value" is undefined
}

// Requires CUDA Managed Memory support
__managed__ int value;

int main() {
    write_value<<<1, 1>>>(&value, 1);
    // Synchronize required
    // (before, cudaMemcpy was synchronizing)
    cudaDeviceSynchronize();
    printf("value = %d\n", value);
    return 0;
}
```

CUDA 中关键字 `__managed__` 隐式表明了这是一个设备变量，即 `__managed__` 变量与 `__managed__ __device__` 变量是等价的，但要注意的是，`__managed__` 与 `__constant__` 是互斥的，即一个 `__constant__` 不能再使用 `__managed__` 声明为统一内存变量。

在访问统一内存变量之前，必须先创建有效的 CUDA Context，如果用户没有显式创建，则 CUDA Context 会由 CUDA Runtime 合适的时候隐式创建（通常在调用第一个 Runtime API 或 Kernel 启动之前），如果主机端在调用第一个 Runtime API 或 Kernel 启动之前访问了 `__managed__` 变量并且没有显式创建 CUDA Context，则 CUDA Runtime 会在访问 `__managed__` 变量前自行创建 CUDA Context。

## 4 统一内存和零拷贝内存的适配性差异 

关于统一内存和零拷贝内存在适配性上的主要区别在于两点：

- 内存操作方面：在支持统一内存的系统上，统一内存保证所有类型的内存操作（例如原子操作等）都收到支持，相比之下，零拷贝内存支持的内存操作不全面。
- 系统方面：对于某些内存操作，支持零拷贝内存的系统要比支持统一内存的系统更多。


## 5 查询指针属性 

应用程序可以通过调用 `cudaPointerGetAttributes()` 获取指针属性，然后通过 `cudaDeviceGetAttribute` 查询指针属性值，根据属性值判断这块统一内存是怎么分配的，具体指针属性枚举值有如下几种：

- `cudaMemoyTypeManaged`：CUDA 分配的统一内存。
- `cudaMemoryTypeHost`：CUDA 分配的主机内存（使用 `cudaMallocHost()` 或 `cudaHostRegister()` 分配）。
- `cudaMemoryTypeDevice`：CUDA 分配的设备内存。
- `cudaMemoryTypeUnregistered`：CUDA 未知的内存（可能是系统 API 分配的）。


要注意的是，检查指针属性并不能判断内存的实际驻留位置（即映射的物理内存的位置），只能说明这块内存是如何分配或注册的。下面的代码示例展示了如何在运行时检测指针类型：
```cuda
char const* kind(cudaPointerAttributes a, bool pma, bool cma) {
    switch(a.type) {
    case cudaMemoryTypeHost: return pma?
      "Unified: CUDA Host or Registered Memory" :
      "Not Unified: CUDA Host or Registered Memory";
    case cudaMemoryTypeDevice: return "Not Unified: CUDA Device Memory";
    case cudaMemoryTypeManaged: return cma?
      "Unified: CUDA Managed Memory" : "Not Unified: CUDA Managed Memory";
    case cudaMemoryTypeUnregistered: return pma?
      "Unified: System-Allocated Memory" :
      "Not Unified: System-Allocated Memory";
    default: return "unknown";
    }
}

void check_pointer(int i, void* ptr) {
    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, ptr);
    int pma = 0, cma = 0, device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, device);
    cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, device);
    printf("Pointer %d: memory is %s\n", i, kind(attr, pma, cma));
}

__managed__ int managed_var = 5;

int main() {
    int* ptr[5];
    ptr[0] = (int*)malloc(sizeof(int));
    cudaMallocManaged(&ptr[1], sizeof(int));
    cudaMallocHost(&ptr[2], sizeof(int));
    cudaMalloc(&ptr[3], sizeof(int));
    ptr[4] = &managed_var;

    for (int i = 0; i < 5; ++i) check_pointer(i, ptr[i]);
    
    cudaFree(ptr[3]);
    cudaFreeHost(ptr[2]);
    cudaFree(ptr[1]);
    free(ptr[0]);
    return 0;
}
```

## 6 GPU 内存超限分配 

前面介绍过，Pascal 架构之前的设备上，统一内存分配在 GPU 内存上，所以其分配的内存大小不能超过 GPU 内存的物理大小。而从 Pascal 架构开始扩展了统一内存寻址模式，支持 49 位虚拟寻址，足以覆盖现代 CPU 的 48 位虚拟地址空间，以及 GPU 的内存大小。更大的虚拟地址空间和页面错误能力使应用程序可以访问整个系统的虚拟内存，而不受任何一个处理器的物理内存大小的限制。

这意味着应用程序可以超限分配内存：换句话说，应用程序可以分配、访问和共享大于任何一个处理器物理内存容量的数组，从而实现超大数据集的核外处理。只要整个系统有足够的内存容量（包括所有的处理器内存）可用于分配，调用 `cudaMallocManaged()` API 就不会耗尽内存。

## 7 性能提示 
为了在使用统一内存时获得最佳性能，CUDA 本身会做出一些与统一内存相关的优化举措（比如前面介绍的“最近分配原则''等），但这些自动优化措施并不一定是最佳的优化决策，因此，除了系统本身的自动优化机制以外，CUDA 还允许用户显式地通过一些 CUDA Runtime API 提供一些性能提示（Performance Hints）。这些提示可用于所有统一内存，比如 CUDA 分配的统一内存、系统 API 分配的统一内存（在``系统级”支持统一内存的系统上）等，使应用程序能够为 CUDA 提供更多信息以提高统一内存的性能。

要注意的是，这些 API 仅仅只是提示，并不影响应用程序的业务逻辑，只会影响应用程序的性能。因此，可以在任何应用程序的任何位置添加或删除对这些 API 的调用，而不会影响应用程序的执行结果。换句话说，实践是检验性能的唯一标准，应用程序只有在确保这些提示提高了性能时才应使用，而不是盲目使用。

### 7.1 数据预取 
**数据预取**（Data Prefetching），即在数据被使用之前，提前将数据迁移到对应处理器的内存中，并在处理器开始访问该数据前将其映射到该处理器的页表中。数据预取的目的是在建立数据局部性的同时避免触发页面错误机制，这极大地提升了处理器访问统一内存中数据的性能。在应用程序的运行过程中，某块数据可能会被多个处理器先后访问，因此可以在应用程序的执行流程中插入相应地预取数据操作。 

CUDA Runtime 提供了 `CudaMemPrefetchAsync()` API 用来将数据迁移到离指定处理器更近的位置，这是一个异步的流排序 API，因此迁移操作会在流中先前的工作完成后才会开始，完成迁移后才会开始后续操作。

```cuda
cudaError_t cudaMemPrefetchAsync(const void *devPtr, 
    size_t count,
    int dstDevice,
    cudaStream_t stream);
```

调用 `CudaMemPrefetchAsync()` API 后，将在给定流 `stream` 内把统一内存空间中 `[DevPtr，DevPtr+count]` 区间的数据迁移到目的地设备 `dstDevice` 的内存上，如果要迁移到主机内存上，则 `dstDevice` 参数应传入 `cudaCpuDeviceID`。下面的代码展示了 `CudaMemPrefetchAsync()` API 的调用示例：

```cuda
void test_prefetch_managed(cudaStream_t s) {
    char *data;
    cudaMallocManaged(&data, N);
    init_data(data, N);                                     // execute on CPU
    cudaMemPrefetchAsync(data, N, myGpuId, s);              // prefetch to GPU
    mykernel<<<(N + TPB - 1) / TPB, TPB, 0, s>>>(data, N);  // execute on GPU
    cudaMemPrefetchAsync(data, N, cudaCpuDeviceId, s);      // prefetch to CPU
    cudaStreamSynchronize(s);
    use_data(data, N);
    cudaFree(data);
}
```

### 7.2 数据使用提示 
当多个处理器需要同时访问相同的数据时，单独的数据预取是不够的，在这种情况下，应用程序通过 `cudaMemAdvise()` API 为 CUDA 提供接下来如何使用数据的提示，对性能提升具有重要意义。

```cuda
cudaError_t cudaMemAdvise(const void *devPtr,
    size_t count,
    enum cudaMemoryAdvise advice,
    int device);
```

`cudaMemAdvise` API 提示了接下来处理器 `dstDevice` 对统一内存空间中 `[DevPtr，DevPtr+count]` 区间的数据的访问形式。关于数据访问形式，通过 `advice` 参数进行指定，主要有如下枚举值：

- `cudaMemAdviseSetReadMostly`：这意味着数据大部分将被读取并且只是偶尔写入。这可能导会使驱动程序在处理器访问数据时在处理器内存中创建数据的只读拷贝，当这块数据上发生写入操作时，为了保持内存一致性，相应页面的只读拷贝也将失效。在宏观角度上，相当于一个只读缓存操作，有效地提升了指定数据的读操作带宽。
- `cudaMemAdviseSetPreferredLocation`：此建议将数据的首选位置设置为指定设备的内存。如果传入的设备参数为 `cudaCpuDeviceId`，会将首选位置设置为 CPU 内存。设置首选位置不会导致数据立即迁移到该位置，相反，它会在该内存区域发生页面错误时指导迁移策略。如果数据已经在它的首选位置并且故障处理器可以建立映射而不需要迁移数据，此时无需数据迁移；另一方面，如果数据不在其首选位置，或者无法建立直接映射，那么数据将被迁移到访问它的处理器。要注意的是，设置首选位置不会阻止使用 `cudaMemPrefetchAsync` 完成数据预取。
- `cudaMemAdviseSetAccessedBy`：此建议意味着数据将被指定设备频繁访问。但这不会导致数据迁移，并且对数据本身的位置没有影响，相反，只要数据的位置允许建立映射，它就会使数据始终映射到指定处理器的页表中。如果数据因任何原因被迁移，则映射也会相应更新。


上述每个数据使用提示都有对应的取消提示的参数枚举值，比如：`cudaMemAdviseUnsetReadMostly`、`cudaMemAdviseUnsetPreferredLocation` 和 `cudaMemAdviseUnsetAccessedBy`，应用程序如果在某个代码路径下需要取消之前设置的提示信息，可以传入这些枚举值重新调用 `cudaMemAdvise()` API 达到目的。

### 7.3 查询数据访问形式 
应用程序可以使用 `cudaMemRangeGetAttribute()` API 查询某个托管内存范围内通过 `cudaMemAdvise()` 或 `cudaMemPrefetchAsync()` 设置的数据访问属性，函数签名如下：
```cuda
__host__ cudaError_t cudaMemRangeGetAttribute(void* data, 
    size_t dataSize, 
    cudaMemRangeAttribute attribute, 
    const void* devPtr, 
    size_t count)
```

此外，如果用户还可以使用 `cudaMemRangeGetAttributes()` API 一次查询多个属性，具体 API 用法可以参阅 CUDA Runtime API 文档，这里不再赘述。

## 8 准 CUDA 级支持统一内存系统上的内存一致性 
前面介绍过，在某些准 CUDA 级支持统一内存的系统上，设备属性 `concurrentManagedAccess` 的值为 0，此时 CPU 和 GPU 无法同时访问托管内存，因为如果 CPU 在 GPU Kernel 执行过程中访问统一内存分配，则无法保证一致性。

### 8.1 GPU 独占托管内存的访问 

为了确保在不支持托管内存并发访问的系统上的统一内存一致性，统一内存编程模型在 CPU 和 GPU 同时执行时对数据访问施加了限制。当 GPU 上有 Kernel 在运行时，无论该 Kernel 是否使用到统一内存，GPU 都会独占所有托管内存的访问，此时不允许 CPU 再去访问统一内存（会报 Segmentation fault 错误）。除了 GPU 上运行 Kernel 的场景之外，当通过 `cudaMemcpy*()` 或 `cudaMemset*()` 等异步 API 操作托管内存时，也会限制 CPU 并发访问该数据。我们来看下面的代码：

```cuda
__global__  void  kernel() {
    if (threadIdx.x == 0) printf("blockIdx.x: %d\n", blockIdx.x);
}

void testExclusiveAccess() {
    int *temp;
    cudaMallocManaged(&temp, sizeof(int) * 1);

    kernel<<< 32, 1024 >>>();

    temp[0] = 1;

    cudaDeviceSynchronize();
    cudaFree(temp);
}
```

显然上述代码中的 Kernel 并没有使用到统一内存，但是在 Kernel 运行过程中，主机端分配并访问了统一内存。针对设备属性 `concurrentManagedAccess` 的值为 0 的系统，这段代码编译是能通过的，但是在运行过程中会报 Segmentation fault 错误。为了解决这个问题，可以在 Kernel 启动之后加一个同步操作，确保主机端操作统一内存时，GPU 是空闲状态，如下所示。

```cuda
__global__  void  kernel() {
    if (threadIdx.x == 0) printf("blockIdx.x: %d\n", blockIdx.x);
}

void testExclusiveAccess() {
    int *temp;
    cudaMallocManaged(&temp, sizeof(int) * 1);

    kernel<<< 32, 1024 >>>();
    cudaDeviceSynchronize();

    temp[0] = 1;

    cudaDeviceSynchronize();
    cudaFree(temp);
}
```

另外要注意的是，如果在 GPU 处于活动状态时主机端使用 `cudaMallocManaged()` 或 `cuMemAllocManaged()` 动态分配托管内存，则在启动其他工作或同步 GPU 之前，该内存的行为是未指定的，在此期间 CPU 尝试访问该内存可能会也可能不会报 Segmentation fault 错误，应用程序开发中应该避免该场景，比如下面的代码应该避免。 

```cuda
__global__  void  kernel() {
    if (threadIdx.x == 0) printf("blockIdx.x: %d\n", blockIdx.x);
}

void testExclusiveAccess() {
    kernel<<< 32, 1024 >>>();

    int *temp;
    cudaMallocManaged(&temp, sizeof(int) * 1);
    temp[0] = 1;

    cudaDeviceSynchronize();
    cudaFree(temp);
}
```

### 8.2 显式同步与 GPU 的逻辑活动 
注意，即使 GPU 上 Kernel 运行很快并在上述代码示例中的 CPU 访问 `temp` 之前完成，也需要进行显式同步。统一内存使用**逻辑活动**（Logical GPU Activity）来判断 GPU 是否空闲，而不是根据实际执行结果来判断。这块逻辑与 CUDA 编程模型一致，即 Kernel 可以在启动后的任何时间运行，并且不保证在主机发出同步调用之前完成。

关于同步的手段，统一内存没有明确要求，任何在逻辑上保证 GPU 完成其工作的函数调用都是有效的。包括：

- 调用 `cudaDeviceSynchronize()` 同步整个 GPU 设备；
- 当指定的流是唯一仍在 GPU 上执行的流时，也可以调用 `cudaStreamSynchronize()` 或 `cudaStreamQuery()`（仅限返回 `cudaSuccess` 时）完成逻辑上的显式同步；
- 当指定的事件之后没有设备操作的情况下，也可以调用事件同步的 API 如 `cudaEventSynchronize()` 或 `cudaEventQuery()` 进行同步；
- 甚至某些调用时会强制同步主机和设备的 API 也可以用来进行同步比如 `cudaMemcpy()` 和 `cudaMemset()` 等。


除了上述通过同步 API 进行显示同步以外，应用程序还可以通过 CUDA 流回调的机制确保不报错的前提下 CPU 访问统一内存。流回调机制会将应用程序提供的主机函数插入到流中，一旦流中排在回调函数之前的所有操作全部完成，流回调 API 指定的主机端函数就会被 CUDA 运行时所调用，在回调函数中 CPU 访问统一内存是允许的，事实上流回调也是一种 CPU 和 GPU 同步机制。

关于这个并发访问的限制，总结下来总共有以下几点需要注意：
- GPU 是否处于活动状态的判定，是基于代码逻辑判断的，并不基于实际运行结果，所以如果要确保 GPU 不处于活动状态，应该显式同步。
- 在 GPU 处于活动状态时，CPU 访问非托管内存（例如零拷贝内存）是不受限制的。
- GPU 在运行任何 Kernel 时都被认为是活动状态，无论该 Kernel 是否使用托管内存数据，此时禁止 CPU 访问统一内存。当然，设备属性 `concurrentManagedAccess` 为 1 的系统不受此限制。
- 多个 GPU 间的并发访问统一内存不受限制。
- 多个 Kernel 间的并发访问统一内存不受限制。


可以通过以下代码示例对这些特殊场景进行展示：
```cuda
int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    int *non_managed, *managed, *also_managed;
    cudaMallocHost(&non_managed, 4);    // Non-managed, CPU-accessible memory
    cudaMallocManaged(&managed, 4);
    cudaMallocManaged(&also_managed, 4);
    // Point 1: CPU can access non-managed data.
    kernel<<< 1, 1, 0, stream1 >>>(managed);
    *non_managed = 1;
    // Point 2: CPU cannot access any managed data while GPU is busy,
    //          unless concurrentManagedAccess = 1
    // Note we have not yet synchronized, so "kernel" is still active.
    *also_managed = 2;      // Will issue segmentation fault
    // Point 3: Concurrent GPU kernels can access the same data.
    kernel<<< 1, 1, 0, stream2 >>>(managed);
    // Point 4: Multi-GPU concurrent access is also permitted.
    cudaSetDevice(1);
    kernel<<< 1, 1 >>>(managed);
    return  0;
}
```

### 8.3 通过 CUDA 流管理数据可见性和 CPU+GPU 的并发访问 
前面两节介绍了设备属性 `concurrentManagedAccess` 为 0 的系统上的 CPU+GPU 并发访问统一内存的限制。这个限制是设备级的限制，即只要 GPU 处于活动状态就会限制 CPU 访问统一内存，本节将介绍一种更细粒度的托管内存控制机制，通过该机制可以缩小并发访问的限制级别，允许 CPU 并发访问。

新机制通过 CUDA 流实现，CUDA 流是 CUDA 编程模型中引入的一个概念，用于控制设备操作（Kernel 启动、内存拷贝等）的依赖性和独立性。启动到同一个流中的操作将会按顺序执行，而不同流中的操作允许并发执行，流本身描述了任务之间的独立性，再加上 CUDA 事件就可以描述任务之间的依赖性。

基于 CUDA 流的独立性特点，统一内存允许应用程序显式地将托管内存分配与 CUDA 流关联。关联之后，应用程序可以通过将 Kernel 启动到流中来指示这些 Kernel 将会使用到托管内存数据，相反，没启动到流中，就认为该 Kernel 不会使用到该托管内存的数据。通过这种关联关系，只要关联流中的所有操作都已完成，统一内存就会认为 GPU 对这块托管内存的访问已经完成，GPU 对托管内存的独占权就此结束，从而允许 CPU 访问统一内存。注意，此时 GPU 上可能还有其他流的操作，GPU 可能还是活跃状态，所以这个控制机制相当于把 GPU 对托管内存区域的独占权从设备级降为 CUDA 流级。

托管内存分配和 CUDA 流关联之后，应用程序必须保证只有流中的操作才会接触到该托管内存数据，统一内存将不再进行错误检查，此时统一内存一致性交由应用程序保证。换句话说，CUDA 将内存一致性交由用户自己负责，用户因此获得了一部分并发性能。

CUDA Runtime 提供了 `cudaStreamAttachMemAsync()` API 来关联托管内存分配和 CUDA 流。
```cuda
cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream,
    void *ptr,
    size_t length=0,
    unsigned int flags=4);
```

目前，参数 `length` 的值仅支持设置为 0 以指示某次分配的托管内存整体都与流关联，更细粒度的就不支持切分了。

`cudaStreamAttachMemAsync()` API 的逻辑相当复杂，会根据最后一个参数 `flag` 的不同而产生不同的变化。该参数有三个枚举值：`cudaMemAttachGlobal`（定义为 0）、`cudaMemAttachHost`（定义为 2）、`cudaMemAttachSingle`（定义为 4）。在 CUDA Progarmming Guide 中 `flag` 的默认值是 0，这是文档的笔误，实际默认值为 4。

当参数 `flag` 设置为默认值（`cudaMemAttachSingle`）时，参数 `stream` 有两个作用，一是将本 API 的操作启动到这个流中，这个很好理解，因为 `cudaStreamAttachMemAsync()` API 是一个异步 API，所以需要发布到一个流中；二是将首地址为 `ptr` 的托管内存分配与流 `stream` 相关联。在这种情况下，只要流中的所有操作都已完成，CPU 就能访问首地址为 `ptr` 的这块统一内存。

```cuda
__device__ __managed__ int x, y=2;
__global__  void  myKernel() {
    x = 10;
}

void testMemAttachSingle()
{
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &x);// Associate “x” with stream1.
                                          // Wait for “x” attachment to occur.
    myKernel<<< 1, 1, 0, stream1 >>>();   // Note: Launches into stream1.
    cudaStreamSynchronize(stream1);
    x = 20;                               // OK
                                          
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream1);
}
```

当参数 `flag` 设置为 2（`cudaMemAttachHost`）时，参数 `stream` 仅剩下一个作用，即将本 API 的操作启动到这个流中，事实上在 `flag` 不为 `cudaMemAttachSingle` 的场景下，参数 `stream` 都只剩下这个作用，不再具备关联内存的效果。在这种情况下，应用程序应当确保首地址为 `ptr` 的托管内存分配将只由 CPU 随时访问，而不允许 GPU 访问，即使在 GPU 活动状态下，CPU 依然可以随时访问这块内存，而 GPU 上的 Kernel 如果要强行访问这块内存，结果是不确定的，旧版本 GPU 设备中可能执行失败，但新设备中强行访问也是可以成功执行的。

```cuda
__device__ __managed__ int x, y=2;
__global__  void  myKernel() {
    x = 10;
}
  
void testMemAttachSingle()
{
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &y, 0, cudaMemAttachHost);
    myKernel<<< 1, 1, 0, stream1 >>>();   

    y = 20;                               // OK
                                          
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream1);
}
```

当参数 `flag` 设置为 0（`cudaMemAttachGlobal`）时，首地址为 `ptr` 的托管内存分配将对设备上的所有流都可见，这也是所有托管内存的默认状态，这与不调用 API 的效果是一样的，此时这块内存不支持 CPU+GPU 并发访问，必须待 GPU 完成所有操作处于非活动状态时才允许 CPU 访问。

### 8.4 在 CPU 多线程中的关联统一内存和 CUDA 流 
在多个 CUDA 流并发的应用程序中（单卡多流，或多卡多流），常见有两种实现方式：一种是 CPU 上的代码是单线程的，然后 CPU 反复在不同的流中发布传输或者计算任务；另外一种则是 CPU 上的代码是多线程的，每个线程创建一个流，并只负责该流中的任务。第一种方式非常常见，但是第二种方式可以尽可能调动 CPU 上多个核心并且将多流并发的逻辑进行简化，每个线程只需要维护一个流，只负责一个确定的任务，也有其优势所在。

本节主要针对第二种多流并发场景下，给出统一内存和 CUDA 流的关联示例，代码如下：

```cuda
// This function performs some task, in its own private stream.
void run_task(int *in, int *out, int length) {
    // Create a stream for us to use.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Allocate some managed data and associate with our stream.
    // Note the use of the host-attach flag to cudaMallocManaged();
    // we then associate the allocation with our stream so that
    // our GPU kernel launches can access it.
    int *data;
    cudaMallocManaged((void **)&data, length, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, data);
    cudaStreamSynchronize(stream);
    // Iterate on the data in some way, using both Host & Device.
    for(int i=0; i<N; i++) {
        transform<<< 100, 256, 0, stream >>>(in, data, length);
        cudaStreamSynchronize(stream);
        host_process(data, length);    // CPU uses managed data.
        convert<<< 100, 256, 0, stream >>>(out, data, length);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(data);
}
```

`run_task()` 函数在 CPU 的每个线程中都会执行，首先给每个主机线程创建一个私有的 CUDA 流，然后线程内动态分配了统一内存 `data`，通过 `cudaStreamAttachMemAsync()` API 将分配的统一内存绑定到私有流上。注意虽然这里的统一内存指针在栈上并且线程结束时也释放了，也是线程私有的，但统一内存缓冲区是整体的默认 GPU 上所有流都可见，所以调用 `cudaStreamAttachMemAsync()` 是必须的。然后通过一个流同步的方式在 GPU 任务与 CPU 任务之间显式同步，使得 CPU 等待 GPU 完成后就立刻正常使用这块统一内存。在这个例子中，只需要开头建立一次统一内存分配与流的关联关系，然后主机和设备都重复使用数据，要比在主机和设备之间反复拷贝数据简单得多。


CUDA 微信群二维码：
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/GJUG0H1sS5rQAGoVPS0HfOlc5kLpl7zuJb9tR9bRdCs12pFIV7icibJdhPxBntMaPa7ibGQ0A1iceYB3B8JCF4ODWw/640?wx_fmt=jpeg&amp;from=appmsg)