#! https://zhuanlan.zhihu.com/p/674838217
![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5oHF3cEibbKowLWTIicsud52YHsELvN5y2T376Yg6MySCndWL8AY2JZ5beKrFKPmhLKtgibIqicjibr0vw/640?wx_fmt=png&amp;from=appmsg)

# 【CUDA编程】传统 CUDA 动态并行详解（CDP1） 

**写在前面**：本文介绍了传统 CUDA 动态并行（CDP1）的执行环境、内存模型、接口、编程指南等内容。需要注意的是，从 CUDA 12.0 版本开始 CDP2 已经是默认的动态并行接口，所以对于大部分用户来说，本文内容无需了解，可以跳过，针对 CDP2 的介绍可以阅读笔者的另一篇文章。

## 1 执行环境和内存模型（CDP1） 
### 1.1 执行环境（CDP1） 

CUDA 执行模型基于线程、线程块和网格三层线程结构，网格中的线程执行的程序由 Kernel 函数定义。当调用 Kernel 时，网格的属性由执行配置参数指定，这些在 CUDA 中有专门的语法。CUDA 中对动态并行特性的支持，扩展了在新线程网格上配置、启动和同步设备上运行的线程的能力。

> 警告：父 block 与子 Kernel 的显式同步（即在设备代码中使用 `cudaDeviceSynchronize()`）在 CUDA 11.6 中被弃用，在 compute\_90+ 编译中被删除，并计划在未来的 CUDA 版本中完全删除。

#### 1.1.1 父网格与子网格（CDP1）
配置并启动新网格的设备线程属于父网格，调用创建的网格是子网格。
子网格的调用和完成是有序嵌套的，这意味着在其线程创建的所有子网格都执行完成之前，父网格也不会被认为是完成状态。即使父线程没有对子网格进行显式同步，Runtime 也会保证父子网格之间的隐式同步。
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5oHF3cEibbKowLWTIicsud52YHsELvN5y2T376Yg6MySCndWL8AY2JZ5beKrFKPmhLKtgibIqicjibr0vw/640?wx_fmt=png&amp;from=appmsg)

#### 1.1.2 CUDA 原语的作用域（CDP1）
针对动态并行，在主机端和设备端，CUDA Runtime 都提供了一个 API，用于启动 Kernel、等待启动的任务执行完毕以及用于通过 CUDA 流和事件跟踪启动之间的依赖关系。

在主机系统上，启动状态以及引用流和事件的 CUDA 原语在进程内的所有线程间共享；但是进程间独立执行，可能不共享 CUDA 对象。

在设备上也存在类似的层次结构：启动的 Kernel 和 CUDA 对象对 block 中的所有线程都可见，但在 block 之间是独立的。这意味着，例如，流可以由一个线程创建，并由同一 block 中的任何其他线程使用，但不能与任何其他 block 的线程共享。

与 CDP2 不同的是，CDP1 中设备端启动的 Kernel 和 CUDA 对象的作用域是 block 级的，而 CDP2 中是 grid 级的。

#### 1.1.3 同步（CDP1）

任何线程的 CUDA Runtime 操作，包括 Kernel 启动，对于 block 中的其他线程都是可见的。这意味着父网格中的调用线程可以在由该线程启动的子网格、block 中的其他线程或在同一 block 中创建的流上执行同步。直到 block 中所有线程的所有启动都完成后，才认为该 block 执行完成。如果一个 block 中的所有线程在所有子启动完成之前退出，将自动触发同步操作。

#### 1.1.4 流和事件（CDP1）

我们知道，通常可以使用 CUDA 流和事件控制网格启动之间的依赖关系：启动到同一流中的网格按顺序执行，事件可用于创建流之间的依赖性。在设备端同样可以通过创建的流和事件完成这个目的。

在网格中创建的流和事件仅在 block 范围内可以使用，在创建他们的 block 之外使用时行为未定义。如上所述，当 block 退出时，block 启动的所有工作都是隐式同步的；启动到流中的工作也在其内，所有依赖关系都得到了适
当的解决。在 block 范围之外修改的流上的操作行为是未定义的。

在 Kernel 中使用由主机端创建的流和事件时，具有未定义的行为，同样在子网格中使用时由父网格创建的流或事件时，行为也是未定义的。

#### 1.1.5 顺序和并发（CDP1） 
设备 Runtime 启动 Kernel 的顺序遵循 CUDA 流排序语义。在一个 block 内，所有 Kernel 启动到同一个流中都是按顺序执行的。当同一个 block 中的多个线程启动到同一个流中时，流内的顺序取决于块内的线程调度，这可以通过同步原语（如 `__syncthreads()`）来控制。

请注意，由于流由 block 内的所有线程共享，因此隐式 NULL 流也被共享。如果 block 中的多个线程启动到隐式流中，则这些启动将按顺序执行。如果多个任务需要并发，则应使用显式命名流。

动态并行使得开发人员能够更容易地在程序中实现并发性；然而，设备 Runtime 在 CUDA 执行模型中没有引入新的并发保证，无法保证设备上任意数量的不同线程块之间的并发执行。

同样地，对于父 block 及其子网格而言，同样无法确保其执行的并行性。当父 block 启动子网格时，一旦满足流依赖性并且硬件资源可用于托管子网格，子网格就可以开始执行，但不能保证在父 block 达到显式同步点（如 `cudaDeviceSynchronize()`）之前开始执行。

虽然在 CUDA 程序中很容易实现并发性，但并发实际效果可能会因设备配置、应用程序工作负载和运行时调度而异。因此，依赖不同线程块之间的任何并发都是不安全的。

#### 1.1.6 设备管理（CDP1）
设备 Runtime 不支持多 GPU；设备 Runtime 只能在其当前执行的设备上运行。但是，允许查询系统中任何支持 CUDA 的设备的属性。

### 1.2 内存模型（CDP1） 

父网格和子网格共享相同的全局内存和常量内存，但具有不同的局部内存和共享内存。

#### 1.2.1 内存连续性和一致性（CDP1）

##### 1.2.1.1 全局内存（CDP1）

父网格和子网格可以连贯地访问全局内存，但子网格和父网格之间的内存一致性不能完全保证。在子网格的执行过程中，有两个时间点的内存视图与父线程完全一致，即父网格启动子网格的时间点、当子网格线程完成时（父线程中调用同步 API）。

父线程中执行的发生在调用子网格之前的所有全局内存操作，对子网格都是可见的。在父网格完成同步后，子网格的所有内存操作对父网格都可见。

在下面的示例中，执行 `child_launch` Kernel 的子网格只能保证看到在子网格启动之前对数据所做的修改。由于父线程 $0$ 正在执行启动，子线程将与父线程 $0$ 看到的内存保持一致。由于第一次 `__syncthreads()` 的调用，子线程
将看到 `data[0]=0, data[1]=1, ..., data[255]=255`（如果没有调用 `__syncthreads()`，那么只有 `data[0]` 可以确定能被子线程看到）。当子网格返回时，线程 $0$ 可以看到其子网格中的线程所做的所有修改。只有在第二次 `__syncthreads()` 调用之后，这些修改才可以被父网格的其他线程看到。

```cuda
__global__ void child_launch(int *data) {
    data[threadIdx.x] = data[threadIdx.x]+1;
}

__global__ void parent_launch(int *data) {
    data[threadIdx.x] = threadIdx.x;

    __syncthreads();

    if (threadIdx.x == 0) {
        child_launch<<< 1, 256 >>>(data);
        cudaDeviceSynchronize();
    }

    __syncthreads();
}

void host_launch(int *data) {
    parent_launch<<< 1, 256 >>>(data);
}
```

##### 1.2.1.2 零拷贝内存（CDP1） 
零拷贝内存（Zero Copy Memory）具有与全局内存相同的一致性和一致性保证，并遵循同样的规则。Kernel 中不能直接分配或释放零拷贝内存，但可以使用从主机程序传入的指向零拷贝内存的指针。

##### 1.2.1.3 常量内存（CDP1） 
常量内存中的数据是不可变的，不能从设备端修改，即使在父子网格启动之间也是如此。也就是说，所有 `__constant__` 变量的值必须在启动之前从主机端设置。所有子 Kernel 都从各自的父 Kernel 自动继承常量内存。

CDP1 中支持从 Kernel 中获取常量内存对象的地址，并且支持将指针从父 Kernel 传递到子 Kernel 或从子 Kernel 传递到父 Kernel 。

##### 1.2.1.4 共享内存和局部内存（CDP1）
共享内存和局部内存是线程块或线程私有的，在父子网格之间不可见。当共享
内存和局部内存的变量在其所属范围之外（即在其他 block 或线程）被引用时，行为未定义，并且可能导致错误。

如果 NVIDIA 编译器可以检测到指向局部或共享内存的指针被作为参数传递给 Kernel 启动，它将尝试发出警告。在 Runtime，应用程序可以使用内部函数 `__isGlobal()` 来确定指针是否指向全局内存，因此可以安全地
传递给子 Kernel 启动。

请注意，对 `cudaMemcpy*Async()` 或 `cudaMemset*Async()` 的调用可能会调用设备上的新的子 Kernel 以保留流语义。因此，将共享或局部内存指针传递给这些 API 也是非法的，并且会返回错误。

##### 1.2.1.5 局部内存（CDP1） 
局部内存（Local Memory）作为线程的私有存储空间，在该线程之外不可见。启动子 Kernel 时将指向局部内存的指针作为启动参数传递是非法的。从子网格取消引用此类本地内存指针的结果将是未定义的。

例如以下代码中将 `x_array` 作为参数传入子 Kernel 是非法的，具有未定义的行为：

```cuda
int x_array[10];       // Creates x_array in parent's local memory
child_launch<<< 1, 1 >>>(x_array);
```
  
开发人员有时无法确定编译器会将变量放入局部内存还是寄存器，所以一般而言，传递给子 Kernel 的所有变量的存储空间都应该从全局内存堆中显式分配，使用 `cudaMalloc()`、`new()` 或通过在全局范围内声明 `__device__` 存储。例如：
  
```cuda
// Correct - "value" is global storage
__device__ int value;
__device__ void x() {
    value = 5;
    child<<< 1, 1 >>>(&value);
}
```
  
```cuda
// Invalid - "value" is local storage
__device__ void y() {
    int value = 5;
    child<<< 1, 1 >>>(&value);
}
```
  
##### 1.2.1.6 纹理内存（CDP1） 
对于纹理访问，对映射纹理的全局内存区域的写入是不连贯的。纹理内存的一致性在子网格的调用和子网格完成时强制执行。这意味着在子 Kernel 启动之前父网格对内存的写入在子内核的纹理内存访问中是可见的。与上面的“全局内存”类似，在父 Kernel 中对子网格进行同步之后，子网格对内存的写入在父网格的纹理内存访问中可见。父网格和子网格如果同时访问可能会导致数据不一致。

## 2 编程接口（CDP1） 
### 2.1 CUDA C++ 参考手册（CDP1） 

本节主要介绍针对动态并行特性对 CUDA C++ 语言扩展的更新和添加的内容。

基于 CUDA C++ 实现动态并行的 CUDA Kernel 中使用的语言接口和 API，称为设备 Runtime，与主机上的 CUDA Runtime API 基本类似。尽可能地保留了 CUDA Runtime API 的语法和语义，提升了代码的可重用性。

与 CUDA C++ 中的所有代码一样，本节介绍的 API 和代码都是线程级的，即在某个线程中使用。这使得每个线程能够针对下一步要执行的 Kernel 或操作做出唯一的、动态的决定。此外，block 内的线程在执行任何设备 Runtime API 时不需要进行同步，这使得设备 Runtime API 函数能够在任意不同的内核代码中调用，而不会出现死锁。

#### 2.1.1 设备端 Kernel 启动（CDP1）

子 Kernel 可以使用标准 CUDA `<<< >>>` 语法从设备端启动：
```cuda
kernel_name<<< Dg, Db, Ns, S >>>([kernel arguments]);
```

- `Dg`：`dim3` 类型，并指定网格（grid）的维度和大小；
- `Db`：`dim3` 类型，指定每个线程块（block）的维度和大小；
- `Ns`：`size_t` 类型，指定除了静态分配的共享内存之外，为每个线程块动态分配的共享内存字节数。`Ns` 是一个可选参数，默认为 $0$；
- `S`：`cudaStream_t` 类型，指定与此调用关联的流。流必须在同一网格中分配，即不能使用跨设备的流。`S` 是一个可选参数，默认为 $0$。

##### 2.1.1.1 子 kernel 的启动是异步的（CDP1）
与主机端启动相同，所有设备端 Kernel 启动相对于父线程都是异步的。也就是说，`<<<...>>>` 启动命令将立即返回，并且父线程将继续执行，直到到达显式同步点（例如父 Kernel 调用了 `cudaDeviceSynchronize()`）。

子网格启动被发布到设备上，并将独立于父线程执行。子网格可以在启动后的任何时间开始执行，并且不能保证在父线程到达显式启动同步点之前开始执行。

##### 2.1.1.2 启动环境配置（CDP1）

子网格的所有全局设备配置设置（例如，从 `cudaDeviceGetCacheConfig()` 返回的共享内存和 L1 缓存大小，以及从`cudaDevicesGetLimit()` 返回的设备限制）都将从父网格继承。同样，设备限制（如堆栈大小）将保持原来的配置。

对于主机启动的 Kernel，从主机端设置的每个 Kernel 的配置将优先于全局设置，这些配置也将在设备端启动子 Kernel 时被使用，设备端启动子 Kernel 时不能再重新配置 Kernel 的环境。

#### 2.1.2 流（CDP1） 

设备 Runtime 提供显式命名流（非默认流）和默认流（NULL 流）。命名流 可以由 block 中的任何线程使用，但流句柄不能传递给其他 block 或该 block 的子、父 Kernel。换句话说，流是由创建它的 block 私有，只能在该 block 中使用，在其他 block 中使用流句柄将导致未定义的行为。

与主机端启动类似，设备端启动到不同流中的 Kernel 可以支持并发运行，但不能保证实际执行过程中一定是并发的。因此，应用程序不应该基于子 Kernel 之间的并发性来实现某种功能，这种并发性在 CUDA 编程模型中是不保证的，并且具有未定义的行为。

设备不支持主机端 NULL 流的跨流屏障语义（详见下文）。为了保持与主机 Runtime 的语义兼容性，必须使用 `cudaStreamCreateWithFlags()` API 创建所有设备流，并传递 `cudaStreamNonBlocking` 标志。`cudaStreamCreate()` 是主机 Runtime 的 API，在设备端调用时将无法通过编译。

因为设备 Runtime 不支持 `cudaStreamSynchronize()` 和 `cudaStreamQuery()`，所以如果应用程序需要知道流中启动的子 Kernel 已经完成，那么应该使用 `cudaDeviceSynchronize()`。

##### 2.1.2.1 隐式（NULL）流（CDP1）

在主机程序中，隐式（NULL）流具有与其他流的额外屏障同步语义。设备 Runtime 中提供了在 block 中的所有线程之间共享的单个隐式未命名流，由于所有命名流都必须使用 `cudaStreamNonBlocking` 标志创建，因此启动到隐式流中的工作不会插入对任何其他流（包括其他 block 的空流）中挂起工作的隐式依赖。

#### 2.1.3 事件（CDP1）

动态并行中仅支持 CUDA 事件的流间同步功能。这意味着支持 `cudaStreamWaitEvent()`，但不支持 `cudaEventSynchronize()`、`cudaEventElapsedTime()` 和 `cudaEventQuery()` 等事件 API。由于不支持 `cudaEventElapsedTime()`，`cudaEvents` 必须通过 `cudaEventCreateWithFlags()` API 创建，并在创建时传递 `cudaEventDisableTiming` 标志。

与显式命名流一样，事件对象可以在创建它们的 block 内的所有线程之间共享，对于该 block 来说是本地的，并且可能不会传递给其他 Kernel。事件句柄不能保证在 block 之间是唯一的，因此在其他 block 中使用该 block 中的事件句柄将导致未定义的行为。


#### 2.1.4 同步（CDP1） 

当线程调用 `cudaDeviceSynchronize()` 函数后将同步 block 中任何线程启动的所有工作，直到其调用 `cudaDeviceSynchronize()` 为止。注意，可以从不同的代码中调用 `cudaDeviceSynchronize()`。

如果调用线程目的是与从其他线程调用的子网格同步，则可以由程序执行足够的额外线程间同步，例如通过调用 `__syncthreads()`。

##### 2.1.4.1 block 间同步（CDP1）
`cudaDeviceSynchronize()` 函数并不意味着 block 内同步。特别是，如果没有通过 `__syncthreads()` 指令进行显式同步，则调用线程无法对除自身之外的任何线程启动的工作做出任何假设。例如，如果一个 block 中的多个线程都在启动工作，并且所有这些工作都需要一次同步（可能是因为基于事件的依赖关系），则需要所有线程调用 `cudaDeviceSynchronize()` 。

子 Kernel 的启动是单独的线程的行为，而同步则是 block 整体的行为：

- 任何一个 block 中的任何一个线程，调用了 `cudaDeviceSynchronize()` 进行同步的时候，将等待该调用前的，该 block 建立的所有流中的所有子 Kernel 们执行完成。不能有选择的等待 $1$ 个或者多个（如果需要有选择的等待$1$ 个或者多个，需要单独用动态并行启动然后阻塞同步一次，然后重复这个过程即可，但这样可能会造成性能上的损失），这是为何说同步是 block 的行为的原因。
- block 中的任何一个线程调用了 `cudaDeviceSynchronize()`，都有可能导致该 block 整体（所有的线程执行状态、当前使用的寄存器数据、当前的共享内存数据）被交换到全局内存上进行状态冻结，暂停运行。也就是说，一旦使用了这唯一的同步方式，任何 $1$ 个线程将会影响当前 block 整体。

#### 2.1.5 设备管理（CDP1）

一个 Kernel 只能控制运行该 Kernel 的设备，这意味着设备 Runtime 不支持 `cudaSetDevice()` 等设备 API。从设备端获取的活动设备（从 `cudaGetDevice()` 返回）将具有与从主机系统看到的相同的设备编号。可以传入指定设备 ID 作为参数调用 `cudaDeviceGetAttribute()` API 获取关于另一个设备的信息。请注意，设备 Runtime 不提供查询全部属性的 `cudaGetDeviceProperties()` API，必须单独查询某个属性。

#### 2.1.6 内存声明（CDP1）
##### 2.1.6.1 设备内存和常量内存（CDP1）

在使用设备 Runtime 时，也可以使用文件范围内通过 `__device__` 或 `__constant__` 等内存空间说明符声明的变量，变量的内存特性不变。无论 Kernel 最初是由主机还是设备 Runtime 启动的，所有 Kernel 都可以读取或写入设备变量。同样，所有 Kernel 都具有在模块作用域中声明的 `__constant__` 变量的相同的视图。

##### 2.1.6.2 纹理内存和表面内存（CDP1）

CUDA 动态并行支持纹理和表面对象，其中纹理引用可以在主机上创建，传递给 Kernel，由该 Kernel 使用（不允许子 Kernel 使用），然后从主机端销毁。设备 Runtime 不允许从设备代码中创建或销毁纹理或表面对象，但从主机创建的纹理和表面对象可以在设备上自由使用和传递。动态创建的纹理对象总是有效的，并且可以从父内核传递给子内核。

总的来说，在动态并行场景下，对于纹理内存和表面内存有如下限制：

- 纹理和表面对象只能在主机端创建，不能在设备端创建。
- 纹理和表面对象可以给任何 Kernel 使用并且传递给其子 Kernel。
- 纹理和表面引用只能给顶层 Kernel 使用，不能给子 Kernel 使用，如果一定要使用，可能不会报错，但是会读取到错误数据。

##### 2.1.6.3 共享内存变量声明（CDP1）

在 CUDA C++ 中，共享内存可以声明为静态大小的文件范围或函数范围的变量，也可以声明为动态大小的外部变量（Kernel 中使用 `extern` 修饰），动态共享内存到的大小由 Kernel 调用者在运行时通过启动配置参数确定。这两种类型的声明在设备运行时都有效。

```cuda
__global__ void permute(int n, int *data) {
    extern __shared__ int smem[];
    if (n <= 1)
        return;
  
    smem[threadIdx.x] = data[threadIdx.x];
    __syncthreads();
  
    permute_data(smem, n);
    __syncthreads();
  
    // Write back to GMEM since we can't pass SMEM to children.
    data[threadIdx.x] = smem[threadIdx.x];
    __syncthreads();
  
    if (threadIdx.x == 0) {
        permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data);
        permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data+n/2);
    }
}
  
void host_launch(int *data) {
    permute<<< 1, 256, 256*sizeof(int) >>>(256, data);
}
```

##### 2.1.6.4 符号地址（CDP1）

设备端的符号（即使用 `__device__` 修饰的符号）可以简单地通过 & 运算符从 Kernel 中直接获取变量地址，因为所有全局范围的设备变量都在 Kernel 的可见地址空间中。这也适用于 `__constant__` 符号，在这种情况下指针引用的是只读数据，虽然说是只读的，但实际上依然是可以有技巧写入的，只是需要下次“从 Host 上”启动的 Kernel 才能生效而已。

在动态并行场景下，鉴于可以直接通过 & 运算符取符号地址，于是设备端的 CUDA Runtime API（例如 `cudaMemcpyToSymbol()` 或 `cudaGetSymbolAddress()`）就取消了相关的函数。

#### 2.1.7 API 错误和启动失败（CDP1）

与主机端 CUDA Runtime 一样，任何 API 函数都可能返回错误代码，最后返回的错误代码被记录下来，并且可以通过调用 `cudaGetLastError()` 来检索。每个线程都会记录错误码，以便每个线程都可以获得最近的报错信息。错误码的类型为 `cudaError_t`。

与主机端启动类似，设备端启动可能会因多种原因而失败（无效参数等）。用户必须调用 `cudaGetLastError()` 来确定启动是否报错，但是启动后没有报错并不意味着子内核成功完成。

对于设备端异常，例如访问无效地址，子网格中的错误将返回给主机，而不是由父调用 `cudaDeviceSynchronize()` 返回。

##### 2.1.7.1 Kernel 启动 API（CDP1）

Kernel 启动是通过设备 Runtime 库公开的系统级机制，因此也可以通过底层 `cudaGetParameterBuffer()` 和 `cudaLaunchDevice()` API 直接从 PTX 启动。CUDA 应用程序可以自己调用这些底层 API 进行 Kernel 启动，具体要求与 PTX 相同，在任何情况下，用户都负责根据规范以正确的格式正确填充所有必要的数据结构进行 Kernel 启动，这些数据结构保证了向后兼容性。

与主机端启动一样，设备端使用操作符 `<<<...>>>` 启动时其实也是映射到底层的启动 API，只是这一步是由编译期自行转换的。设备端通过底层 API 启动需要涉及到两部分：

- 如何获取一个为 Kernel 启动所准备的参数缓冲区，然后在这个缓冲区中，按照一种特定的方式填充上参数，即 `cudaGetParameterBuffer()` API 要实现的内容。
- 用这个缓冲区，外加特定的启动配置（例如启动形状、动态共享内存大小）来启动特定的 Kernel，即通过 `cudaLaunchDevice()` 启动。


可见，这个实际上就是 `<<<Configuration...>>>(args...)` 的方式，只是被拆分成两部分了。

这些启动函数的 API 与 CUDA Runtime API 不同，定义如下：
```cuda
extern   device   cudaError_t cudaGetParameterBuffer(void **params);
extern __device__ cudaError_t cudaLaunchDevice(void *kernel,
                                        void *params, dim3 gridDim,
                                        dim3 blockDim,
                                        unsigned int sharedMemSize = 0,
                                        cudaStream_t stream = 0);
```

#### 2.1.8 API 手册（CDP1）

本节详细介绍了设备 Runtime 支持的 CUDA Runtime API 部分。主机和设备运行时 API 具有相同的语法，除非另有说明，否则语义也是相同的。下表提供了与主机可用版本相关的 API 概览。

| **Runtime API** | **注意点** |
|---|---|
| `cudaDeviceGetCacheConfig` |  |
| `cudaDeviceGetLimit` | 最后一个错误是每个线程的状态，而不是每个 block 的状态 |
| `cudaGetLastError` |  |
| `cudaGetErrorString` |  |
| `cudaGetDeviceCount` |  |
| `cudaDeviceGetAttribute` | 将返回任何 GPU 设备的某个指定属性 |
| `cudaGetDevice` | 返回从主机中可以看到的当前设备 ID |
| `cudaStreamCreateWithFlags` | 必须传递 `cudaStreamNonBlocking` 标识 |
| `cudaStreamDestroy` |  |
| `cudaStreamWaitEvent` |  |
| `cudaEventCreateWithFlags` | 必须传递 `cudaEventDisableTiming` 标识 |
| `cudaEventRecord` |  |
| `cudaEventDestroy` |  |
| `cudaFuncGetAttributes` |  |
| `cudaMemcpyAsync` | 关于所有 Memcpy、Memset 函数的说明：仅支持异步 Memcpy、Memset 函数；仅允许设备到设备的 Memcpy；不能传入局部内存或共享内存指针。 |
| `cudaMemcpy2DAsync` |  |
| `cudaMemcpy3DAsync` |  |
| `cudaMemsetAsync` |  |
| `cudaMemset2DAsync` |  |
| `cudaMemset3DAsync` |  |
| `cudaRuntimeGetVersion` |  |
| `cudaMalloc` | 不能对主机上创建的指针调用设备上的 `cudaFree`，反之亦然 |
| `cudaFree` |  |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |  |
| `cudaOccupancyMaxPotentialBlockSize` |  |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMem` |  |

### 2.2 设备端通过 PTX 启动（CDP1） 

本节适用于以并行线程执行（PTX）为目标并计划在其语言中支持动态并行的编程语言和编译器实现者，提供了与 PTX 级别的 Kernel 启动相关的底层详细信息。

#### 2.2.1 Kernel 启动 API

可以通过以下两个可从 PTX 访问的 API 来实现设备端的 Kernel 启动：`cudaLaunchDevice()` 和 `cudaGetParameterBuffer()`。首先通过调用 `cudaGetParameterBuffer()` 获得参数缓冲区，然后调用 `cudaLaunchDevice()` 使用上面的参数缓冲区启动指定的 Kernel，并将参数填充到启动的 Kernel。参数缓冲区可以为 `NULL`，即，如果启动的 Kernel 不带任何参数，则无需调用 `cudaGetParameterBuffer()`。

##### 2.2.1.1 cudaLaunchDevice（CDP1）

在 PTX 级别，`cudaLaunchDevice()` 需要在使用前通过以下两种形式之一声明：

```cuda
// PTX-level Declaration of cudaLaunchDevice() when .address_size is 64
.extern .func(.param .b32 func_retval0) cudaLaunchDevice
(
  .param .b64 func,
  .param .b64 parameterBuffer,
  .param .align 4 .b8 gridDimension[12],
  .param .align 4 .b8 blockDimension[12],
  .param .b32 sharedMemSize,
  .param .b64 stream
)
;

// PTX-level Declaration of cudaLaunchDevice() when .address_size is 32
.extern .func(.param .b32 func_retval0) cudaLaunchDevice
(
  .param .b32 func,
  .param .b32 parameterBuffer,
  .param .align 4 .b8 gridDimension[12],
  .param .align 4 .b8 blockDimension[12],
  .param .b32 sharedMemSize,
  .param .b32 stream
)
;
```

下面的 CUDA 级声明被映射到上述 PTX 级声明之一，可在系统头文件 cuda\_device\_runtime\_api.h 中找到。该函数在 cudadevrt 系统库中定义，必须与程序链接才能使用设备端 Kernel 启动功能。

```cuda
// CUDA-level declaration of cudaLaunchDevice()
extern "C" __device__
cudaError_t cudaLaunchDevice(void *func, void *parameterBuffer,
                              dim3 gridDimension, dim3 blockDimension,
                              unsigned int sharedMemSize,
                              cudaStream_t stream);
```

第一个参数是指向要启动的 kernel 的指针，第二个参数是保存已启动 kernel 的实际参数的参数缓冲区。关于参数缓冲区的布局在下面会进行详细说明。其他参数指定了启动配置，即 grid 维度、block 维度、共享内存大小以及启动关联的流等。

##### 2.2.1.2 cudaGetParameterBuffer（CDP1）

在 PTX 级别，`cudaGetParameterBuffer()` 需要在使用前通过以下两种形式之一声明，具体取决于地址大小：
```cuda
// PTX-level Declaration of cudaGetParameterBuffer() when .address_size is 64
.extern .func(.param .b64 func_retval0) cudaGetParameterBuffer
(
  .param .b64 alignment,
  .param .b64 size
)
;

// PTX-level Declaration of cudaGetParameterBuffer() when .address_size is 32
.extern .func(.param .b32 func_retval0) cudaGetParameterBuffer
(
  .param .b32 alignment,
  .param .b32 size
)
;
```

下面的 CUDA 级声明被映射到上述 PTX 级声明：

```cuda
// CUDA-level Declaration of cudaGetParameterBuffer()
extern "C" __device__
void *cudaGetParameterBuffer(size_t alignment, size_t size);
```

第一个参数指定参数缓冲区的对齐要求，第二个参数指定大小要求（以字节为单位）。在当前实现中，`cudaGetParameterBuffer()` 返回的参数缓冲区始终保证为 $64$ 字节对齐，忽略对齐要求参数。但是，建议将正确的对齐要求值（即要放置在参数缓冲区中的任何参数的最大对齐）传递给 `cudaGetParameterBuffer()` 以确保代码将来的可移植性。

#### 2.2.2 参数缓冲区布局（CDP1）

参数缓冲区中的参数是被禁止重新排序的，并且要求每个参数内存对齐。也就是说，每个参数必须放置到自己内存大小的整数倍。例如一个 $16$ 字节的参数（如 `double2`），必须放置到 $16$ 字节的边界上，即在参数缓冲区中的位置必须要能被 $16$ 整除。参数缓冲区的最大大小为 $4$ KB。以下面的 Kernel 为例：

```cuda
__global__ void your_kernel(uint8_t a, double b);
```

那么实际上在参数缓冲区中的布局是：第 $1$ 个字节存放 `a`，第 $8$ 个字节存放 `b`，中间的 $7$ 个字节都是空位，有点类似于 C++ 中对象的内存布局。

### 2.3 动态并行的 Toolkit 支持（CDP1） 
#### 2.3.1 在 CUDA 代码中包含设备 Runtime API（CDP1）

与主机 Runtime API 类似，CUDA 设备 Runtime API 的模型会在程序编译期间自动包含在内，应用程序无需显式包含 cuda\_device\_runtime\_api.h 头文件。

#### 2.3.2 编译和链接（CDP1）

在使用动态并行特性的时候，当使用 nvcc 编译和链接 CUDA 程序时，程序将自动链接到静态设备 Runtime 库 libcudadevrt。

设备 Runtime 以静态库（Windows 上的 cudadevrt.lib，Linux 下的 libcudadevrt.a）的形式提供，使用设备 Runtime 的 GPU 应用程序必须链接到该静态库。设备库的链接可以通过 nvcc 或 nvlink 完成。下面展示了两个简单的示例。

如果可以从命令行指定所有必需的源文件，则可以在一个步骤中编译和链接设备 Runtime 程序：

```cuda
$ nvcc -arch=sm_75 -rdc=true hello_world.cu -o hello -lcudadevrt
```

也可以先将 CUDA 程序的 .cu 源文件编译为目标文件，然后将它们链接在一起：

```cuda
$ nvcc -arch=sm_75 -dc hello_world.cu -o hello_world.o
$ nvcc -arch=sm_75 -rdc=true hello_world.o -o hello -lcudadevrt
```

总的来说，本章节给出的头文件和库均不需要用户专门显式地包含和指定，全部都会在编译时自动加上。唯一需要注意的是 `rdc` 编译选项（设备端代码重定位），这个还是需要的，当然这是对于命令行编译的用户说的。

对于常见的 Windows 上的 VS 用户，如果是使用的默认的安装 CUDA 时候的 CUDA 自定义模板，直接在 Solution Explorer 里面，右键属性中，选择 `rdc` 打开即可，这个非常简单。

## 3 编程指南（CDP1） 
### 3.1 基本原则（CDP1） 
设备 Runtime 是主机 Runtime 的功能子集，提供了 API 级设备管理、Kernel 启动、设备 memcpy、流管理和事件管理等功能。

设备 Runtime API 的语法和语义与主机 Runtime API 的语法和语义基本相同，关于一些例外情况，本章节前面也进行了介绍。

以下示例展示了一个使用动态并行特性的简单 Hello World 程序：

```cuda
#include <stdio.h>

__global__ void childKernel()
{
    printf("Hello ");
}

__global__ void tailKernel()
{
    printf("World!\n");
}

__global__ void parentKernel()
{
    // launch child
    childKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError()) {
        return;
    }

    // launch tail into cudaStreamTailLaunch stream
    // implicitly synchronizes: waits for child to complete
    tailKernel<<<1,1,0,cudaStreamTailLaunch>>>();

}

int main(int argc, char *argv[])
{
    // launch parent
    parentKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError()) {
        return 1;
    }

    // wait for parent to complete
    if (cudaSuccess != cudaDeviceSynchronize()) {
        return 2;
    }

    return 0;
}
```

以上程序可以从命令行一行代码进行构建，如下所示：
```cuda
$ nvcc -arch=sm_75 -rdc=true hello_world.cu -o hello -lcudadevrt
```

### 3.2 性能（CDP1） 
#### 3.2.1 同步（CDP1）
一个线程的同步可能会影响同一 block 中其他线程的性能，即使这些其他线程自己不调用 `cudaDeviceSynchronize()` 也是如此。这种影响将取决于底层实现。通常，与显式调用 `cudaDeviceSynchronize()` 相比，在 block 结束时完成子 Kernel 的隐式同步效率更高。因此，建议仅在需要在 block 结束之前与子 Kernel 同步的场景下调用 `cudaDeviceSynchronize()`。

#### 3.2.2 启用动态并行的 Kernel 开销（CDP1）

在控制动态启动时，处于活动状态的系统软件可能会给当时正在运行的任何 Kernel 施加开销，无论它是否调用自己的 Kernel 启动。这种开销源于设备 Runtime 的执行跟踪和软件管理，这些行为可能导致性能下降。通常，链接到设备 Runtime 库的应用程序会产生这种开销。

### 3.3 限制和约束（CDP1） 

本章中介绍的所有动态并行相关的功能，可能会由于某些硬件和软件资源的限制，导致在使用设备 Runtime 时，程序的规模、性能和其他属性也受到限制。

#### 3.3.1 Runtime（CDP1） 
##### 3.3.1.1 内存占用（CDP1）

设备 Runtime 系统软件会出于各种管理需要而预留一些内存，特别是用于在同步期间保存父网格状态的内存，以及用于跟踪待启动 Kernel 的预留内存。配置控制可用于减少此预留内存的大小，以满足某些启动限制。

大多数预留内存会在父 Kernel 同步子网格时，被用于存储父 Kernel 的整体状态。保守地说，该内存大小必须满足为设备上所有可能存在的活动线程存储状态。这意味着在启动链上，每个由于同步而被“冻结”在显存上的父层，可能需要多达 $860$ MB 的设备内存，具体取决于设备配置，即使这些内存没有全部消耗，也无法供程序使用。可见，随着动态并行同步深度增加，显存消耗也是巨大的。

另外，设备 Runtime 系统软件本身会维持一个需要被动态启动的子 Kernel 列表，这个列表本身也占用空间，需要在使用动态并行的时候预留出来，这个默认的大小大约是 $2000$ 个左右等待启动的子 Kernel。

##### 3.3.1.2 嵌套和同步深度（CDP1）

动态运行场景下，一个 Kernel 可能会启动另一个 Kernel，而该 Kernel 可能会启动另一个 Kernel，以此类推。每个从属启动都被认为是一个新的嵌套层级，层级总数就是程序的**嵌套深度**。**同步深度**被定义为程序在子启动时显式同步的最深级别。通常这比程序的嵌套深度小 $1$，但如果程序不需要在所有级别调用 `cudaDeviceSynchronize()`，则同步深度远小于嵌套深度。

总体最大嵌套深度限制为 $24$，但实际上，真正限制的系统为每个新层级预留的内存量（用于同步时存储父 Kernel 状态），一旦超出限制将会启动失败。请注意，这也可能适用于 `cudaMemcpyAsync()`，异步拷贝底层也可能会进行 Kernel 启动。

默认情况下，为两级同步保留足够的存储空间。这个最大同步深度（以及因此保留的存储）可以通过调用 `cudaDeviceSetLimit()` API 并指定 `cudaLimitDevRuntimeSyncDepth` 参数来控制。必须在主机启动首层 Kernel 之前配置要支持的层数，以保证嵌套程序的成功执行。如果调用 `cudaDeviceSynchronize()` 时实际同步深度大于指定的最大同步深度，将返回错误。

在父 Kernel 从不调用 `cudaDeviceSynchronize()` 的情况下，如果系统检测到不需要为父 Kernel 状态保留空间，则允许进行优化。在这种情况下，由于永远不会发生显式父/子同步，因此程序所需的内存占用量将远小于保守的最大值。这样的程序可以指定较小的最大同步深度，以避免过度分配预留内存。

##### 3.3.1.3 待启动 Kernel（CDP1） 

启动 Kernel 时，设备 Runtime 系统软件将跟踪所有相关的配置和参数，直到 Kernel 完成，这些数据存储在系统管理的启动池中，这个启动池就是前面说的待启动的子 Kernel 列表。

启动池分为固定大小的池和性能较低的虚拟化池。 设备运行时系统软件将首先尝试跟踪固定大小池中的启动数据。当固定大小的启动池已满时，虚拟化池将用于跟踪新的启动。

如果要调整固定大小启动池的大小，可以通过从主机端调用 `cudaDeviceSetLimit()` API 并指定 `cudaLimitDevRuntimePendingLaunchCount` 参数进行配置。

##### 3.3.1.4 配置选项（CDP1）

设备 Runtime 系统软件的资源分配可以由主机程序通过 `cudaDeviceSetLimit()` API 控制，需要注意的是，该设置必须在启动任何 Kernel 之前完成，并且在 GPU 运行程序时不得更改此资源限制。

下面列出了一些可以设置的资源限制属性，详细介绍如下：

- `cudaLimitDevRuntimeSyncDepth`：设置可以调用 `cudaDeviceSynchronize()` 的最大深度。实际启动深度可以大于这个值，但是超过这个限制的显式同步将返回 `cudaErrorLaunchMaxDepthExceed` 错误。默认的最大同步深度为 $2$。
- `cudaLimitDevRuntimePendingLaunchCount`：控制为待启动 Kernel 的启动、由于不满足依赖关系或缺乏执行资源而尚未开始执行的事件而预留的内存量。当此预留内存用完时，设备 Runtime 系统软件将尝试在较低性能的虚拟化池中跟踪新的待启动 Kernel。如果虚拟化池也已满，即当所有可用堆空间都被耗尽时，将不会启动，并且线程的最后一个错误将被设置为 `cudaErrorLaunchPendingCountExceed`。启动槽的默认数量为 $2048$。
- `cudaLimitStackSize`：控制每个 GPU 线程的堆栈大小（以字节为单位）。CUDA 驱动程序会根据需要自动增加每次 Kernel 启动的单个线程的堆栈大小。每次启动后，此大小不会重置回原始值。要将每个线程堆栈的大小设置为不同的值，可以调用 `cudaDeviceSetLimit()` 来设置此限制，设置完成后立即生效，堆栈将立即调整大小。在有些情况下，设备将锁定，直到所有先前请求的任务完成后再进行调整。可以调用 `cudaDeviceGetLimit()` 来获取当前单个线程的堆栈大小。


##### 3.3.1.5 内存分配与生命周期（CDP1）

`cudaMalloc()` 和 `cudaFree()` 两个 API 在主机和设备环境之间具有不同的语义。当从主机调用时，`cudaMalloc()` 从当前未使用的设备内存中分配一个新区域。当从设备 Runtime 调用时，这两个 API 分别映射到设备端的 `malloc()` 和 `free()` API，这意味着在设备环境中，总的可分配内存被限制为设备 `malloc()` 堆的大小（上面介绍的 `cudaLimitStackSize`），这个堆的大小可能小于当前可用的未使用设备内存。此外，在设备端使用 `cudaMalloc()` 分配的内存指针不可以从主机端调用 `cudaFree()` 进行释放，反之亦然。

总的来说，设备端的内存分配不能从主机端释放，主机端的内存分配也不能从设备端释放。主机端和设备端的内存分配可以看成是两个独立的堆，每个堆都有自己的对应的堆管理函数，不能混淆使用，两个堆的最小分配粒度和对齐大小也不相同。

##### 3.3.1.6 SM Id 与 Warp Id（CDP1）

请注意，在 PTX 中，`%smid` 和 `%warpid` 被定义为 `volatile` 值。在动态并行场景下，设备 Runtime 可以将 block 重新调度到不同的 SM 上，以便更有效地管理资源。因此，依赖 `%smid` 或 `%warpid` 在线程或 block 的生命周期内保持不变是不安全的。这里的 `%smid` 和 `%warpid`，都需要通过 PTX 来访问，在 CUDA C++ 里面没有直接导出这两个属性，因此本节主要是面向是 PTX 用户的，对于不写 PTX 代码的用户可以忽略。

##### 3.3.1.7 ECC 错误 

CUDA Kernel 内的代码无法收到 ECC 错误的通知，也就是说子 Kernel 的 ECC 错误是不会报告给父 Kernel 的。一旦完成了整个启动树并且存在 ECC 错误，从主机端启动的最初的父 Kernel 和它的所有子 Kernel 将被作为一个整体向主机端反馈。在执行嵌套程序期间出现的任何 ECC 错误都将生成异常或继续执行（取决于具体错误和配置）。


