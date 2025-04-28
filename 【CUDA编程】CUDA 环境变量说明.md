#! https://zhuanlan.zhihu.com/p/689788980

# 【CUDA编程】CUDA 环境变量说明
**写在前面**：本文主要介绍了 CUDA 编程模型中的环境变量的作用，本文所有的内容全部来自官方文档，再结合笔者的理解进行阐述。如有错漏之处，请读者们务必指出，感谢！

CUDA 编程模型中的环境变量在 CUDA 应用程序开发中具有重要作用，如需使用，需要在程序运行前在系统中进行设置。

## 1 设备枚举和设备属性相关 

### 1.1 CUDA\_VISIBLE\_DEVICES
环境变量 `CUDA_VISIBLE_DEVICES` 用于指定 CUDA 应用程序将在哪些 GPU 设备上运行，通常用于控制程序在多 GPU 系统上的 GPU 使用情况，对于单 GPU 系统和纯主机代码的程序没有意义。通过设置 `CUDA_VISIBLE_DEVICES`，可以限制应用程序访问的 GPU 设备，以便在多任务或多用户环境中更好地管理和分配 GPU 资源。

`CUDA_VISIBLE_DEVICES` 的值是一个以英文逗号分隔的 GPU  设备索引表，例如 `0,1,2`。这表示应用程序将只能在索引为 0、1、2 的 GPU 设备上运行，而忽略其他 GPU 设备。如果用户没有显式设置 `CUDA_VISIBLE_DEVICES` 的值，应用程序将默认使用所有可用的 GPU 设备。要注意的是，除了使用整数索引以外，还支持使用 UUID 字符串的方式设置，UUID 字符串可以通过 `nvidia-smi -L` 获取，例如：
```cuda
user-xxxx@machine-xxxx:~$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 2070 SUPER (UUID: GPU-dcf83300-9db0-fa07-4b37-237b8a4eea0e)
```

上面的 UUID 字符串比较长，方便起见，也支持前缀匹配模式，即只要开头字符串能唯一匹配上即可。如果系统上没有其他的 GPU 设备的 UUID 字符串前几位也是 `GPU-dcf83300`，那么使用 `CUDA_VISIBLE_DEVICES=GPU-dcf83300` 进行设置，也是有效的。

在设置 `CUDA_VISIBLE_DEVICES` 时，只有其索引出现在 `nvidia-smi -L` 序列中的设备才可以设置，如果其中一个索引无效，则 CUDA 应用程序只能看到在无效索引之前的设备。例如将 `CUDA_VISIBLE_DEVICES` 设置为 `0,1,-1,2` 将导致设备 0 和 1 可见，设备 2 不可见。

环境变量 `CUDA_VISIBLE_DEVICES` 的主要用途有以下几个场景：
- 资源管理：在多用户或共享 GPU 资源的环境中，可以通过设置 `CUDA_VISIBLE_DEVICES` 以避免冲突和资源争夺。不同的任务可以限制在不同的 GPU 上运行，以确保资源的有效使用。
- 分布式训练：在深度学习中，分布式训练通常涉及多个 GPU 设备。通过设置 `CUDA_VISIBLE_DEVICES` 可以控制哪些 GPU 设备将用于训练。
- 调试和测试：在调试或测试程序时，可以选择一个或一组 GPU 设备，以加速代码迭代和问题排查。


### 1.2 CUDA\_MANAGED\_FORCE\_DEVICE\_ALLOC
环境变量 `CUDA_MANAGED_FORCE_DEVICE_ALLOC` 主要用于统一内存（Unified Memory）编程中，可以设置为 0 或 1，若不进行设置，系统默认为 0。如果设置为非零值，则强制要求 CUDA 驱动程序在分配统一内存时始终使用设备内存进行分配。这样的话，统一内存的将不再具有超量分配的优势。

### 1.3 CUDA\_DEVICE\_ORDER
当系统中有多个 GPU 设备时，有时候在程序开发中会发现我们使用的某个 GPU 设备的索引和 `nvidia-smi` 显示的并不相同，比如某个 GPU 设备明明在 `nvidia-smi` 中显示的序号为 0，实际用 API 获取到的索引却是 1，这是由于两者对于 GPU 的排序方式有所不同。

环境变量 `CUDA_DEVICE_ORDER` 主要用于指定 CUDA 程序中 GPU 设备的排列顺序，可以设置为 `FASTEST_FIRST` 和 `PCI_BUS_ID` 两种，系统默认为 `FASTEST_FIRST`。如果设置为 `FASTEST_FIRST`，则要求 CUDA 按照 GPU 设备性能从快到慢的顺序枚举；如果设置为 `PCI_BUS_ID`，则要求 CUDA 按照 PCI 总线 ID 升序排列。

## 2 编译相关
### 2.1 CUDA\_CACHE\_DISABLE
在 \ref{sec:PI-CWN-CW-Just-in-Time-Compilation}即时编译 中介绍过，当设备驱动程序为某些应用程序实时编译一些 PTX 代码时，它会自动缓存生成二进制代码的副本，以避免在应用程序的后续调用中重复编译。即时编译的缓存在设备驱动程序升级时自动失效，此时应用程序将使用设备驱动程序中内置的新即时编译器进行重新编译。

即时编译的缓存可以通过环境变量 `CUDA_CACHE_DISABLE` 控制是否启用。如果设置为 1，则表示禁用缓存，此时即时编译过程中不会向缓存添加二进制代码或从缓存中检索二进制代码；如果设置为 0，则表示启用缓存，系统默认设置为 0。

### 2.2 CUDA\_CACHE\_PATH
环境变量 `CUDA_CACHE_PATH` 用于指定即时编译器缓存二进制代码的目录，默认值为：
- Windows 系统：`%APPDATA%\NVIDIA\ComputeCache`；
- Linux 系统：`~/.nv/ComputeCache`。


### 2.3 CUDA\_CACHE\_MAXSIZE
环境变量 `CUDA_CACHE_MAXSIZE` 用于指定即时编译器使用的缓存大小（以字节为单位）。超过 `CUDA_CACHE_MAXSIZE` 后，较旧的二进制代码将从缓存中清除，以便在需要时为较新的二进制代码腾出空间。

`CUDA_CACHE_MAXSIZE` 必须设置为整数，以字节为单位，最大设置为 4294967296（4 GB），默认设置为：

- PC 端和服务器：1073741824（1 GB）；
- 嵌入式平台：268435456（256 MB）。


### 2.4 CUDA\_FORCE\_PTX\_JIT
环境变量 `CUDA_FORCE_PTX_JIT` 设置为 1 时，强制设备驱动程序忽略应用程序中嵌入的任何二进制代码（参阅 \ref{sec:Application-Compatibility}应用程序兼容性），并即时编译嵌入的 PTX 代码，如果 Kernel 没有嵌入 PTX 代码，它将装载失败；设置为 0 时正常进行即时编译流程，系统默认设置为 0。

此环境变量可用于验证 PTX 代码是否嵌入到应用程序中，以及即时编译是否按预期工作，以保证应用程序与未来体系结构的向前兼容性。

### 2.5 CUDA\_DISABLE\_PTX\_JIT
环境变量 `CUDA_DISABLE_PTX_JIT` 设置为 1 时，禁用嵌入 PTX 代码的即时编译，并使用应用程序中嵌入的兼容二进制代码。如果 Kernel 没有嵌入二进制代码，或者嵌入的二进制代码是为不兼容的体系结构编译的，它将装载失败；设置为 0 时正常进行即时编译流程，系统默认设置为 0。

此环境变量可用于验证应用程序是否具有为每个 Kernel 生成的兼容 SASS 代码（参阅 \ref{sec:PI-CWN-Binary-Compatibility}二进制兼容性）。

### 2.6 CUDA\_FORCE\_JIT
环境变量 `CUDA_FORCE_JIT` 设置为 1 时，强制设备驱动程序忽略应用程序中嵌入的任何二进制代码（参阅 \ref{sec:Application-Compatibility} 应用程序兼容性），并改为即时编译嵌入的 PTX 代码。如果 Kernel 没有嵌入 PTX 代码，它将装载失败；设置为 0 时正常进行即时编译流程，系统默认设置为 0。

此环境变量可用于验证 PTX 代码是否嵌入到应用程序中，以及即时编译是否按预期工作，以保证应用程序与未来体系结构的向前兼容性。

该环境变量设置为 1 时，如果环境变量 `CUDA_FORCE_PTX_JIT` 设置为 0，则以后者为准。

### 2.7 CUDA\_DISABLE\_JIT
环境变量 `CUDA_DISABLE_JIT` 设置为 1 时，禁用嵌入 PTX 代码的即时编译，并使用应用程序中嵌入的兼容二进制代码。如果 Kernel 没有嵌入二进制代码，或者嵌入的二进制代码是为不兼容的体系结构编译的，它将装载失败；设置为 0 时正常进行即时编译流程，系统默认设置为 0。

此环境变量可用于验证应用程序是否具有为每个 Kernel 生成的兼容 SASS 代码。

该环境变量设置为 1 时，如果环境变量 `CUDA_DISABLE_PTX_JIT` 设置为 0，则以后者为准。

## 3 执行相关
### 3.1 CUDA\_LAUNCH\_BLOCKING
环境变量 `CUDA_LAUNCH_BLOCKING` 用于控制程序运行时是否禁用异步启动模式。系统默认设置为 0，此时主机端在启动 Kernel 后会异步执行主机端的代码，不会等待 Kernel 在设备上运行完毕；如果设置为 1，则表示禁用异步启动模式，此时主机端在启动 Kernel 后会等待 Kernel 执行完毕才会继续执行后面的主机代码。

### 3.2 CUDA\_DEVICE\_MAX\_CONNECTIONS
在多 GPU 系统下，可以使用环境变量 `CUDA_DEVICE_MAX_CONNECTIONS` 来控制主机端并行连接的设备（计算能力不小于 3.5）的数量，可以设置为 1 至 32 的整数，由于每个链接都需要消耗额外的内存和资源，所以系统默认设置为 8。

对于多流并行的任务，如果 CUDA 流的数量超过了硬件连接的数量，多个流会共享相同的硬件工作队列，可能产生虚假依赖。

### 3.3 CUDA\_AUTO\_BOOST
环境变量 `CUDA_LAUNCH_BLOCKING` 用于设置是否开启 GPU 自动增强模式，在自动增强模式下 GPU 性能会功率、热量和利用率允许的情况下进一步提高。可以设置为 0 或 1，1 表示启用，0 表示不启用。

是否启用自动增强模式也可以通过 `nvidia-smi` 的选项 `–auto-boost-default` 进行设置，但是优先级低于环境变量，同时设置的情况下会被 `CUDA_LAUNCH_BLOCKING` 覆盖。

## 4 cuda-gdb 相关（Linux 系统）
### 4.1 CUDA\_DEVICE\_WAITS\_ON\_EXCEPTION
环境变量 `CUDA_LAUNCH_BLOCKING` 用于设置 Linux 系统上的调试行为，可以设置为 0 或 1。系统默认设置为 0；当设置为 1 时，如果设备端代码运行错误，CUDA 应用程序将停止，允许附加调试器进行进一步调试。

## 5 MPS 服务相关（Linux 系统）
### 5.1 CUDA\_DEVICE\_DEFAULT\_PERSISTING\_L2\_CACHE\_PERCENTAGE\_LIMIT
前面介绍过，具有计算能力 8.x 的设备允许预留出一部分 L2 高速缓存用于持久存储对全局内存的数据访问。当使用多进程服务（MPS）时，通过调用 `cudaDeviceSetLimit` API 无法更改 L2 缓存预留大小。相反，只能在 MPS 服务器启动时通过环境变量 `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT` 指定预留大小，可以设置为 0 至 100 的百分比，默认为 0。

## 6 模块加载相关
### 6.1 CUDA\_MODULE\_LOADING
环境变量 `CUDA_MODULE_LOADING` 用于指定应用程序的模块加载模式，可以设置为 `DEFAULT`、`LAZY`、`EAGER`，默认 `LAZY`。在未来的 CUDA 版本中，默认值可能会更改。

设置为 `EAGER` 时，Cubin、Fatbin 或 PTX 文件中的所有 Kernel 和数据都会在相应的 `cuModuleLoad*` 和 `cuLibraryLoad*` API 调用中完全加载。设置为 `LAZY` 时，特定 Kernel 的加载将延迟到调用 `cuModuleGetFunction` 或 `cuKernelGetFunction` API 提取 CUfunc 句柄时，Cubin 中的数据的加载将会延迟到 Cubin 中的第一个 Kernel 加载时或 Cubin 中的变量第一次被访问时。

### 6.2 CUDA\_MODULE\_DATA\_LOADING
环境变量 `CUDA_MODULE_DATA_LOADING` 用于指定应用程序的数据加载模式，可以设置为 `DEFAULT`、`LAZY`、`EAGER`，默认 `LAZY`。在未来的 CUDA 版本中，默认值可能会更改。

设置为 `EAGER` 时，Cubin、Fatbin 或 PTX 文件中的所有数据都会在相应的 `cuLibraryLoad*` API 调用时完全加载到内存中。设置为 `LAZY` 时，Cubin 中的数据的加载将会延迟到 Cubin 中的第一个 Kernel 加载时或 Cubin 中的变量第一次被访问时。

该环境变量用于单独控制数据加载模式，不会影响到 Kernel 加载，如果未设置此环境变量，则数据加载模式将取决于环境变量 `CUDA_MODULE_LOADING`。

## 7 预加载依赖库相关
### 7.1 CUDA\_FORCE\_PRELOAD\_LIBRARIES
环境变量 `CUDA_FORCE_PRELOAD_LIBRARIES` 用于指定驱动程序在初始化期间是否预加载 NVVM 和 PTX 即时编译的依赖库。可以设置为 0 或 1，默认为 0。

当设置为 1 时，强制驱动程序在驱动程序初始化期间预加载 NVVM 和 PTX 实时编译所需的库，这将增加内存占用和 CUDA 驱动程序初始化所需的时间。设置此环境变量有时可以避免涉及多个 CUDA 线程的某些死锁情况。

## 8 CUDA 图相关
### 8.1 CUDA\_GRAPHS\_USE\_NODE\_PRIORITY
环境变量 `CUDA_GRAPHS_USE_NODE_PRIORITY` 会覆盖 CUDA 图实例化时调用 `cudaGraphInstantiate` API 传入的 `cudaGraphInstantiateFlagUseNodePriority` 标志，可以设置为 0 或 1。当设置为 1 时，将为所有 CUDA 图设置该标志；当设置为 0 时，将清除所有 CUDA 图的该标志。

