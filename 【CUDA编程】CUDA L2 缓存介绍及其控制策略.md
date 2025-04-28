# 【CUDA编程】CUDA L2 缓存介绍

**写在前面**：本文详细介绍了 CUDA 编程模型中的 L2 缓存，L2 缓存与全局内存、共享内存不同，它不能由开发人员显式使用，而且在早期的 CUDA 版本和设备上，也没有提供能够影响 L2 缓存策略的接口，所以在 CUDA 程序开发中我们通常会忽略它的存在。值得一提的是，从 CUDA 11.0 开始，计算能力 8.0 及以上的设备上能够显式控制 L2 缓存中数据的持久性，使得开发人员可以更细粒度地控制内存访问，从而写出更高性能的 CUDA 并行代码。

## 1 L2 级缓存
我们知道，设备端的全局内存有两个主要特点，一是容量大，所以通常设备端的数据大都存储在这里以备 Kernel 访问；二是延迟高，因此应该尽可能减少对设备内存的访问，L2 缓存就是专门用来优化这个问题。

L2 缓存是一块设备级的存储介质，也就是说，所有 SM 共享同一块硬件内存专门用于缓存对于全局内存的访问数据，L2 缓存的容量远小于全局内存同时延迟也小于全局内存。因此，在了解 L2 缓存机制之前，我们要先对全局内存的访问进行分类，从而搞清楚哪些数据应该被缓存，而哪些数据无需缓存。

L2 缓存的大小和具体设备型号有关，在应用程序中可以通过查询设备属性获取，具体地，通过调用 `cudaGetDeviceProperties` 函数查询 `l2CacheSize` 属性即可。

通常我们把对全局内存的访问分为两类：持久化访问、流式访问。当一个 CUDA Kenerl 重复访问全局内存中的一个数据区域时，这种数据访问可以被认为是持久化的；另一方面，如果数据只被访问一次，那么这种数据访问可以被认为是流式的。显然，对于持久化的数据访问，应当优先被缓存到 L2 缓存中。

从 CUDA 11.0 开始，计算能力 8.0 及以上的设备能够显式控制 L2 缓存中数据的持久性，从而可能提供对全局内存的更高带宽和更低延迟的访问。

## 2 为持久化访问预留 L2 缓存空间

CUDA Runtime 中提供了相关的 API，可以使应用程序显式地预留一部分 L2 缓存空间专门用于缓存持久化访问的数据，从而加速对全局内存数据的访问，而对全局数据的正常访问和流式访问只能缓存在 L2 缓存的其他部分。具体地，可以调用 `cudaDeviceSetLimit` 函数设置 L2 缓存中预留的用于持久访问的空间大小：
```cpp
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
/* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
```

从上面的代码可以看出，通常为了保证设置 L2 缓存持久预留空间时不出错，应用程序应当确保设置的值不超过两个限制，一个是不能超过设备上可用的 L2 缓存数量，还有就是不能超过设备允许的最大 L2 缓存持久预留空间，这个 L2 缓存持久预留空间的上限也是可以通过查询设备属性获取的，具体地，可以调用 `cudaGetDeviceProperties` 函数查询 `persistingL2CacheMaxSize` 属性。

另外要注意的是，在某些场景下，预留 L2 缓存功能会有些限制：
- 在 MIG 模式（GPU 虚拟化）下，L2 缓存预留功能被禁用。
- 在开启 MPS 服务时，即一个 GPU 上并发执行多个进程的 CUDA 调用，此时 `cudaDeviceSetLimit` 函数无法更改 L2 缓存预留大小，只能在 MPS 服务器启动时通过环境变量 `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT` 指定预留大小。

## 3 L2 持久化访问策略
除了 L2 缓存预留以外，CUDA Runtime 还提供了一个**访问策略窗口**（Access Policy Window）的概念用于实现不同的持久化访问策略，访问策略窗口中指定了需要持久化访问的全局内存的区域（连续地址空间）、访问属性以及缓存命中概率等等指标。

下面的代码演示了如何通过 CUDA 流设置 L2 访问策略窗口。
```cpp
// Stream level attributes data structure
cudaStreamAttrValue stream_attribute;   
// Global Memory data pointer                                      
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr);
// Number of bytes for persistence access.
// (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;    
// Hint for cache hit ratio              
stream_attribute.accessPolicyWindow.hitRatio  = 0.6; 
// Type of access property on cache hit                         
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; 
// Type of access property on cache miss.
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  

//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

上述代码中，设置了访问策略窗口对象 `stream_attribute.accessPolicyWindow` 的几个重要属性：全局内存起始地址、窗口大小、命中概率、命中数据的访问属性。另外要注意的是，访问策略窗口的大小也是有上限的，应当小于设备属性 `accessPolicyMaxWindowSize`，具体查询方式同上。

通过上述设置，当 Kernel 随后在 CUDA 流中执行时，全局内存范围 `[ptr, ptr+num_bytes)` 内的数据相比其他全局内存位置的数据更有可能被缓存在 L2 缓存中。同时，全局内存区域 `[ptr, ptr+num_bytes)` 中 60% 的内存访问具有持久化属性，40% 的内存访问具有流属性，具体哪些访问是持久的是随机的，概率分布取决于硬件架构和内存范围。

要注意的是，这个访问策略窗口和上一节说的 L2 预留缓存空间不是一个概念，举个例子，如果 L2 预留缓存大小为 16KB，而 `accessPolicyWindow` 中的 `num_bytes` 为 32KB，并且 `hitProp` 设置为持久化属性时：
- 如果 `hitRatio` 为 0.5，GPU 将随机选择 32KB 的访问策略窗口中的 16KB 作为持久化部分，并缓存在预留的 L2 缓存区域中。
- 如果 `hitRatio` 为 1.0，GPU 将尝试在预留的 L2 缓存区域中缓存整个 32KB 的访问策略窗口。但由于预留区域小于窗口大小，GPU 会将 32KB 数据中最近使用的 16KB 保留在 L2 缓存的预留部分中，窗口中的其他部分将被清出缓存。

在多流并发的场景下，`hitRatio` 还可用于避免因缓存竞争而导致的缓存被异常清出，从整体上控制 L2 缓存移入、移出的数据规模。具体地，可以将 `hitRatio` 设置为一个小于 1.0 的值，从而手动控制不同 CUDA 流的访问策略窗口中缓存在 L2 中的数据量。例如，假设 L2 预留缓存大小为 16KB，两个不同 CUDA 流的访问策略窗口都为 16KB 且两者的 `hitRatio` 值都为 1.0。此时两个流中并发执行的 Kernel 在竞争 L2 缓存资源时，可能会互相清除对方的缓存数据，导致性能下降。但是，如果两个 CUDA 流的访问策略窗口的 `hitRatio` 值都设置为 0.5，则可以有效缓解这种互相清除对方缓存数据的情况。


## 4 访问属性
前面介绍过，Kernel 对全局内存的访问分为两类：持久化访问、流式访问。对应地，CUDA Runtime 中访问策略窗口对象针对不同的数据访问类型，定义了三种访问属性以供开发人员灵活控制：
- `cudaAccessPropertyStreaming`：即流式访问，基于流属性的内存访问数据一般不会在 L2 缓存中持续存在，这些数据将被优先清出 L2 缓存。
- `cudaAccessPropertyPersisting`：即持久化访问，基于持久属性的内存访问的数据更有可能保留在 L2 缓存的预留空间中。
- `cudaAccessPropertyNormal`：即访问属性重置，相当于访问策略窗口没有设置访问属性。该属性是为了解决 **persistence-after-use** 场景带来的无效缓存问题，即对于一些之前就已经执行完毕的 CUDA Kernel 的具有持久性属性的内存访问数据，即使这部分数据后面不再被访问了，但是还会在很长一段时间内继续保留在 L2 缓存中，这种无效缓存数据会减少后续 Kernel 可用的 L2 缓存空间。除了 `cudaAccessPropertyNormal` 属性以外，还有其他方式可以处理这种缓存重置问题，下面将进行介绍。

在 persistence-after-use 场景下，为了保障后续 Kernel 的内存访问能够正常使用 L2 缓存，需要将 L2 缓存的访问属性重置。除了前面提到的将访问策略窗口的访问属性设置为 `cudaAccessPropertyNormal` 属性以外，还有两种方式：一种是显式的，由应用程序显式调用 `cudaCtxResetPersistingL2Cache()` 清除之前 L2 缓存中的持久化数据；还有一种是隐式的，如果这些 L2 缓存中的持久化数据最终没有被访问，CUDA 底层也有自动重置机制将其清出 L2 缓存，这种自动重置的行为是隐式的，具体重置的时间点不能保证，为了追求极致性能，通常不推荐这种不可控的方式。

前面我们提到的 L2 缓存的访问策略窗口设置都是针对某个 CUDA 流的，其目的是为了加速流中执行的 Kernel 的数据访问，要明确的是，从硬件层面来说，L2 缓存的预留空间是由所有并发的 CUDA Kernel 之间共享的。也就是说，针对多流并发的场景，随着并发 Kernel 变多，持久访问数据量也逐渐增多，此时持久访问的数据量很容易超过 L2 缓存预留空间的容量，这时候基于 L2 缓存持久访问数据的收益也会逐渐减小。因此，针对这种多流并发的场景，为了实现极致性能，开发人员应当综合考虑 L2 缓存预留空间的容量、Kernel 并发的设计、访问策略窗口的属性设置以及访问属性重置的时机等因素。

总的来说，相比较全局内存、共享内存这些设备端内存，L2 缓存是一块不能由应用程序显式使用的内存空间，即开发人员控制不了具体某个数据访问是否被缓存在 L2 中，但是应用程序可以通过 CUDA Runtime API 从宏观层面上影响 L2 缓存的策略，下面我们通过一个例子来介绍 CUDA 开发中调用 CUDA Runtime API 设置 L2 缓存的策略的方式。

```cpp
// Create CUDA stream
cudaStream_t stream;
cudaStreamCreate(&stream);                      
// CUDA device properties variable
cudaDeviceProp prop;           
// Query GPU properties                       
cudaGetDeviceProperties( &prop, device_id);               
size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
// set-aside 3/4 of L2 cache for persisting accesses or the max allowed
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);   

// Select minimum of user defined num_bytes and max window size.
size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes); 

// Stream level attributes data structure
cudaStreamAttrValue stream_attribute;       
// Global Memory data pointer                                                
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);   
// Number of bytes for persistence access       
stream_attribute.accessPolicyWindow.num_bytes = window_size;      
// Hint for cache hit ratio                         
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;       
// Persistence Property                             
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;   
// Type of access property on cache miss            
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;           
// Set the attributes to a CUDA Stream
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   

// This data1 is used by a kernel multiple times
// [data1 + num_bytes) benefits from L2 persistence
for (int i = 0; i < 10; i++) {
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);            
}             
// A different kernel in the same stream can also benefit from the persistence of data1  
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);    

// Setting the window size to 0 disable it
stream_attribute.accessPolicyWindow.num_bytes = 0;   
// Overwrite the access policy attribute to a CUDA Stream                                      
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   
// Remove any persistent lines in L2
cudaCtxResetPersistingL2Cache();          
// data2 can now benefit from full L2 in normal mode       
cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                 
```

在上面的代码中，首先 L2 缓存持久预留空间的大小设置为 `size`，然后在 `stream` 流中将全局内存空间 `[data1, data1+window_size)` 设置为访问策略窗口，随机选择其中 60% 缓存到 L2 预留空间，所以流中执行的 `cuda_kernelA` 和 `cuda_kernelB` 都可以从预留空间中加载缓存数据。随后由于流中执行的 Kernel 不再使用到 `data1`，因此代码中将访问策略窗口的大小置为 0 并清空了 L2 预留空间中的缓存的数据。此后在执行 `cuda_kernelC` 的时候由于前面访问策略窗口大小被设置为 0，因此虽然 `data2` 中的数据依旧可以缓存在 L2 中，但是不再具有持久化属性。


## 5 小结
本文详细介绍了 CUDA 编程模型中 L2 级缓存的作用及缓存策略，总结如下：
- L2 缓存是一块设备级的存储介质，也就是说，所有 Kernel 共享同一块L2 缓存，L2 缓存的容量远小于全局内存同时延迟也小于全局内存。
- L2 缓存是一块不能由应用程序显式使用的内存空间，但是应用程序可以通过 CUDA Runtime API 从宏观层面上影响 L2 缓存的策略。
- 从 CUDA 11.0 开始，计算能力 8.0 及以上的设备能够显式控制 L2 缓存中数据的持久性，通过访问策略窗口进行控制。