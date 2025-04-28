#! https://zhuanlan.zhihu.com/p/679629649
![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5oABN9aOICUMV2VXV9xL4k1mUaNjDsQWP7byFIQqlPUKSJ09hertj6Uib3I5X82sn2Eut8IticKe8bw/640?wx_fmt=png&amp;from=appmsg)
# 【CUDA编程】基于 CUDA 的 Kmeans 算法的进阶实现（二）

**写在前面**：本文主要介绍如何使用 CUDA 并行计算框架实现机器学习中的 Kmeans 算法，Kmeans 算法的原理见笔者的另一篇文章[【机器学习】K均值聚类算法原理](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484421&idx=1&sn=f6e5c4fa2a4f289766bcc665f0e8a5dd&chksm=e92781bcde5008aa0d3ecbe8bc61901b219e453167a7027b7d94a6779729a773ba1f860e1e68&token=1840879869&lang=zh_CN#rd)，笔者之前的两篇文章分别给出了 naive 版本和进阶版本的 Kmeans 算法的 CUDA C++ 程序，本文将在进阶版本的基础上再进行一定的优化。

## 1 优化内容
关于 Kmeans 算法的原理与具体计算过程，可以参考笔者的前几篇文章，本文只讨论基于进阶版本的最新的优化内容，为了便于读者阅读和理解，建议读者结合下面三篇文章一起来看。
- [【机器学习】K 均值聚类算法原理](https://mp.weixin.qq.com/s/o9bl1M9G1cOSYzzTZ3eYxw)
- [【CUDA编程】基于 CUDA 的 Kmeans 算法的简单实现](https://mp.weixin.qq.com/s/2PfocGm9l84l5Jj1vYF5bg)
- [【CUDA编程】基于 CUDA 的 Kmeans 算法的进阶实现（一）](https://mp.weixin.qq.com/s/5Kr8ltlzy1nL7aeGrETYvA)

先来看一下 `KmeansGPUV3` 类的结构：
```
template <typename DataType>
class KmeansGPUV3 : public Kmeans<DataType>
{
public:
    KmeansGPUV3(int num_clusters, int num_features, DataType *clusters, int num_samples);
    KmeansGPUV3(int num_clusters, int num_features, DataType *clusters, int num_samples,
                int max_iters, float eplison);
    virtual ~KmeansGPUV3();
    void fit(const DataType *v_data);

    DataType *d_data;     // [num_samples, num_features]
    DataType *d_clusters; // [num_clusters, num_features]
    int *d_sampleClasses; // [num_samples, ]
    float *d_min_dist;    // [num_samples, ]
    float *d_loss;        // [num_samples, ]
    int *d_cluster_size;  //[num_clusters, ]
    cudaStream_t calculate_stream;
    cudaStream_t update_stream;
    cudaEvent_t calculate_event;
    cudaEvent_t update_event;

private:
    KmeansGPUV3(const Kmeans &model);
    KmeansGPUV3 &operator=(const Kmeans &model);
};

```
关于基类 `Kmeans` 的结构以及上一个版本的 `KmeansGPUV2` 类的结构，可以参阅笔者前几篇文章，这里就不大量粘贴重复代码了。

相比于前面的版本，`KmeansGPUV3` 只重写了 `fit` 函数，不再需要 `getDistance` 和 `updateClusters` 两个函数，所有的计算逻辑全在 `fit` 函数里。此外，成员变量中多了两个 CUDA 流句柄 `calculate_stream` 和 `update_stream`，以及两个 CUDA 事件 `calculate_event` 和 `update_event`，其中 `calculate_stream` 相当于之前的 `master_stream`，主要的计算操作及数据拷贝都在这个流中进行，而 `update_stream` 中主要进行样本簇中心点更新操作，在上一篇文章中说过，`loss` 计算和中心点更新可以并发，就体现在这里。两个事件用于控制计算操作与更新操作的依赖关系，确保每一轮训练前已完成上一轮的中心点更新、每次更新中心点前已完成本轮距离计算。

新版本的优化内容如下：
- 所有内存操作（申请、拷贝、释放）均使用异步 API，主机端不用再等待内存操作完成。
- 新增两个 CUDA 流 `calculate_stream` 和 `update_stream`，让 `loss` 计算和中心点更新并发进行。
- 所有的计算逻辑都包含在 `fit` 函数中，简化了对外接口。当然这里代码写的有点问题，实际 `KmeansGPUV3` 还会继承 `Kmeans` 的  `getDistance` 和 `updateClusters` 两个函数，这里规范起见应该单独提一个虚基类出来，但是我太懒了，先这样吧。
- Pack 数据，计算距离的函数中加入了向量化加载数据机制，提升访问带宽。
- 把直方图统计操作融合到计算距离的 kernel 中。

整体操作执行流程见下图：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5oABN9aOICUMV2VXV9xL4k1mUaNjDsQWP7byFIQqlPUKSJ09hertj6Uib3I5X82sn2Eut8IticKe8bw/640?wx_fmt=png&amp;from=appmsg)

## 2 构造函数和析构函数
在构造函数中主要做了 4 件事：调用基类构造函数完成成员变量初始化、创建 CUDA 流和事件、分配设备内存、从主机端拷贝中心点坐标数据到设备内存 `d_clusters` 并插入 `update_event` 事件。这里使用异步版本的 CUDA 运行时 API（如 `cudaMallocAsync` 和 `cudaMemcpyAsync`），让内存操作都在 `update_stream` 中异步进行，此时主机端不用等待设备端的内存操作，节省了实例化对象的时间。具体代码如下：
```cuda
template <typename DataType>
KmeansGPUV3<DataType>::KmeansGPUV3(int num_clusters, int num_features, DataType *clusters, int num_samples,
                                   int max_iters, float eplison)
    : Kmeans<DataType>(num_clusters, num_features, clusters, num_samples, max_iters, eplison)
{
    CHECK(cudaStreamCreate(&calculate_stream));
    CHECK(cudaStreamCreate(&update_stream));
    CHECK(cudaEventCreate(&calculate_event));
    CHECK(cudaEventCreate(&update_event));

    int data_buf_size = m_num_samples * m_num_features;
    int cluster_buf_size = m_num_clusters * m_num_features;
    int mem_size = sizeof(DataType) * (data_buf_size + cluster_buf_size) + sizeof(int) * (m_num_samples) +
                   sizeof(float) * (m_num_samples + m_num_samples) + sizeof(int) * m_num_clusters;

    CHECK(cudaMallocAsync((void **)&d_data, mem_size, calculate_stream));

    d_clusters = (DataType *)(d_data + data_buf_size);
    d_sampleClasses = (int *)(d_clusters + cluster_buf_size);
    d_min_dist = (float *)(d_sampleClasses + m_num_samples);
    d_loss = (float *)(d_min_dist + m_num_samples);
    d_cluster_size = (int *)(d_loss + m_num_samples);

    CHECK(cudaMemcpyAsync(d_clusters, m_clusters, sizeof(DataType) * cluster_buf_size, cudaMemcpyHostToDevice, update_stream));
    CHECK(cudaEventRecord(update_event, update_stream));

    printf("num_samples: %d  num_clusters: %d  num_features: %d\n", num_samples, num_clusters, num_features);
}
```

在析构函数中只有两件事：释放设备内存、销毁 CUDA 流和事件。

```cuda
template <typename DataType>
KmeansGPUV3<DataType>::~KmeansGPUV3() 
{
    CHECK(cudaFreeAsync(d_data, calculate_stream));
    CHECK(cudaStreamDestroy(calculate_stream));
    CHECK(cudaStreamDestroy(update_stream));
    CHECK(cudaEventDestroy(calculate_event));
    CHECK(cudaEventDestroy(update_event));
}
```

## 3 fit 函数
`fit` 函数的主要框架与之前并无不同，只是把调用 `getDistance` 和 `updateClusters` 两个函数改成了只调用 `launchFit` 函数，核心逻辑还在 `launchFit` 函数内，代码如下：
```cuda
template <typename DataType>
void KmeansGPUV3<DataType>::fit(const DataType *v_data)
{
    float *h_loss = new float[m_num_samples]{0.0};

    CHECK(cudaMemcpyAsync(d_data, v_data, sizeof(DataType) * m_num_samples * m_num_features, cudaMemcpyHostToDevice, calculate_stream));

    float lastLoss = 0;
    for (int i = 0; i < m_max_iters; ++i)
    {
        launchFit<DataType>(d_data, d_clusters, d_sampleClasses, d_cluster_size, d_min_dist, d_loss,
                            m_num_clusters, m_num_samples, m_num_features, calculate_stream, update_stream, 
                            calculate_event, update_event);

        CHECK(cudaMemcpyAsync(h_loss, d_loss, sizeof(float) * m_num_samples, cudaMemcpyDeviceToHost, calculate_stream));
        CHECK(cudaStreamSynchronize(calculate_stream));
        this->m_optTarget = h_loss[0];
        if (std::abs(lastLoss - this->m_optTarget) < this->m_eplison)
            break;
        lastLoss = this->m_optTarget;
        std::cout << "Iters: " << i + 1 << "  current loss : " << m_optTarget << std::endl;
    }

    CHECK(cudaMemcpyAsync(m_clusters, d_clusters, sizeof(DataType) * m_num_clusters * m_num_features, cudaMemcpyDeviceToHost, calculate_stream));
    CHECK(cudaMemcpyAsync(m_sampleClasses, d_sampleClasses, sizeof(int) * m_num_samples, cudaMemcpyDeviceToHost, calculate_stream));

    delete[] h_loss;
}
```
关于 `launchFit` 函数中的操作执行流程图已经在前面给出，读者可以结合流程图阅读代码。下面先给出代码再逐步分析代码实现意图。
```cuda
template <typename DataType>
void launchFit(const DataType *d_data, DataType *d_clusters, int *d_sample_classes,
               int *d_cluster_size, float *d_min_dist, float *d_loss, const int num_clusters,
               const int num_samples, const int num_features, cudaStream_t calculate_stream,
               cudaStream_t update_stream, cudaEvent_t calculate_event, cudaEvent_t update_event)
{
    CHECK(cudaStreamWaitEvent(calculate_stream, update_event));

    initV2<int><<<1, 1024, 0, calculate_stream>>>(d_cluster_size, 0.0f, num_clusters);

    if (num_features % 4)
    {
        calClustersDistPackedkernel<DataType, 1><<<num_samples, block_size, 0, calculate_stream>>>(d_data, d_clusters,
            d_sample_classes, d_min_dist, d_cluster_size, num_features, num_clusters);
    }
    else
    {
        calClustersDistPackedkernel<DataType, 4><<<num_samples, block_size, 0, calculate_stream>>>(d_data, d_clusters,
            d_sample_classes, d_min_dist, d_cluster_size, num_features, num_clusters);
    }
    CHECK(cudaEventRecord(calculate_event, calculate_stream));

    vec1DReduce<SumOp><<<block_size, block_size, 0, calculate_stream>>>(d_min_dist, d_loss, num_samples);
    vec1DReduce<SumOp><<<1, block_size, 0, calculate_stream>>>(d_loss, d_loss, block_size);

    CHECK(cudaStreamWaitEvent(update_stream, calculate_event));

    initV2<DataType><<<1, 1024, 0, update_stream>>>(d_clusters, 0.0f, num_clusters * num_features);
    update<DataType><<<num_samples, block_size, 0, update_stream>>>(d_data, d_clusters,
                                                                    d_sample_classes, d_cluster_size, num_samples, num_features);
    CHECK(cudaEventRecord(update_event, update_stream));
}
```
首先让 `calculate_stream` 流中的后续操作等待 `update_event` 事件，这个事件是插入在 `update_stream` 流中的，一旦捕获到这个事件则表明中心点坐标已经完成更新（对于第一轮训练来说，则是完成了中心点坐标的内存拷贝）可以用来计算，然后在 `calculate_stream` 中插入 `initV2` kernel 进行 `d_cluster_size` 初始化，这里是因为直方图统计操作已经被融合到 `calClustersDistPackedkernel` kernel 中，所以要保证在直方图更新前完成赋零操作，这里其实可以把这个 kernel 插入到 `update_stream` 流随样本簇中心点更新一块完成的，不过这个操作代价很小就先放在 `calculate_stream` 流中吧。

### 3.1 向量化数据加载
很多时候由于 kernel 中的计算逻辑较为简单，性能瓶颈主要是在带宽利用上。英伟达官方博客[CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) 中提到，使用向量化操作能够提升读写的带宽，所以笔者对原本的距离计算 kernel 进行了改写加入了向量化数据加载的内容。

在调用 `calClustersDistPackedkernel` 前需要先确定向量化的数据的大小，也就是 `pack_size`，这里要对 `num_features` 进行判断，如果能被 `4` 整除 `pack_size` 就取 `4`，这里是因为笔者偷懒了，笔者事先知道数据用的是 `float` 类型，最多可以取到 `4`，所以如果要 Pack 那直接就把 `4` 个数据 Pack 到一起，否则 `pack_size` 等于 `1` 相当于不 Pack。关于这个 `pack_size` 的确定可以参考 OneFlow 的源码。
```
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
CUDA 编程模型中对设备内存的访问粒度最大为 `128`bit 也就是 
`4` 个 `float` 的大小。通过上面的代码在一般情况下可以获取到最大的 `pack_size`。

#### 3.1.1 Pack 类型定义
关于 Pack 数据结构，笔者参考的也是 OneFlow 的源码，定义一个 `Packed` 结构体，注意构造函数要用 `__device__` 修饰，否则在设备代码中无法实例化对象，结构体中有一个 `union` 对象，其中包含了 `storage` 和 `elem` 两个成员，`storage` 就是用来占位置的，其内存大小与 `elem` 相同，要表示这个 `storage` 的类型，我们还需要定义一个能表示 `pack_size` 个 `T` 对象的类型 `PackType`，可以通过 `std::aligned_storage` 来获取这样一个类型。这里编译的时候可能会报错，可以加上 `-D _ENABLE_EXTENDED_ALIGNED_STORAGE` 避免。
```cuda
template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template <typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed
{
    __device__ Packed()
    {
        // do nothing
    }
    union
    {
        PackType<T, pack_size> storage;
        T elem[pack_size]; 
    };
};
```

### 3.2 calClustersDistPackedkernel 函数
`calClustersDistPackedkernel` 核函数用于计算样本到各个中心点的距离、找到最近中心点以及统计直方图的工作，`grid_size` 设置为 `num_samples`，即一个 block 内完成一个样本到各个中心点距离的计算。相比于 `KmeansGPUV2` 的 `calClustersDistkernel`，增加了向量化数据加载、直方图统计两个优化内容，其中直方图统计没什么好说的，就是在 `0` 号线程内把当前 block 处理的样本的最近中心点的索引更新到 `d_cluster_size` 中，注意这里是一个原子操作。而向量化数据加载主要在于设备函数 `calDistPacked` 中。

```cuda
template <typename DataType, int pack_size>
__global__ void calClustersDistPackedkernel(const DataType *d_data,
                                            const DataType *d_clusters, // [num_clusters, num_features]
                                            int *d_sample_classes,      // [nsamples, ]
                                            float *d_min_dist,          // [nsamples, ]
                                            int *d_clusterSize,         // [nsamples, ]
                                            const int num_features,
                                            const int num_clusters)
{
    // grid_size = num_samples, block_size = 256
    float min_dist = 1e9f;
    float dist;
    int min_idx;

#pragma unroll
    for (int i = 0; i < num_clusters; ++i)
    {
        dist = calDistPacked<DataType, pack_size>(d_data, d_clusters, i, num_features);
        if (dist < min_dist)
        {
            min_dist = dist;
            min_idx = i;
        }
    }

    if (threadIdx.x == 0)
    {
        d_sample_classes[blockIdx.x] = min_idx;
        d_min_dist[blockIdx.x] = sqrtf(min_dist);
        atomicAdd(&(d_clusterSize[min_idx]), 1);
    }
}
```
首先对中心点进行循环，调用设备函数 `calDistPacked` 计算该样本到指定中心点的距离 `dist`，将其与 `min_dist` 进行比较，最终循环结束时就找到了最近的中心点，在 `0` 号线程内部，把最近的中心点和对应的距离写入设备内存的变量中，该核函数的核心逻辑都在 `calDistPacked` 中，下面来看一下具体代码。
```
template <typename DataType, int pack_size>
__device__ float calDistPacked(const DataType *d_data,
                               const DataType *d_clusters, // [num_clusters, num_features]
                               const int clusterNo, const int num_features)
{
    // grid_size = num_samples, block_size = 128
    const int sample_offset = num_features * blockIdx.x;
    const int cluster_offset = num_features * clusterNo;

    const PackType<DataType, pack_size> *buf = reinterpret_cast<const PackType<DataType, pack_size> *>(d_data + sample_offset);
    const PackType<DataType, pack_size> *cluster_buf = reinterpret_cast<const PackType<DataType, pack_size> *>(d_clusters + cluster_offset);
    int num_packs = num_features / pack_size;

    float distance = 0.0f;
    float sub_val;
    Packed<DataType, pack_size> data_pack;
    Packed<DataType, pack_size> cluster_pack;

#pragma unroll
    for (int pack_id = threadIdx.x; pack_id < num_packs; pack_id += blockDim.x)
    {
        data_pack.storage = *(buf + pack_id);
        cluster_pack.storage = *(cluster_buf + pack_id);
#pragma unroll
        for (int elem_id = 0; elem_id < pack_size; ++elem_id)
        {
            sub_val = (float)(data_pack.elem[elem_id] - cluster_pack.elem[elem_id]);
            distance += sub_val * sub_val;
        }
    }
    __syncthreads();

    distance = blockReduce<SumOp, float>(distance);
    return distance;
}
```
关于两点距离的计算公式大家都很熟悉，对应坐标分量（属性值）差值的平方和再开平方，即 $\sqrt{\sum {(x_i - y_i)^2}}$，但是在 `calDistV2` 内没必要开平方，计算到平方和就够了。由于是每个 block 处理一行数据，所以先计算出当前 block 处理的元素的偏移量 `sample_offset` 和 `cluster_offset`，由于使用了数据 Pack，所以总共 block 内需要处理 `num_packs` 对数据，每个线程总共需要处理 `num_packs / blockDim.x` 对 Pack 对象，索引步长为 `blockDim.x`，单个线程内直接循环处理得到这 `num_packs / blockDim.x` 对 Pack 对象，Pack 内部循环计算对应元素的差值的平方和 `distance`，同步之后再进行块内规约，就得到整个 block 内 `distance` 的和，这其实就是 $\sum {(x_i - y_i)^2}$。

重点在于 Pack 对象的读取和使用，先使用 `reinterpret_cast<const PackType<DataType, pack_size> *>` 将 `DataType` 类型的指针转换为 `PackType<DataType, pack_size>` 类型的指针，注意这里一定要用 `PackType` 不能用 `Packed` 类型，要与 `storage` 的类型对应上，否则会因为不存在 `Packed` 到 `PackType` 的隐式转换而报错。然后定义两个寄存器变量 `data_pack` 和 `cluster_pack` 用来存储从源数据中读取的 Pack 对象，取 Pack 的时候直接用 `storage = *(ptr + offset)` 的形式即可，因为 `storage` 与 `elem` 共享同一块内存，所以使用的时候可以通过 `elem` 来使用。这样的话，相当于一次从设备内存取 `pack_size` 个数据到寄存器，再从寄存器中按个拿出来用，因为寄存器速度最快，所以相比于直接从设备内存挨个加载数据使用带宽更高一些。

### 3.3 异步计算 loss 和更新中心点坐标
在 `calClustersDistPackedkernel` 后面插入 `calculate_event` 事件，用于通知 `update_stream` 流距离计算已经完成，然后在 `calculate_stream` 中启动两个 `vec1DReduce` kernel 用于计算整体 loss，关于这个 kernel 的计算逻辑这里不再赘述，有疑问的读者可以参阅前面的文章。与此同时在 `update_stream` 流中捕获 `calculate_event` 当距离计算已经完成时即刻进行样本中心点的更新，此时样本中心点的更新就与 loss 计算并发了，而我们每一轮结束判断 loss 的时候并不需要完成样本簇中心点的更新，这相当于把样本簇中心点的更新的操作隐藏在 loss 计算和主机代码运行中了，加快了训练过程。 

## 4 小结
本文内容总结如下：
- 加入了向量化加载数据的内容。
- kernel 融合，把直方图统计的 kernel 合并到距离计算 kernel 中。
- 异步计算 loss 和更新中心点坐标。 
- 通过 CUDA 事件代替了显式流同步，将尽可能避免和主机端进行同步。