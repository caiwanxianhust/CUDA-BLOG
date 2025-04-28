#! https://zhuanlan.zhihu.com/p/679372001
![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rmd1deWjCc5Cs3icydarodtxIn39CVhq9PWvNE2Xxd7c42324qOZqXXj2IUVL8klXLZ0WNPzeicYzg/640?wx_fmt=png&amp;from=appmsg)
# 【CUDA编程】基于 CUDA 的 Kmeans 算法的进阶实现（一）

**写在前面**：本文主要介绍如何使用 CUDA 并行计算框架实现机器学习中的 Kmeans 算法，Kmeans 算法的原理见笔者的另一篇文章[【机器学习】K均值聚类算法原理](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484421&idx=1&sn=f6e5c4fa2a4f289766bcc665f0e8a5dd&chksm=e92781bcde5008aa0d3ecbe8bc61901b219e453167a7027b7d94a6779729a773ba1f860e1e68&token=1840879869&lang=zh_CN#rd)，笔者之前在另一篇文章中给出了一个 naive 版本，而本次提供的进阶实现相对于 naive 版本大约可以再加速 3 倍，相比于原始的 CPU 版本在笔者的用例上可以加速 300 多倍。

## 1 优化内容
关于 Kmeans 算法的原理与具体计算过程，可以参考笔者的前两篇文章，本文只讨论进阶优化实现，建议读者可以结合下面两篇文章一起来看。
- [【机器学习】K均值聚类算法原理](https://mp.weixin.qq.com/s/o9bl1M9G1cOSYzzTZ3eYxw)
- [【CUDA编程】基于CUDA的Kmeans算法的简单实现](https://mp.weixin.qq.com/s/2PfocGm9l84l5Jj1vYF5bg)

先来看一下 `Kmeans` 类的结构：
```cpp
template <typename DataType>
class Kmeans
{
public:
    Kmeans(int num_clusters, int num_features, DataType *clusters, int num_samples);
    Kmeans(int num_clusters, int num_features, DataType *clusters, int num_samples,
           int max_iters, float eplison);
    virtual ~Kmeans();
    void getDistance(const DataType *v_data);
    void updateClusters(const DataType *v_data);
    virtual void fit(const DataType *v_data);
    float accuracy(int *label);

    DataType *m_clusters; //[num_clusters, num_features]
    int m_num_clusters;
    int m_num_features;
    float *m_distances;   // [num_samples, num_clusters]
    int *m_sampleClasses; // [num_samples, ]
    int m_num_samples;
    float m_optTarget;
    int m_max_iters;
    float m_eplison;

private:
    Kmeans(const Kmeans &model);
    Kmeans &operator=(const Kmeans &model);
};
```
主要就是 3 个成员函数 `getDistance`、`updateClusters` 和 `fit`，以及一些临时存放中间计算结果的指针，看了前两篇文章的读者不难理解这些成员变量的作用。

再来看一下 `KmeansGPU` 的结构：

```cuda
template <typename DataType>
class KmeansGPU : public Kmeans<DataType>
{
public:
    KmeansGPU(int num_clusters, int num_features, DataType *clusters, int num_samples);
    KmeansGPU(int num_clusters, int num_features, DataType *clusters, int num_samples,
              int max_iters, float eplison);
    virtual ~KmeansGPU();
    void getDistance(const DataType *d_data);
    void updateClusters(const DataType *d_data);
    virtual void fit(const DataType *v_data);

    DataType *d_data;     // [num_samples, num_features]
    DataType *d_clusters; // [num_clusters, num_features]
    int *d_sampleClasses; // [num_samples, ]
    float *d_distances;   // [num_samples, num_clusters]
    float *d_min_dist;    // [num_samples, ]
    float *d_loss;        // [num_samples, ]
    int *d_cluster_size;  //[num_clusters, ]
    cudaStream_t master_stream;

private:
    KmeansGPU(const Kmeans &model);
    KmeansGPU &operator=(const Kmeans &model);
};
```
在 `Kmeans` 类的基础上增加了不少 `d_` 开头的成员变量，主要是用于指向设备端的内存，作为参数传递给 kernel 使用。

最后是进阶版本的 `KmeansGPUV2` 的结构：

```cuda
template <typename DataType>
class KmeansGPUV2 : public Kmeans<DataType>
{
public:
    KmeansGPUV2(int num_clusters, int num_features, DataType *clusters, int num_samples);
    KmeansGPUV2(int num_clusters, int num_features, DataType *clusters, int num_samples,
                int max_iters, float eplison);
    virtual ~KmeansGPUV2(){};
    void getDistance(const DataType *v_data);
    void updateClusters(const DataType *v_data);
    void fit(const DataType *v_data);

    DataType *d_data;     // [num_samples, num_features]
    DataType *d_clusters; // [num_clusters, num_features]
    int *d_sampleClasses; // [num_samples, ]
    float *d_min_dist;    // [num_samples, ]
    float *d_loss;        // [num_samples, ]
    int *d_cluster_size;  //[num_clusters, ]
    cudaStream_t master_stream;

private:
    KmeansGPUV2(const Kmeans &model);
    KmeansGPUV2 &operator=(const Kmeans &model);
};
```
细心的读者很容易发现，成员变量中多了 `d_data` 和 `master_stream`，少了 `d_distances`。其中 `d_data` 变量在 `KmeansGPU` 中也有体现，只不过是放在 `fit` 函数内作为局部变量使用的，这里把它放在了 `KmeansGPUV2` 类的成员变量中，含义不变，都是用来存储待聚类的数据；`master_stream` 是一个 CUDA 流的句柄，主要的计算操作都会启动到这个流中；少了 `d_distances` 是因为进阶版本不再使用到这个变量，不再产生这个中间结果。

本次进阶实现优化内容如下：
- 设备内存统一分配和释放，不再单独分配。把 `d_data` 挪到类结构中就是这个目的，方便统一管理，在类的构造函数中分配，在析构函数中释放。
- 通过 CUDA 流，把直方图统计（统计各中心的样本数量）、向量初始化等操作启动到其他流中实现与 `master_stream` 的并发。
- 优化了计算样本到各中心点距离的 kernel，并把寻找最近中心点的任务融合到了计算距离的 kernel。
- 优化了直方图统计的 kernel。

整体操作执行流程见下图。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rmd1deWjCc5Cs3icydarodtxIn39CVhq9PWvNE2Xxd7c42324qOZqXXj2IUVL8klXLZ0WNPzeicYzg/640?wx_fmt=png&amp;from=appmsg)

## 2 构造函数和析构函数
在构造函数中主要做了 4 件事：调用基类构造函数完成成员变量初始化、创建 `master_stream` 流、分配设备内存、从主机端拷贝中心点坐标数据到设备内存 `d_clusters`。值得一提的是这里也可以使用异步版本的 CUDA 运行时 API（如 `cudaMallocAsync` 和 `cudaMemcpyAsync`），让内存操作都在 `master_stream` 中异步进行。具体代码如下：

```
template <typename DataType>
KmeansGPUV2<DataType>::KmeansGPUV2(int num_clusters, int num_features, DataType *clusters, int num_samples,
                                   int max_iters, float eplison)
    : Kmeans<DataType>(num_clusters, num_features, clusters, num_samples, max_iters, eplison)
{
    CHECK(cudaStreamCreate(&master_stream));

    int data_buf_size = m_num_samples * m_num_features;
    int cluster_buf_size = m_num_clusters * m_num_features;
    int mem_size = sizeof(DataType) * (data_buf_size + cluster_buf_size) + sizeof(int) * (m_num_samples) +
                   sizeof(float) * (m_num_samples + m_num_samples) + sizeof(int) * m_num_clusters;

    CHECK(cudaMalloc((void **)&d_data, mem_size));

    d_clusters = (DataType *)(d_data + data_buf_size);
    d_sampleClasses = (int *)(d_clusters + cluster_buf_size);
    d_min_dist = (float *)(d_sampleClasses + m_num_samples);
    d_loss = (float *)(d_min_dist + m_num_samples);
    d_cluster_size = (int *)(d_loss + m_num_samples);

    CHECK(cudaMemcpy(d_clusters, m_clusters, sizeof(DataType) * cluster_buf_size, cudaMemcpyHostToDevice));
}
```

可以看到，设备内存是整块分配的，然后根据偏移量确定各个变量指针中的地址，大幅减少了内存申请的次数，另外由于少了 `d_distances`，可以节省不少设备内存。

在析构函数中只有两件事：释放设备内存、销毁 `master_stream` 流。

```cuda
template <typename DataType>
KmeansGPUV2<DataType>::~KmeansGPUV2() 
{
    CHECK(cudaFree(d_data));
    CHECK(cudaStreamDestroy(master_stream));
}
```

## 3 getDistance 函数
`getDistance` 函数中计算了样本到各中心点的距离、距离样本最近的中心点、整体的 `loss` 以及一些数据变量初始化赋零的操作，所有的逻辑都在 `calDistWithCudaV2` 函数中，下面看一下代码：
```cuda
template <typename DataType>
void calDistWithCudaV2(const DataType *d_data, DataType *d_clusters, int *d_sample_classes, int *d_cluster_size,
                       float *d_min_dist, float *d_loss, const int num_clusters, const int num_samples, const int num_features,
                       cudaStream_t master_stream)
{
    cudaEvent_t cal_dist_event;
    CHECK(cudaEventCreate(&cal_dist_event));
    calClustersDistkernel<DataType><<<num_samples, block_size, 0, master_stream>>>(
        d_data, d_clusters, d_sample_classes, d_min_dist, num_features, num_clusters);

    CHECK(cudaEventRecord(cal_dist_event, master_stream));

    cudaStream_t tmp_stream[2];
    CHECK(cudaStreamCreate(&tmp_stream[0]));
    CHECK(cudaStreamCreate(&tmp_stream[1]));

    CHECK(cudaStreamWaitEvent(tmp_stream[0], cal_dist_event));
    CHECK(cudaStreamWaitEvent(tmp_stream[1], cal_dist_event));

    vec1DReduce<SumOp><<<block_size, block_size, 0, master_stream>>>(d_min_dist, d_loss, num_samples);
    vec1DReduce<SumOp><<<1, block_size, 0, master_stream>>>(d_loss, d_loss, block_size);

    initV2<int><<<1, 1024, 0, tmp_stream[0]>>>(d_cluster_size, 0.0f, num_clusters);
    int grid_size = (num_samples - 1) / block_size + 1;
    histCount<<<grid_size, block_size, 0, tmp_stream[0]>>>(d_sample_classes, d_cluster_size, num_clusters, num_samples);

    initV2<DataType><<<1, 1024, 0, tmp_stream[1]>>>(d_clusters, 0.0f, num_clusters * num_features);

    CHECK(cudaStreamDestroy(tmp_stream[1]));
    CHECK(cudaStreamDestroy(tmp_stream[0]));

    CHECK(cudaEventDestroy(cal_dist_event));
}

template <typename DataType>
void KmeansGPUV2<DataType>::getDistance(const DataType *d_data)
{
    calDistWithCudaV2<float>(d_data, d_clusters, d_sampleClasses, d_cluster_size, d_min_dist, d_loss, m_num_clusters, m_num_samples, m_num_features, master_stream);
}
```
首先创建了 CUDA 事件 `cal_dist_event`，该事件会被插入到距离计算操作完成后，用于通知其他流此时 `calClustersDistkernel` 已完成。

启动 `calClustersDistkernel` 时，还涉及线程网格大小的确定，这里简单起见，每个 block 处理一行数据，所以 `grid_size` 就直接是 `num_samples`，而 `block_size` 取 `128`。为什么要取 `128` 呢？主要是受到 SM 占用率和寄存器数量限制，要实现最大占用率需要 `block_size` 不低于 `128`，若要不影响寄存器使用则，`block_size` 不超过 `256`，所以综合来看也就两个值（`128`、`256`）可以选，这里实测取 `128` 性能更好。关于 `block_size` 的确定，有兴趣可以参考笔者的另一篇文章：[【CUDA编程】OneFlow Element-Wise 算子源码解读](https://mp.weixin.qq.com/s/tEUg_b5qH066qvMZJp88vQ)

将 `calClustersDistkernel` 启动到 `master_stream` 后紧接着插入 `cal_dist_event` 事件，然后又创建了两个临时的 CUDA 流 `tmp_stream[2]`，用于计算直方图和初始化向量。因为从 Kmeans 计算原理来看，如果此时完成距离计算并获取到样本到最近的中心点的距离时，此时统计中心点对应样本簇的大小（实质就是直方图统计）与计算整体 `loss` 的操作可以并行，另外对于 `d_clusters` 的初始化赋零操作也可以与二者并行，所以这里将计算直方图和初始化向量分别启动到两个临时流中，并且通过 `cal_dist_event` 事件设置依赖关系，保证两个临时流的操作在 `calClustersDistkernel` 完成之后进行。

另外再把计算整体 `loss` 的操作启动到 `master_stream` 流中，这里计算 `loss` 是通过对一维数组 `d_min_dist` 的规约求和进行的，通过两步进行：第一步把长度为 `num_samples` 的数组规约到长度为 `block_size` 的数组 `loss` 中，第二步把 `loss` 中的元素规约到首元素的位置（即 `loss[0]`）。这里给出通用的一维向量的规约代码。
```cuda
template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T WarpReduce(T val)
{
    auto func = ReductionOp<T>();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val = func(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}

template <template <typename> class ReductionOp, typename T>
__inline__ __device__
    T
    blockReduce(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = WarpReduce<ReductionOp, T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = WarpReduce<ReductionOp, T>(val);
    return val;
}

template <template <typename> class ReductionOp>
__global__ void vec1DReduce(float *vec, float *reduce, const int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0.0f;

    auto func = ReductionOp<float>();

#pragma unroll
    for (; n < N; n += blockDim.x * gridDim.x)
        val = func(val, vec[n]);
    __syncthreads();

    float block_sum = blockReduce<ReductionOp, float>(val);
    if (threadIdx.x == 0)
        reduce[blockIdx.x] = block_sum;
}
```
`vec1DReduce` 中首先基于 Grid-Stride Loops 思路让一个线程循环处理多个元素得到该线程处理的元素的规约值 `val`，然后在 block 范围内通过块内规约求出一个 block 处理的所有元素的规约值 `block_sum`，在 `0` 号线程内把规约值写入输出数组中。

块内规约通过束内规约实现，先在共享内存中定义一个长度为 `32` 的数组 `shared` 用来存储 block 内每个 warp 的规约值，使用束内规约把每个 warp 的规约值写入 `shared`，在第一个 warp 内把这些规约值传递给各个线程的寄存器变量 `val`，在第一个 warp 内进行束内规约，这样 `0` 号线程的 `val` 值就是 block 的规约值。

束内规约是基于折半规约的思想使用线程束洗牌指令实现的，这里不再赘述。

### 3.1 calClustersDistkernel 核函数
`calClustersDistkernel` 核函数代码如下：
```cuda
template <typename DataType>
__global__ void calClustersDistkernel(const DataType *d_data,
                                      const DataType *d_clusters, // [num_clusters, num_features]
                                      int *d_sample_classes,      // [nsamples, ]
                                      float *d_min_dist,          // [nsamples, ]
                                      const int num_features,
                                      const int num_clusters)
{
    // grid_size = num_samples, block_size = 128
    float min_dist = 1e9f;
    float dist;
    int min_idx;

#pragma unroll
    for (int i = 0; i < num_clusters; ++i)
    {
        dist = calDistV2<DataType>(d_data, d_clusters, i, num_features);
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
    }
}
```
`calClustersDistkernel` 核函数用于计算样本到各个中心点的距离以及找到最近中心点的工作。`grid_size` 设置为 `num_samples`，即一个 block 内完成一个样本到各个中心点距离的计算，首先对中心点进行循环，调用设备函数 `calDistV2` 计算该样本到指定中心点的距离 `dist`，将其与 `min_dist` 进行比较，最终循环结束时就找到了最近的中心点，在 `0` 号线程内部，把最近的中心点和对应的距离写入设备内存的变量中，该核函数的核心逻辑都在 `calDistV2` 中，下面来看一下具体代码。
```cuda
template <typename DataType>
__device__ float calDistV2(const DataType *d_data,
                           const DataType *d_clusters, // [num_clusters, num_features]
                           const int clusterNo, const int num_features)
{
    // grid_size = num_samples, block_size = 256
    const int sample_offset = num_features * blockIdx.x;
    const int cluster_offset = num_features * clusterNo;

    float distance = 0.0f;
    float sub_val;

#pragma unroll
    for (int i = threadIdx.x; i < num_features; i += blockDim.x)
    {
        sub_val = (float)(d_data[sample_offset + i] - d_clusters[cluster_offset + i]);
        distance += sub_val * sub_val;
    }
    __syncthreads();

    distance = blockReduce<SumOp, float>(distance);
    return distance;
}
```
关于两点距离的计算公式大家都很熟悉，对应坐标分量（属性值）差值的平方和再开平方，即 $\sqrt{\sum {(x_i - y_i)^2}}$，但是在 `calDistV2` 内没必要开平方，计算到平方和就够了。由于是每个 block 处理一行数据，所以先计算出当前 block 处理的元素的偏移量 `sample_offset` 和 `cluster_offset`，每个线程总共需要处理 `num_features / blockDim.x` 对元素，索引步长为 `blockDim.x`，单个线程内直接循环处理得到这 `num_features / blockDim.x` 对元素的差值的平方和 `distance`，同步之后再进行块内规约，就得到整个 block 内 `distance` 的和，这其实就是 $\sum {(x_i - y_i)^2}$。

### 3.2 hitCount 核函数
在 `calClustersDistkernel` 中得到了距离每个样本最近的中心点，姑且称之为样本的**最近中心点**，把所有最近中心点相同的样本称为一个新的样本簇，这个样本簇的中心就是所有样本的坐标的均值，要获得这个均值，还需要样本簇的大小 `d_cluster_size`，所以需要根据 `d_sample_classes` 统计出来，`d_sample_classes` 的长度为 `num_samples`，存储着每个样本的最近中心点的索引，所以这实际上是一个直方图统计的问题，关于直方图统计任务，笔者早期有一篇文章有过介绍，有兴趣的读者可以移步：[【CUDA编程】CUDA并行化中的直方图问题](https://mp.weixin.qq.com/s/4rn609iKWs7uepvW5ERSwQ)，下面我们来看下 `histCount` 的代码：
```cuda
__global__ void histCount(int *d_sample_classes, // [N, ]
                          int *d_clusterSize,    // [num_clusters, ]
                          const int num_clusters, const int N)
{
    // block_size = 128, grid_size = (num_samples - 1) / block_size + 1;
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ int s_histo[256];
    if (threadIdx.x < num_clusters)
        s_histo[threadIdx.x] = 0;
    __syncthreads();

#pragma unroll
    for (; n < N; n += gridDim.x * blockDim.x)
    {
        atomicAdd(&s_histo[d_sample_classes[n]], 1);
    }
    __syncthreads();
    if (threadIdx.x < num_clusters)
        atomicAdd(&d_clusterSize[threadIdx.x], s_histo[threadIdx.x]);
}
```
`histCount` 核函数中每个线程只处理 `d_sample_classes` 中的一个元素，首先在 kernel 内部定义了一个长度为 `256` 的共享内存变量 `s_histo`，用来临时存储 block 内的直方图统计情况，为什么长度是 `256`，这里我们假设样本簇的数量不会超过 `256`。在进行正式统计之前对 `s_histo` 进行初始化赋零操作并同步，然后找到线程对应元素 `d_sample_classes[n]`，这其实就是第 `n` 个样本的最近中心点的索引，然后在 `s_histo` 中相应位置加一，注意这里是一个原子操作，因为 block 内的线程会同时操作 `s_histo`，必然会有读写竞争问题。有人可能会有疑问，这里为什么有个  Grid-Stride Loops 的写法？这是一种通用写法，可以允许 `grid_size` 取其他值，使得一个线程可以循环处理多个元素，这里实测区别不大，所以干脆在调用 `histCount` 核函数的时候就一个线程处理一个元素了。把 block 内的直方图统计做完之后把 `s_histo` 更新到 `d_cluster_size` 中即可。

在 naive 版本中，直接一行代码就完成了直方图统计，即每个线程直接在 `d_cluster_size` 上原子加一，这会带来什么问题？很明显，直方图统计任务要把某个样本簇的大小更新到一个设备内存变量上面，这必然存在严重的读写竞争，说白了大家还是得按顺序更新，这个并行性是很低的而且存在对设备变量的大量读写，带宽很低。而 `histCount` 不同，在 `histCount` 中是先在 block 内完成直方图统计，然后才更新到 `d_cluster_size` 中的，针对 `d_cluster_size`，相当于读写竞争降低了 `block_size` 倍，而在 `block` 内部由于使用了共享内存，读写带宽大幅度提高，直方图统计效率也极高。

## 4 updateClusters 函数
在 `updateClusters` 函数中根据前面得到的样本簇中的样本坐标、样本簇大小，求出样本簇的中心点的坐标，即所有样本点的坐标均值。`updateClusters` 函数的主要逻辑在 `update` 函数中，这也是一个存在大量读写竞争的任务，关于这个 kernel 笔者本次没有更新，代码如下：
```cuda
template <typename DataType>
__global__ void update(
    const DataType *d_data,
    DataType *d_clusters,
    int *d_sampleClasses,
    int *d_cluster_size,
    const int num_samples,
    const int num_features)
{
    // grid_size = num_samples, block_size = block_size
    int clusterId = d_sampleClasses[blockIdx.x];
    int clustercnt = d_cluster_size[clusterId];

#pragma unroll
    for (int i = threadIdx.x; i < num_features; i += blockDim.x)
    {
        atomicAdd(&(d_clusters[clusterId * num_features + i]), d_data[num_features * blockIdx.x + i] / clustercnt);
    }
}
```

## 5 小结
本文内容总结如下：
- 关于 kmeans 算法实现，本文不涉及样本初始中心点的选取的内容，也就是说本文中初始的中心点需要通过其他方式获取到之后再拿过来计算。kmeans 的 CUDA 实现有很多，这里笔者只是给出一种例子，写到这里笔者又产生了不少新的想法，待优化的内容还有很多。
- 关于 `calClustersDistkernel` 还可以尝试一下向量化加载数据，看一下是否还有提升空间。
- 可以考虑一下 kernel 融合，把直方图统计的 kernel 合并到 `calClustersDistkernel` 中试试。
- 样本簇的更新其实和 `loss` 计算也是可以并发的，可以把样本簇的更新放到另一个流中。 