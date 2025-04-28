#! https://zhuanlan.zhihu.com/p/707675388
# 【CUDA编程】基于CUDA的Fuzzy C-means算法的简单实现

**写在前面**：这段时间笔者工作和家庭事情比较多，已经接近 3 个月没有更新公众号文章，近来抽个空闲时间把本文涉及的模糊聚类算法的 CUDA 实现做个介绍。本文涉及的模糊聚类算法——Fuzzy C-means 算法，笔者之前并没有接触过，来源于笔者 CUDA 交流群的一位读者同学（@被你逮到了呢），在他的建议下笔者对 Fuzzy C-means 算法做了一个简单的 CUDA 实现，本文将对代码逻辑进行详细介绍。

## 1 FCM 聚类算法简介
提到聚类算法，通常我们想到的就是 Kmeans、层次聚类等算法，这些算法可以根据样本特征属性将相似的样本都归到某一个样本簇，对于某一个样本来说，其跟样本簇的隶属关系是非 0 即 1 的，这种聚类方法也被称为**硬聚类**。

除此之外还有一种**软聚类**方法，使用模糊集合理论，将样本对簇的隶属度扩展为 0 到 1 之间的任意值，一个样本可以以不同的隶属度属于不同的簇，并且所有隶属度之和为 1，即更接近于哪个簇，隶属度就越高，其相似度也越高。**模糊 C 均值聚类（Fuzzy C-means）算法**简称 FCM 算法，就是软聚类方法的一种。

## 2 算法原理
FCM 算法使用隶属度来表示样本与样本簇之间的关系，对于给定的含有 `n` 个样本的样本集，如果要将这些样本划分为 `c` 类，那么显然隶属度矩阵应该是一个 `n*c` 的二维矩阵。

同时 FCM 算法也是一种基于目标函数的算法，对于给定的含有 `n` 个样本的样本集 $X=\{ x_1, x_2, \cdots x_n \}$，$x_i$ 表示第 `i` 个样本，每个样本包含 `d` 个属性，$x_i^j$ 表示第 `i` 个样本的第 `j` 个属性，`c` 个聚类中心用 $V={v_1, v_2, \cdots, v_c}$ 表示，则目标函数和约束条件如下：
$$
J(U,V) = \sum _{i=1}^{n}\sum _{j=1}^{c} u_{ij}^m d_{ij}^2  \\
\sum _{j=1}^{c} u_{ij} = 1, \quad u_{ij} \in [0, 1]
$$
其中，$u_{ij}$ 是样本点 $x_i$ 与聚类中心（样本簇）$v_j$ 的隶属度，`m` 是模糊指数（`m>1`），$d_{ij}$ 是样本点 $x_i$ 与聚类中心（样本簇）$v_j$ 的距离，一般采用欧式距离。

从公式可以看出，目标函数其实就相当于对样本与聚类中心距离平方的加权求和，权重就是隶属度的 $m$ 次方，如果考虑到特殊情况隶属度非 0 即 1，就退化到了硬聚类的目标函数。笔者从网上其他文章中也发现了有人直接计算隶属度矩阵的范式作为目标函数值，从原理上来讲，由于迭代到最后隶属度矩阵也会收敛到稳定值，这种直接取范式作为目标函数也说得通。

聚类目标即为求目标函数 $J(U,V)$ 在约束条件下的最小值，通过对目标函数的迭代优化来进行聚类。为使目标函数 $J(U,V)$ 取得最小值，在满足约束条件的情况下对目标函数使用拉格朗日乘数法，得到隶属度矩阵 $U$ 和聚类中心 $V$。

即，通过聚类中心 $v_j$ 计算隶属度矩阵用如下公式：
$$
u_{ij} = \frac{1}{\sum _{k=1}^c (\frac{d_{ij}}{d_{ik}})^{\frac{2}{m-1}}} 
$$
通过隶属度矩阵计算聚类中心 $v_j$ 用如下公式：
$$
v_j = \frac{\sum _{i=1}^{n}u_{ij}^m x_i}{\sum _{i=1}^{n}u_{ij}^m}
$$
​
通过上面的两个公式可以看出，在样本集已知的情况下，隶属度矩阵和聚类中心是可以相互求解的，那么算法的初始化就会有两种方式：一种是**初始化隶属度矩阵**，然后根据隶属度矩阵求聚类中心，再计算目标函数值，然后再根据聚类中心求隶属度矩阵开始第二轮迭代；另一种是**初始化聚类中心**，根据聚类中心求隶属度矩阵，再计算目标函数值，然后再根据隶属度矩阵求聚类中心开始第二轮迭代。两种方式在网上都有相关的文章，本文选择初始化隶属度矩阵。

## 3 算法步骤
FCM 算法具体迭代步骤如下：
- 输入：聚类数目 `c`、样本集 $X=\{ x_1, x_2, \cdots x_n \}$，停止阈值 $\epsilon$，模糊因子 $m$，最大迭代次数 $T$。
- 输出：聚类中心 $V={v_1, v_2, \cdots, v_c}$，隶属度矩阵 $U$。
- step_1：初始化隶属度矩阵 $U$、迭代次数 $t=0$。
- step_2：根据公式计算聚类中心，根据隶属度矩阵计算目标函数值 $J(t)$，根据聚类中心计算隶属度矩阵。
- step_3：比较 $J(t)$ 和 $J(t-1)$，若 $J(t) - J(t-1) \leq \epsilon$，则停止迭代，否则 $t=t+1$，返回 step_2 继续迭代。

## 4 FCM 算法的 CUDA 实现
### 4.1 FuzzyCmeans 类结构
`FuzzyCmeans` 类的主要计算逻辑在成员函数 `fit` 中，此外还定义了一些成员变量，下面将一一介绍这些成员变量的作用。
```cpp
namespace clustering {
namespace fuzzyCmeans {

template <typename DataType>
class AbstractFuzzyCmeans {
public:
    
    virtual void fit(const DataType *) = 0;
    virtual ~AbstractFuzzyCmeans() {}
};

template <typename DataType>
class FuzzyCmeans : public AbstractFuzzyCmeans<DataType>
{
public:
    FuzzyCmeans(float *h_membership, int num_clusters, int num_features, int num_samples, float scale, int max_iters = 50);
    virtual ~FuzzyCmeans();
    void fit(const DataType *v_data);

    int m_num_clusters;
    int m_num_features;
    int m_num_samples;
    int m_max_iters;
    float m_eplison;
    float m_optTarget;
    float m_scale;

    DataType *m_clusters;
    float *m_membership;
    int *m_labels;
    int *d_labels;
    float *d_membership;
    DataType *d_data;                       // [num_samples, num_features]
    DataType *d_clusters;                   // [num_clusters, num_features]
    float *d_distances;                     // [num_samples, num_clusters]
    float *d_loss;                          // [1]

private:
    FuzzyCmeans(const FuzzyCmeans &model);
    FuzzyCmeans &operator=(const FuzzyCmeans &model);
};

}   // fuzzyCmeans
}   // clustering
```
- `m_num_clusters`：样本簇的数量，相当于公式中的 $c$。
- `m_num_features`：样本的特征数量。
- `m_num_samples`：样本的数量，相当于 $n$。
- `m_max_iters`：最大迭代次数，相当于 $T$。
- `m_eplison`：迭代停止阈值，相当于 $\epsilon$。
- `m_optTarget`：目标函数值（优化目标），相当于 $J$。
- `m_scale`：模糊指数，相当于 $m$。
- `m_clusters`：聚类中心，形状为 `[num_clusters, num_features]` 的二维矩阵，相当于 $V$，主机端的变量。
- `m_membership`：隶属度矩阵 $U$，形状为 `[num_clusters, num_samples]` 的二维矩阵，主机端变量。
- `m_lables`：标签向量，隶属度最大的样本簇的编号，主机端变量。
- `d_labels`：标签向量，设备端变量。
- `d_membership`：隶属度矩阵，设备端变量。
- `d_data`：样本集，形状为 `[num_samples, num_features]` 的二维矩阵，设备端变量。
- `d_distances`：样本与聚类中心的距离，形状为 `[num_samples, num_clusters]` 的二维矩阵，设备端变量。
- `d_loss`：目标函数值（优化目标），设备端变量。

### 4.2 构造函数和析构函数
构造函数主要用于初始化一些基本的参数如样本簇数量 $c$、样本特征数量、样本数量 $n$ 等等，在构造函数内部还分别对主机端变量、设备端变量进行了内存分配，特别地，将事先初始化的聚类矩阵 `h_membership` 拷贝到设备变量 `d_membership`，用于后续 Kernel 中的计算。

析构函数比较简单，对构造函数中分配的堆内存和设备内存进行释放即可。

```cuda
template <typename DataType>
FuzzyCmeans<DataType>::FuzzyCmeans(float *h_membership, int num_clusters, int num_features, int num_samples, float scale, int max_iters) :
    m_num_clusters(num_clusters), m_num_features(num_features), m_num_samples(num_samples), m_scale(scale), m_max_iters(max_iters), 
    m_optTarget(1e7f), m_eplison(1e-10f)
{
    int mem_size = sizeof(DataType) * (m_num_clusters * m_num_features) + sizeof(float) * (m_num_samples * m_num_clusters) + 
        sizeof(int) * m_num_samples;
    m_clusters = (DataType *)malloc(mem_size);
    m_membership = (float *)(m_clusters + m_num_clusters * m_num_features);
    m_labels = (int *)(m_membership + m_num_samples * m_num_clusters);

    int cache_size = m_num_samples * m_num_clusters;
    mem_size = sizeof(DataType) * (m_num_samples * num_features) + sizeof(DataType) * (m_num_clusters * m_num_features) + 
        sizeof(float) * (m_num_samples * m_num_clusters) + sizeof(float) * (m_num_samples * m_num_clusters) + 
        sizeof(int) * m_num_samples + sizeof(float) * (1 + cache_size);
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, mem_size));
    d_clusters = (DataType *)(d_data + m_num_samples * num_features);
    d_membership = (float *)(d_clusters + m_num_clusters * m_num_features);
    d_distances = (float *)(d_membership + m_num_samples * m_num_clusters);
    d_labels = (int *)(d_distances + m_num_samples * m_num_clusters);
    d_loss = (float *)(d_labels + m_num_samples);

    CHECK_CUDA_ERROR(cudaMemcpy(d_membership, h_membership, sizeof(float) * (m_num_samples * m_num_clusters), cudaMemcpyHostToDevice));

    printf("num_samples: %d  num_clusters: %d  num_features: %d\n", num_samples, num_clusters, num_features);
}

template <typename DataType>
FuzzyCmeans<DataType>::~FuzzyCmeans()
{
    free(m_clusters);
    CHECK_CUDA_ERROR(cudaFree(d_data));
}
```

### 4.3 launchFit 函数
`FuzzyCmeans` 类的成员函数 `fit` 的主要计算逻辑都在 `launchFit` 函数中，根据前面介绍的计算逻辑，`launchFit` 函数中包含两个主要的计算步骤：根据隶属度矩阵计算聚类中心、计算目标函数值并更新隶属度矩阵。这两个步骤分别对应 `launchUpdateClustersKernel` 和 `launchCalculateLossAndUpdateMembershipKernel` 两个函数，其中为了计算出总的目标函数值 $J(t)$，又另外调用了一个 `cubDeviceReduce` 函数用于设备级别的规约求和。

```cuda
template <typename DataType>
void launchFit(const DataType *d_data, DataType *d_clusters, float *d_membership, float *d_distances, float *d_loss, char *d_cache_buf, 
    const int num_clusters, const int num_samples, const int num_features, const float scale, cudaStream_t calculate_stream)
{
    launchUpdateClustersKernel<DataType>(d_data, d_membership, d_clusters, num_features, num_samples, num_clusters, scale, calculate_stream);

    launchCalculateLossAndUpdateMembershipKernel<DataType>(d_data, d_clusters, d_membership, d_distances, num_features, num_clusters,
        num_samples, scale, calculate_stream);
    cubDeviceReduce<SumOp, float>(d_distances, d_loss, num_samples * num_clusters, 0.0f, d_cache_buf, calculate_stream);
}
```

### 4.4 launchUpdateClustersKernel 函数
`launchUpdateClustersKernel` 函数的目的是根据隶属度矩阵计算聚类中心，根据公式可知，这其实就是一个加权平均求和的过程，求和的维度是样本数量的维度，而在样本特征数量、聚类中心数量维度上都是完全并行的，所以不妨把网格大小设置为 `[num_features, num_clusters]`，每个 block 内完成 `num_samples` 个数据的规约求和。

```cuda
/**
 * @brief 根据隶属度矩阵求聚类中心
 *  gird: [num_features, num_clusters, 1]   block: [block_size, 1, 1]
 * @param d_data           [num_samples, num_features] 
 * @param d_clusters       [num_clusters, num_features]     聚类中心 
 * @param membership       [num_clusters, num_samples]      隶属度矩阵
 * @param num_features
 * @param num_samples
 * @param scale                                             模糊系数
 * @return void
*/
template <typename DataType>
__global__ void updateClustersKernel(const DataType *d_data, 
    const float *membership,
    DataType *d_clusters, 
    const int num_features,
    const int num_samples, 
    const float scale)
{
    int cluster_id = blockIdx.y;
    int feature_id = blockIdx.x;
    float val = 0.0f;
    float u_ij_m_sum = 0.0f;
    float u_ij_m;
    for (int i=threadIdx.x; i<num_samples; i+=blockDim.x) {
        u_ij_m = __powf(__ldg(membership + cluster_id * num_samples + i), scale);
        val += u_ij_m * __ldg(d_data + i * num_features + feature_id);
        u_ij_m_sum += u_ij_m;
    }
    __syncthreads();

    val = blockAllReduce<SumOp, DataType>(val);
    u_ij_m_sum = blockAllReduce<SumOp, DataType>(u_ij_m_sum);
    if (threadIdx.x == 0) {
        d_clusters[cluster_id * num_features + feature_id] = (DataType)(val / u_ij_m_sum);
    }
}

template <typename DataType>
void launchUpdateClustersKernel(const DataType *d_data, 
    const float *membership,
    DataType *d_clusters, 
    const int num_features,
    const int num_samples,
    const int num_clusters,
    const float scale,
    cudaStream_t stream)
{
    dim3 grid(num_features, num_clusters);
    dim3 block(block_size);
    updateClustersKernel<DataType><<<grid, block, 0, stream>>>(d_data, membership, d_clusters, num_features, num_samples, scale);
}
```
在核函数内部定义了两个浮点变量 `val`、`u_ij_m_sum`，分别用来存储公式的分子、分母的计算结果。首先取出隶属度矩阵中 `cluster_id` 和当前线程处理的样本的隶属度 $u_{ij}$，再计算 $m$ 次幂得到 `u_ij_m`（相当于 $u_{ij}^m$，即分母），再乘以对应样本的特征分量得到分子 $u_{ij}^m x_i$，最后对分子分母分别规约求和，在线程 ID 为 0 的线程内把分子分母相除的计算结果存入 `d_clusters`。

### 4.5 launchCalculateLossAndUpdateMembershipKernel 函数
`launchCalculateLossAndUpdateMembershipKernel` 函数顾名思义有两个计算任务：计算目标函数值（损失）、更新隶属度矩阵。

```cuda
/**
 * @brief 根据聚类中心计算损失，并求新的隶属度矩阵
 *  gird: [num_samples, 1, 1]   block: [block_size, 1, 1]
 * @param d_data           [num_samples, num_features] 
 * @param d_clusters       [num_clusters, num_features]     聚类中心 
 * @param membership       [num_clusters, num_samples]      隶属度矩阵  
 * @param distances        [num_samples, num_clusters]      distance 矩阵  
 * @param scale                                             模糊系数
 * @return void
*/
template <typename DataType>
__global__ void calculateDistancesAndUpdateMembershipKernel(const DataType *d_data, 
    const DataType *d_clusters, 
    float *membership,
    float *distances,
    const int num_features,
    const int num_clusters,
    const int num_samples,
    const float scale)
{
    const DataType *sample = d_data + blockIdx.x * num_features;
    float distance_ij;
    __shared__ float s_distance_ik[256];

    for (int j=0; j<num_clusters; ++j) {
        const DataType *cluster = d_clusters + j * num_features;
        distance_ij = calculateDistance<DataType>(sample, cluster, num_features);
        s_distance_ik[j] = distance_ij;
    }
    __syncthreads();

    if (threadIdx.x < num_clusters) {
        float old_u_ij_m = __powf(__ldg(membership + threadIdx.x * num_samples + blockIdx.x), scale);
        distances[blockIdx.x * num_clusters + threadIdx.x] = s_distance_ik[threadIdx.x] * old_u_ij_m;

        float distance_ik_sum = 0.0f;
        for (int j=0; j<num_clusters; ++j) {
            distance_ik_sum += __powf((s_distance_ik[threadIdx.x] / s_distance_ik[j]), (1.0f/(scale - 1.0f))); 
        }
        float u_ij = 1.0 / distance_ik_sum;
        membership[threadIdx.x * num_samples + blockIdx.x] = u_ij;
    }
}

template <typename DataType>
void launchCalculateLossAndUpdateMembershipKernel(const DataType *d_data, 
    const DataType *d_clusters, 
    float *membership,
    float *distances,
    const int num_features,
    const int num_clusters,
    const int num_samples,
    const float scale, 
    cudaStream_t stream)
{
    assert(num_clusters <= 256);
    calculateDistancesAndUpdateMembershipKernel<DataType><<<num_samples, block_size, 0, stream>>>(d_data, d_clusters, membership, 
        distances, num_features, num_clusters, num_samples, scale);
}
```
根据聚类中心计算损失时，核心就是计算每个样本到聚类中心的欧式距离，要计算距离显然是要在 `num_features` 维度上进行规约求和，这里我们假设聚类特征数量大于 `256`，因此 `block_size` 直接取 `128` 或 `256` 即可。而对于特征数量较小比如不超过 `64` 的场景，通常为了保证 SM 占用率足够高，block 的维度可以设置为 `[32, 4]`，在 `num_features` 维度上规约求和时采用束内规约即可，此时一个 block 可以处理 `4` 个样本，这种场景的代码建议感兴趣的读者自行实现，本文不做讨论。

核函数内部定义了一个共享内存数组变量 `s_distance_ik`，长度为 `256`，即不支持  `num_clusters` 超过 `256` 的场景，用于存储每个样本到所有聚类中心的距离，注意这里所说的距离其实是欧氏距离的平方。距离计算函数 `calculateDistance` 的代码如下：
```cuda
template <typename DataType>
__inline__ __device__ float calculateDistance(const DataType *sample, const DataType *cluster, const int num_features)
{
    float sub_square_sum = 0.0f;
    float sub_val;

#pragma unroll
    for (int idx=threadIdx.x; idx<num_features; idx+=blockDim.x) {
        sub_val = (float)(__ldg(sample + idx) - __ldg(cluster + idx));
        sub_square_sum += sub_val * sub_val;
    }
    __syncthreads();

    sub_square_sum = blockAllReduce<SumOp, float>(sub_square_sum);
    return sub_square_sum;
}
```
得到距离平方之后按照公式要乘以一个权重 $u_{ij}^m$ 再进行规约求和，规约长度为 `num_samples * num_clusters`，由于当前 Kernel 的网格维度为 `[num_samples, 1, 1]`，由于 block 内可以基于共享内存交换数据，因此适合处理一些单个样本维度的计算，而不适宜整体样本的计算，我们计算损失的规约任务放在其他函数中进行，这里只是在线程 ID 小于 `num_clusters` 的线程中对距离平方乘以一个 $u_{ij}^m$。

接着根据公式，通过一个循环，计算出 $\sum _{k=1}^c (\frac{d_{ij}}{d_{ik}})^{\frac{2}{m-1}}$，再取倒数得到当前样本对于各个聚类中心的隶属度，至此隶属度矩阵计算完毕。

### 4.6 cubDeviceReduce 函数
`cubDeviceReduce` 函数要解决的是一个全局内存数组变量的设备级规约问题，这里采用 cub 库的 Reduce API，关于数组规约问题，笔者在之前的文章中也有提及，通常是采用两级规约的方式，先进行块内规约，这样就把一个长度为 `N` 的数组规约到了一个长度为 `block_size` 的数组内，再进行一次块内规约得到最终结果，这里会使用到一个临时数组存储规约的中间结果（即长度为 `block_size` 的数组）。

而 cub 库中的 Reduce API 也涉及临时空间，库示例代码如下：
```cuda
// Determine temporary device storage requirements
void     *d_temp_storage = nullptr;
size_t   temp_storage_bytes = 0;
cub::DeviceReduce::Reduce(
  d_temp_storage, temp_storage_bytes,
  d_in, d_out, num_items, sum_op, init);

// Allocate temporary storage
cudaMalloc(&d_temp_storage, temp_storage_bytes);

// Run reduction
cub::DeviceReduce::Reduce(
  d_temp_storage, temp_storage_bytes,
  d_in, d_out, num_items, sum_op, init);

// d_out <-- [0]
```
主要涉及一个临时全局内存空间 `d_temp_storage` 以及对应的空间大小 `temp_storage_bytes`。由于在规约之前我们并不知道需要的空间大小是多少，所以需要先把 `temp_storage_bytes` 初始化为 `0` 并执行一次 `cub::DeviceReduce::Reduce` 函数，此时 API 并不会执行规约逻辑，而是会通过引用传递的方式返回一个 `temp_storage_bytes` 的值，然后根据这个返回的 `temp_storage_bytes` 申请一块全局内存绑定到 `d_temp_storage`，最后再执行一次 `cub::DeviceReduce::Reduce` 函数得到规约结果。

可以发现，上面的示例代码中两次 API 执行过程中还会涉及一个动态全局内存分配操作，这其实会损失一部分性能，因此笔者把这个临时的全局内存分配放在了 `FuzzyCmeans` 类初始化中，相当于提前分配了，然后第二次调用 `cub::DeviceReduce::Reduce` 函数时用 `d_cache_buf` 代替了 `d_temp_storage`，避免了临时分配内存的性能损失，具体代码如下。

```cuda
template <template <typename T> class ReductionOp, typename T>
void cubDeviceReduce(const T *d_in, T *d_out, const int num_items, T init, char *d_cache_buf, cudaStream_t stream)
{
    auto op_func = ReductionOp<T>();
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    // cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op_func, init, stream);
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op_func, init, stream);

    // Run reduction, use the preallocte buffer instead of allocate temporary storage
    cub::DeviceReduce::Reduce((void *)d_cache_buf, temp_storage_bytes, d_in, d_out, num_items, op_func, init, stream);
}
```

## 5 小结
本文提供了一个基于 CUDA 的 Fuzzy C-means 算法的简单实现，其中还有不少优化空间，比如合并访存、Kernel 特化等等，有兴趣的读者可以继续深入或者留言讨论。

本文的代码地址如下：
>https://github.com/caiwanxianhust/ClusteringByCUDA/tree/master