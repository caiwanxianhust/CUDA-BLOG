#! https://zhuanlan.zhihu.com/p/646999265
# 【CUDA编程】基于CUDA的Kmeans算法的简单实现
**写在前面**：本文主要介绍如何使用CUDA并行计算框架实现机器学习中的Kmeans算法，Kmeans算法的原理见笔者的上一篇文章[【机器学习】K均值聚类算法原理](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484421&idx=1&sn=f6e5c4fa2a4f289766bcc665f0e8a5dd&chksm=e92781bcde5008aa0d3ecbe8bc61901b219e453167a7027b7d94a6779729a773ba1f860e1e68&token=1840879869&lang=zh_CN#rd)，本文重点在于并行实现的过程。

## 1 Kmeans 聚类算法过程回顾
输入：`n` 个样本集合 `X`；  
输出：样本集合的聚类 `C`；  
1. 初始化。令 `t=0`，选择初始化的 `k` 个样本作为初始聚类中心；
2. 对样本进行聚类。针对数据集中每个样本 $x_i$ 计算它到 `k` 个聚类中心的距离并将其分到距离最小的聚类中心所对应的类中，构成聚类结果 `C(t)`；
3. 计算新的类中心。对聚类结果 `C(t)`，计算每个类中的样本均值，作为类的新的中心；
4. 如果迭代满足停止条件，输出聚类结果 `C(t)`，作为最终结果。否则，令 `t=t+1`，返回2 。

## 2 数据集
### 2.1 数据生成
本文数据集是笔者通过 Sklearn 库生成的数据集，共有 1000000 条数据，每条数据由 `100` 个特征构成，并对应一个数据标签，总共分为 `4` 个类别，本文不涉及 `K` 的选取工作，所以后面的算法中笔者默认 `K=4`。数据生成代码如下：
```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000000, n_features=100, n_classes=4, n_clusters_per_class=4, random_state=1024, n_informative=8,
)
X.shape, y.shape

"""
# output:
((1000000, 100), (1000000,))
"""
```
数据占用的磁盘空间约 2.4G，保存为csv格式用逗号隔开。
```python
import numpy as np

xy = np.concatenate((X, y.reshape(-1, 1)), axis=1)
np.savetxt("sample_1e6_fea_100_class_4_lable_1_ninfo_8.csv", xy, delimiter=",")
```

### 2.2 数据读取
因为后面我们要使用 C++ 实现算法，所以这里读取就不用 Python 了，笔者给出 C++ 版本的数据读取代码：
```c++
void readCoordinate(float *data, int *label, const int n_features, int &n) {
    std::ifstream ifs;
    ifs.open("./sample_1e6_fea_100_class_4_lable_1_ninfo_8.csv", std::ios::in);
    if (ifs.fail()) {
        std::cout << "No such file or directory: sample_1e6_fea_100_class_4_lable_1_ninfo_8.csv" << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream sstream(line);
        if (line.empty()) continue;
        int m = 0;
        std::string s_fea;
        while (std::getline(sstream, s_fea, ',')) {
            if (m < n_features) data[n * n_features + m] = std::stod(s_fea);
            else label[n] = std::stoi(s_fea);
            m++;
        }
        n++;
    }
    ifs.close();
}
```
参数说明：
- `data`：数据集，不包括标签，读取了 `n` 行 `n_features` 列数据全部放进一个一维数组中，`data` 是首地址
- `label`：数据标签，总共包含 `n` 个元素，`label` 是首地址
- `n_features`：特征数量
- `n`：数据行数，引用传递

## 3 基于CPU编程的 Kmeans 聚类算法
### 3.1 Kmeans 类基本结构
在进行 CUDA 并行之前，我们基于CPU编程思想，使用 C++ 语言对 Kmeans 算法做一个简单实现。下面看一下 `Kmeans` 类的基本结构：
```cuda
class Kmeans {
public:
    Kmeans(int numClusters, int numFeatures, float *clusters, int nsamples);
    Kmeans(int numClusters, int numFeatures, float *clusters, int nsamples, 
        int maxIters, float eplison);
    ~Kmeans();
    virtual void getDistance(const float *v_data); 
    virtual void updateClusters(const float *v_data);
    virtual void fit(const float *v_data);

    float *m_clusters; //[numClusters, numFeatures]
    int m_numClusters;
    int m_numFeatures;
    float *m_distances; // [nsamples, numClusters]
    int *m_sampleClasses; // [nsamples, ]
    int m_nsamples;
    float m_optTarget;
    int m_maxIters;
    float m_eplison;
private:
    Kmeans(const Kmeans& model);
    Kmeans& operator=(const Kmeans& model);
};
```
这里为了方便起见所有成员变量全部设置为公有属性，否则还得开发 `set`、`get` 接口访问，我有些懒。。。  
成员变量说明：  
- `float *m_clusters`：`[numClusters, numFeatures]`，用于存储当前各个类的中心点坐标
- `int m_numClusters`：类别数量，就是 `k` 值
- `int m_numFeatures`：特征数量
- `float *m_distances`：`[nsamples, numClusters]`，用于存储每个样本到每个类的中心两两之间的距离
- `int *m_sampleClasses`：`[nsamples, ]`，记录每个样本属于的类别编号
- `int m_nsamples`：样本数量
- `float m_optTarget`：优化目标值，这里就是 `loss`
- `int m_maxIters`：迭代次数，超过次数停止迭代
- `float m_eplison`：目标阈值，两次 `loss` 相差不超过 `m_eplison` 停止迭代

### 3.2 构造函数和析构函数
这里笔者声明并定义了2个有参构造函数，用于初始化一些成员变量，并且拒绝编译器默认提供的无参构造函数，防止没有初始化中心点的坐标导致后面计算出BUG。  
另外笔者声明了一个私有的拷贝构造函数，但是故意没有定义。主要目的是拒绝编译器默认提供的拷贝构造函数同时拒绝拷贝构造操作，原因是成员变量中有很多指针类的成员，如果使用拷贝构造，需要写深拷贝版本的构造函数，代码量有点大，我有些懒。。。  
同理笔者也把赋值操作符重载函数也声明为私有并且不予实现。。。
```cuda
Kmeans::Kmeans(int numClusters, int numFeatures, float *clusters, int nsamples):
    m_numClusters(numClusters), m_numFeatures(numFeatures), m_maxIters(50),
    m_optTarget(1e7), m_eplison(0.001), m_nsamples(nsamples) {
    m_clusters = new float[numClusters * numFeatures];
    for (int i=0; i<this->m_numClusters * this->m_numFeatures; ++i) {
        this->m_clusters[i] = clusters[i];
    } 
    m_distances = new float[nsamples * numClusters]{0.0};
    m_sampleClasses = new int[nsamples]{0};
} 

Kmeans::Kmeans(int numClusters, int numFeatures, float *clusters, int nsamples, 
    int maxIters, float eplison):
    m_numClusters(numClusters), m_numFeatures(numFeatures), m_maxIters(maxIters),
    m_optTarget(1e7), m_eplison(eplison), m_nsamples(nsamples) {
    m_clusters = new float[numClusters * numFeatures];
    for (int i=0; i<this->m_numClusters * this->m_numFeatures; ++i) {
        this->m_clusters[i] = clusters[i];
    }
    m_distances = new float[nsamples * numClusters]{0.0};
    m_sampleClasses = new int[nsamples]{0};
}
```
析构函数很简单，delete掉几个堆上的指针即可，防止内存泄漏。
```cuda
Kmeans::~Kmeans() {
    if (m_clusters) delete[] m_clusters;
    if (m_distances) delete[] m_distances;
    if (m_sampleClasses) delete[] m_sampleClasses;
}
```

### 3.3 getDistance 函数
针对数据集中每个样本 $x_i$ 计算它到 `k` 个聚类中心的距离，同时找出距离最近的中心点，更新样本对应的类别，最后将所有样本与其最近中心的距离求和作为 `loss`。

```cuda
void Kmeans::getDistance(const float *v_data) {
    /*
        v_data: [nsamples, numFeatures, ]
    */

    float loss = 0.0;
    for (int i=0; i<m_nsamples; ++i) {
        float minDist = 1e8;
        int minIdx = -1;
        for (int j=0; j<m_numClusters; ++j) {
            float sum = 0.0;
            for (int k=0; k<m_numFeatures; ++k) {
                sum += (v_data[i * m_numFeatures + k] - m_clusters[j * m_numFeatures + k]) * 
(v_data[i * m_numFeatures + k] - m_clusters[j * m_numFeatures + k]);
            }
            this->m_distances[i * m_numClusters + j] = sqrt(sum);
            if (sum <= minDist) {
                minDist = sum;
                minIdx = j;
            }
        }
        m_sampleClasses[i] = minIdx;
        loss += m_distances[i * m_numClusters + minIdx];
    }
    m_optTarget = loss;
}
```
### 3.4 updateClusters 函数
根据聚类结果 `m_sampleClasses`，计算每个类中的样本均值，作为类的新的中心，然后更新到 `m_clusters` 上。
```cuda
void Kmeans::updateClusters(const float *v_data) {
    for (int i=0; i<m_numClusters * m_numFeatures; ++i) this->m_clusters[i] = 0.0;
    for (int i=0; i<m_numClusters; ++i) {
        int cnt = 0;
        for (int j=0; j<m_nsamples; ++j) {
            if (i != m_sampleClasses[j]) continue;
            for (int k=0; k<m_numFeatures; ++k) {
                this->m_clusters[i * m_numFeatures + k] += v_data[j * m_numFeatures + k];
            }
            cnt++;
        }
        for (int ii=0; ii<m_numFeatures; ii++) this->m_clusters[i * m_numFeatures + ii] /= cnt;
    }
}
```

### 3.5 fit 函数
`fit` 函数是聚类算法的启动函数，函数内调用 `getDistance` 函数和 `updateClusters` 函数，并通过停止条件控制迭代过程。
```cuda
void Kmeans::fit(const float *v_data) {
    float lastLoss = this->m_optTarget;
    for (int i=0; i<m_maxIters; ++i) {
        this->getDistance(v_data);
        this->updateClusters(v_data);
        if (std::abs(lastLoss - this->m_optTarget) < this->m_eplison) break;
        lastLoss = this->m_optTarget;
        std::cout << "Iters: " << i+1 << "  current loss : " << m_optTarget << std::endl;
    }
}
```
两个停止条件：迭代次数不超过 `m_maxIters`；相邻的两次迭代结果 `loss` 绝对值之差小于 `m_eplison` 停止迭代。


### 3.7 开始聚类
运行 `fit` 函数后，迭代 `33` 轮后相邻的两次迭代结果 `loss` 绝对值之差小于 `m_eplison`，模型收敛，运行时间约 `17322.8 ms`。

```cuda
Using CPU:
*********starting fitting*********
Iters: 1  current loss : 1.43821e+07
Iters: 2  current loss : 1.09721e+07
Iters: 3  current loss : 1.08641e+07
Iters: 4  current loss : 1.08439e+07
Iters: 5  current loss : 1.08367e+07
Iters: 6  current loss : 1.08326e+07
Iters: 7  current loss : 1.08301e+07
Iters: 8  current loss : 1.08287e+07
Iters: 9  current loss : 1.08277e+07
Iters: 10  current loss : 1.08271e+07
Iters: 11  current loss : 1.08266e+07
Iters: 12  current loss : 1.08264e+07
Iters: 13  current loss : 1.08262e+07
Iters: 14  current loss : 1.08261e+07
Iters: 15  current loss : 1.08261e+07
Iters: 16  current loss : 1.08259e+07
Iters: 17  current loss : 1.08259e+07
Iters: 18  current loss : 1.08259e+07
Iters: 19  current loss : 1.08259e+07
Iters: 20  current loss : 1.08259e+07
Iters: 21  current loss : 1.08259e+07
Iters: 22  current loss : 1.08259e+07
Iters: 23  current loss : 1.08259e+07
Iters: 24  current loss : 1.08259e+07
Iters: 25  current loss : 1.08259e+07
Iters: 26  current loss : 1.0826e+07
Iters: 27  current loss : 1.08259e+07
Iters: 28  current loss : 1.08259e+07
Iters: 29  current loss : 1.0826e+07
Iters: 30  current loss : 1.0826e+07
Iters: 31  current loss : 1.08259e+07
Iters: 32  current loss : 1.08259e+07
Iters: 33  current loss : 1.08259e+07
Time = 17322.8 ms.
********* final clusters**********
clusters 1
0.00135362  -0.00436829  0.00143691  -0.00433175  -0.00126249  0.00293742  0.00362213  -0.135102  -0.00270963  -0.000969095  0.00106209  0.00148805  -0.00349138  -0.668611  -0.000208263  0.00241531  0.00317462  -0.606494  6.02375e-05  0.00305834  0.00333136  0.00250743  0.00250087  -0.000714736  -0.00118934  -0.00222592  0.00113979  -1.06601  3.01919e-06  6.18578e-05  0.180001  0.00130381  0.00158767  0.00111992  0.00165197  -0.0010685  -0.000516302  -0.00175122  -0.00136144  -0.0010593  -0.000731016  0.000308565  0.00300345  0.000228965  0.00332303  0.00146533  0.00345964  0.00359196  -0.00315863  -0.00127007  -0.00210936  0.000320757  0.0346409  -0.00307558  -0.00151595  0.00407643  0.00151323  0.00182943  -3.73458e-05  0.00370085  0.000265956  -0.00168623  0.00503313  -0.477866  -0.00161701  -0.0023707  -0.00117495  1.02645  -0.000643958  0.00071756  0.00413504  -0.00155079  -0.00164189  0.00306905  -0.00113062  -0.00523529  -0.00243497  -0.00279149  0.00149723  -0.0023071  -0.00572529  -0.000549263  -0.00207719  -0.000386163  -0.820618  -0.000631672  -0.00106813  0.00326451  -0.000328932  -0.000478803  0.00258156  -0.000586523  -0.00250226  0.00141438  -0.00215083  0.00265429  0.000552067  -0.550796  0.000477271  -0.00196779
clusters 2
-0.00194203  -0.00164453  -0.000122132  -0.00168593  -0.00165588  0.0010895  0.00106805  -4.79281  0.000424808  -0.00206315  0.000322566  0.00471343  0.000834374  -2.67674  -0.000497463  -0.00208716  0.00262159  -0.316545  0.00291554  -0.00118353  0.00424287  0.000376611  -0.000259759  -0.000911404  0.00196577  -0.00163495  -0.000899112  0.867919  -0.00148784  -0.000205066  -1.31622  -0.00369622  -8.8451e-05  -0.00276815  0.00429006  -0.00391626  -0.000681551  -0.00245076  0.00162618  0.00178954  -0.00185512  -0.000394879  -0.00210911  0.00106342  0.00417279  -0.00205136  0.00286898  -0.00167855  0.0015179  0.000682002  0.00179277  0.00243299  -1.75739  -0.00108967  -0.000826672  -0.00120179  -0.00131274  -0.000817148  -0.00258151  -0.00392083  0.00024333  0.00282693  -0.000999048  -0.951875  -0.000508  0.000184719  0.000364489  0.58316  -0.00462551  0.00326746  0.000437131  0.000515362  -0.002355  0.00272101  0.00281304  0.00347533  -0.00418443  0.000702084  -0.00253757  0.000268111  -0.0011675  0.00151085  -0.00274651  -0.00426217  -1.02024  0.00268259  0.00262944  -0.00546804  0.00226117  0.00177111  0.0023911  0.00222939  0.00164531  0.000123123  0.00176449  -1.4516e-05  -0.00172494  -1.13087  -0.000899755  -0.00128382
clusters 3
0.0012379  -0.0038503  0.00135718  0.00174353  -0.00110076  -9.04162e-05  -0.000781017  -1.31877  0.00275596  0.000867588  -0.0012499  -0.000173576  0.00032865  0.555334  -0.000648075  -0.00181736  0.000878216  -0.0561488  0.00209253  -0.00188699  -0.00112445  -0.000509834  -0.00798074  -0.000916144  -0.00464598  0.00205881  -0.00156118  1.2207  0.00145849  0.000248683  -0.781434  -0.00306122  -0.00333605  0.000267487  0.00199873  0.00220486  0.00242323  0.000626134  0.00032185  0.000723335  0.00178548  0.000645121  -0.00196796  -0.000394865  -0.000267085  0.000617321  -0.00124905  0.0012794  0.00138048  0.00228282  0.00119127  0.00391252  -0.64559  0.00271288  0.00475292  0.00178865  0.00174548  0.00240863  0.000738086  -0.000265929  -0.00272517  -0.00293344  -0.00072009  1.15287  -0.00100652  0.0022417  -0.00194703  0.136672  -0.000227936  0.00525902  0.000178473  -0.0030583  0.000428586  0.00144548  -0.000354203  0.000109064  -0.000134745  0.000173136  -0.00169671  -0.00172857  -0.000666526  -0.00260149  -0.00144469  0.000671148  0.978586  0.00125423  -0.0018544  -0.00194628  0.000836093  0.000308424  0.00179499  0.00128853  -0.00298858  -0.00394571  -0.00318117  0.00137591  -0.000815955  0.249268  -0.000204362  0.00293572
clusters 4
0.00310996  0.000210244  0.00161979  0.00146684  -0.00103305  -0.00437169  0.00123377  3.89196  0.00101941  -0.000431054  -0.00211129  -3.95512e-05  0.00101944  2.09593  -0.00179757  0.000807649  0.00177102  0.0481786  0.00218751  -0.00265079  0.00186006  -0.0039261  0.00386576  -0.000302028  0.00181874  -0.0030542  4.51879e-05  -1.39049  0.00103703  -0.000223172  0.869388  0.000801458  0.00189288  0.000404353  -0.00183143  -0.00163687  0.0016658  -0.000226338  -0.00226052  0.00170732  0.00135896  0.000154566  0.00659345  -0.00105883  0.00317892  0.00210709  -0.00138408  -0.00284854  0.000619164  -0.00306363  -0.000814287  0.000874715  0.817652  -0.00162136  -0.00501512  0.000184863  -0.00099965  -0.00135189  0.00422345  0.00134315  0.00269236  0.00227658  0.000663711  0.893527  0.000625904  0.00273823  0.00417765  0.161069  0.000731467  0.00163335  0.000836738  0.00151034  -0.000870908  0.000325944  -0.00121529  -6.09558e-05  0.000524801  0.00193909  0.00084036  -0.000324457  -0.00104298  -0.00246107  -0.000233543  0.00020954  1.01607  -0.000241049  -0.000903253  -0.000302991  0.00322567  -0.00392602  0.00354434  0.000701865  -0.000214893  -0.00314099  -0.00039492  -0.000816782  -0.000869031  1.02608  0.00133142  -0.000392192
sampleClasses_10 1
1  0  2  1  0  3  2  0  2  2
```

## 4 基于CUDA编程的 Kmeans 聚类算法
### 4.1 KmeansGPU 类基本结构
`KmeansGPU` 类继承了 `Kmeans` 类，新增了 `6` 个成员变量，另外重新定义了 `getDistance`、`updateClusters`、`fit`三个成员函数。
```cuda
class KmeansGPU: public Kmeans {
public:
    KmeansGPU(int numClusters, int numFeatures, float *clusters, int nsamples);
    KmeansGPU(int numClusters, int numFeatures, float *clusters, int nsamples, 
        int maxIters, float eplison);
    virtual void getDistance(const float *d_data); 
    virtual void updateClusters(const float *d_data);
    virtual void fit(const float *v_data);

    float *d_clusters; // [numClusters, numFeatures]
    int *d_sampleClasses;
    float *d_distances;
    float *d_minDist; // [nsamples, ]
    float *d_loss; // [nsamples, ]
    int *d_clusterSize; //[numClusters, ]

private:
    KmeansGPU(const Kmeans& model);
    KmeansGPU& operator=(const Kmeans& model);
};
```
成员变量说明：  
- `float *d_clusters`：`[numClusters, numFeatures]`，与 `m_clusters` 作用相同，用于存储当前各个类的中心点坐标，但这是一个设备端的变量。
- `int *d_sampleClasses`：`[nsamples, ]`，与 `m_sampleClasses` 作用相同，用于记录每个样本属于的类别编号，但这是一个设备端的变量。
- `float *d_distances`：`[nsamples, numClusters]`，与 `m_distances` 作用相同，用于存储每个样本到每个类的中心两两之间的距离，但这是一个设备端变量。
- `float *d_minDist`：`[nsamples, ]`，用于存储每个样本到对应类中心的距离，用于后面的规约操作，这是一个设备端变量。
- `float *d_loss`：`[nsamples, ]`，用于存储 `d_minDist` 的规约结果，这是一个设备端变量。
- `int *d_clusterSize`：`[numClusters, ]`，用于存储每个类的样本数量，方便求样本均值更新中心点坐标，这是一个设备端变量。

### 4.1 构造函数和析构函数
`KmeansGPU` 类新增的 `6` 个成员变量，无需在构造函数里初始化，所以这里构造函数直接调用基类的构造函数即可。
```cuda
KmeansGPU::KmeansGPU(int numClusters, int numFeatures, float *clusters, int nsamples):
    Kmeans(numClusters, numFeatures, clusters, nsamples) {}

KmeansGPU::KmeansGPU(int numClusters, int numFeatures, float *clusters, int nsamples, 
        int maxIters, float eplison):
    Kmeans(numClusters, numFeatures, clusters, nsamples, 
        maxIters, eplison) {}
```

### 4.2 getDistance 函数
我们知道核函数有两个重要特点：1、核函数不能作为类的成员函数。2、成员函数不能直接调用核函数。  
所以如果要在成员函数中实现CUDA并行计算，需要在核函数外再套一层函数，这里我们定义一个全局函数 `calDistWithCuda`，使用 `getDistance` 调用 `calDistWithCuda`，在 `calDistWithCuda` 内部调用核函数完成并行计算。
```cuda
template <typename T>
__global__ void init(T *x, const T value, const int N) {
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N) x[n] = value;
}

void calDistWithCuda(
    const float *d_data, 
    float *d_clusters, 
    float *d_distance,
    int *d_sampleClasses,
    float *d_minDist,
    float *d_loss,
    int *d_clusterSize,
    const int numClusters,
    const int nsamples,
    const int numFeatures) {

    init<int><<<1, 128>>>(d_clusterSize, 0, numClusters);
    int smem = sizeof(float) * 128;
    cudaStream_t streams[20];
    for (int i = 0; i < numClusters; i++)
    {
        CHECK(cudaStreamCreate(&(streams[i])));
    }
    for (int i = 0; i < numClusters; i++)
    {
        calDistKernel<<<nsamples, 128, smem, streams[i]>>>(d_data, d_clusters, 
d_distance, numClusters, i, nsamples, numFeatures);
    }
    for (int i=0; i< numClusters; ++i) {
        CHECK(cudaStreamDestroy(streams[i]));
    }
    int blockSize = 256;
    int gridSize = (nsamples - 1) / blockSize + 1;
    reduceMin<<<gridSize, blockSize, sizeof(int) * blockSize>>>(d_distance, d_sampleClasses, 
d_clusterSize, numClusters, nsamples, d_minDist);
    reduceSum<<<256, 256, sizeof(float) * 256>>>(d_minDist, d_loss, nsamples);
    reduceSum<<<1, 256, sizeof(float) * 256>>>(d_loss, d_loss, 256);
}

void KmeansGPU::getDistance(const float *d_data) {
    calDistWithCuda(d_data, d_clusters, d_distances, d_sampleClasses, d_minDist, 
d_loss, d_clusterSize, m_numClusters, m_nsamples, m_numFeatures);
}
```
#### 4.2.1 计算距离
在 `calDistWithCuda` 函数内部，先调用函数模板 `init<int>` 将 `d_clusterSize` 元素全部初始化为 `0`。然后开始计算每个样本 $x_i$ 到 `k` 个聚类中心的距离，同时找出距离最近的中心点，对于每个中心点来说这个计算过程是完全独立的，所以我们这里用CUDA流的方式进行一个并行操作，将 `K` 个中心的计算并行起来，这一步的核函数为 `calDistKernel`，代码如下：
```cuda
__global__ void calDistKernel(
    const float *d_data, 
    const float *d_clusters, // [numClusters, numFeatures]
    float *d_distance, // [nsamples, numClusters]
    const int numClusters,
    const int clusterNo,
    const int nsamples,
    const int numFeatures) {
    
    int n = threadIdx.x + numFeatures * blockIdx.x;
    int m = threadIdx.x + numFeatures * clusterNo;
    extern __shared__ float s_c[];
    s_c[threadIdx.x] = 0.0;
    if (n < numFeatures * nsamples && threadIdx.x < numFeatures) {
        s_c[threadIdx.x] = powf(d_data[n] - d_clusters[m], 2);
    }
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) s_c[threadIdx.x] += s_c[threadIdx.x + offset];
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) s_c[threadIdx.x] += s_c[threadIdx.x + offset];
        __syncwarp();
    }
    if (threadIdx.x == 0) d_distance[blockIdx.x * numClusters + clusterNo] = sqrt(s_c[0]);

}
```
`calDistKernel` 核函数执行的时候，笔者为每个样本分配一个网格，每个特征值分配一个线程，即一个样本对应一个线程块。核函数内部，先声明了一个共享内存变量 `s_c` 用来存储样本与中心点对应特征差值的平方 $(x_{mi} - c_{m_j}) ^ 2$。平方值计算完毕后就是朴实无华的规约操作，关于并行规约笔者在之前的文章中也有介绍[【CUDA编程】CUDA编程中的并行规约问题](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484415&idx=1&sn=e72656c1c1d8b6e6d7f424572b782969&chksm=e9278646de500f50898dcbcd233e5123a847a7fb5aa60e5cd52922f7143d21c04434b46ed925&token=1840879869&lang=zh_CN#rd)，这里使用折半规约法求平方和。最后当 `threadIdx.x == 0` 时将平方和开平方后写入 `d_distance`。  
值得注意的是，`s_c[threadIdx.x] = 0.0;` 这行代码不能省略，否则当 `threadIdx.x > numfeatures` 时，会把 `s_c[threadIdx.x]` 随机初始化，导致规约计算错误。

#### 4.2.2 找最近的中心点
根据上一步计算结果 `d_distance` 计算出距离每个样本最近的中心点，这里笔者为每个样本分配一个线程，在核函数内简单循环找出距离最近的中心点，记录最近的距离和中心点编号。  
值得关注的是，在访问全局内存 `d_distance` 时会出现非合并访问的问题，导致访问效率变低，这里我们借助只读数据缓存加载函数 `__ldg()` 缓解这个问题。

```cuda
__global__ void reduceMin(
    float *d_distance, 
    int *d_sampleClasses, 
    int *d_clusterSize,
    int numClusters,
    int nsamples,
    float *d_minDist) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < nsamples) {
        float minDist = d_distance[n * numClusters + 0];
        int minIdx = 0;
        float tmp;
        for (int i=1; i<numClusters; i++) {
            tmp = __ldg(&d_distance[n * numClusters + i]);
            if (tmp < minDist) {
                minDist = tmp;
                minIdx = i;
            }
        }
        d_sampleClasses[n] = minIdx;
        d_minDist[n] = minDist;
    }
}
```

#### 4.2.3 对距离规约求 loss
朴实无华的折半规约，计算 `loss`，无须赘述，具体可见笔者的文章：[【CUDA编程】CUDA编程中的并行规约问题](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484415&idx=1&sn=e72656c1c1d8b6e6d7f424572b782969&chksm=e9278646de500f50898dcbcd233e5123a847a7fb5aa60e5cd52922f7143d21c04434b46ed925&token=1840879869&lang=zh_CN#rd)

```cuda
__global__ void reduceSum(
    float *d_minDist, 
    float *d_loss, 
    int nsamples) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float s_y[];
    float y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (; n<nsamples; n+=stride) y += d_minDist[n];
    s_y[threadIdx.x] = y;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncwarp();
    }
    if (threadIdx.x == 0) d_loss[blockIdx.x] = s_y[0];
}
```

### 4.3 updateClusters 函数
和 `getDistance` 函数一样，这里我们定义一个全局函数 `updateClusterWithCuda`，使用 `updateClusters` 调用 `updateClusterWithCuda`，在 `calDistWithCuda` 内部调用核函数完成并行计算。

```cuda
void updateClusterWithCuda(
    const float *d_data, 
    float *d_clusters, 
    int *d_sampleClasses, 
    int *d_clusterSize,
    const int nsamples,
    const int numClusters,
    const int numFeatures) {
    
    init<float><<<1, 1024>>>(d_clusters, 0.0, numClusters * numFeatures);
    int blockSize = 1024;
    int gridSize = (nsamples - 1) / blockSize + 1;
    countCluster<<<gridSize, blockSize>>>(d_sampleClasses, d_clusterSize, nsamples);
    update<<<nsamples, 128>>>(d_data, d_clusters, d_sampleClasses, d_clusterSize, nsamples, numFeatures);
}
```
在更新新的中心点之前，先调用函数模板 `init<float>` 将 `d_clusters` 初始化为 `0` 方便后面的累加操作。

#### 4.3.1 统计各中心的样本数量
根据 `d_sampleClasses` 计算各中心的样本数量，就是一个 `count + groupby` 操作，因为核函数里面存在一个原子操作需要轮流读写，所以这一步甚至我觉得在主机端整个循环甚至还更快一点。当然，这一步也是可以优化的，这里就不讨论了。

```cuda
__global__ void countCluster(int *d_sampleClasses, int *d_clusterSize, int nsamples) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < nsamples) {
        int clusterID = d_sampleClasses[n];
        atomicAdd(&(d_clusterSize[clusterID]), 1);
    }
}
```

#### 4.3.2 更新中心点的坐标
一个样本对应一个线程块，将每个样本的坐标值除以对应类的样本数量然后累加上去，得到最终的中心点坐标，核函数代码如下：
```cuda
__global__ void update(
    const float *d_data, 
    float *d_clusters, 
    int *d_sampleClasses, 
    int *d_clusterSize,
    const int nsamples,
    const int numFeatures) {

    int n = threadIdx.x + numFeatures * blockIdx.x;
    int clusterId = d_sampleClasses[blockIdx.x];
    int clustercnt = d_clusterSize[clusterId];
    if (threadIdx.x < numFeatures) {
        atomicAdd(&(d_clusters[clusterId * numFeatures + threadIdx.x]), d_data[n] / clustercnt);
    }
}
```
至此就完成了一轮计算，得到聚类结果和新的中心点坐标。

### 4.4 fit 函数
与 `Kmeans::fit()` 相比，`KmeansGPU::fit()` 主体部分没有什么变化，只是多了申请设备内存、主机端与设备端数据传输、主机端和设备端内存释放等3个操作。具体代码如下：

```cuda
void KmeansGPU::fit(const float *v_data) {
    float *d_data;
    int datamem = sizeof(float) * m_nsamples * m_numFeatures;
    int clustermem = sizeof(float) * m_numClusters * m_numFeatures;
    int sampleClassmem = sizeof(int) * m_nsamples;
    int distmem = sizeof(float) * m_nsamples * m_numClusters;
    int *h_clusterSize = new int[m_numClusters]{0};
    float *h_loss = new float[m_nsamples]{0.0};

    CHECK(cudaMalloc((void **)&d_data, datamem));
    CHECK(cudaMalloc((void **)&d_clusters, clustermem));
    CHECK(cudaMalloc((void **)&d_sampleClasses, sampleClassmem));
    CHECK(cudaMalloc((void **)&d_distances, distmem));
    CHECK(cudaMalloc((void **)&d_minDist, sizeof(float) * m_nsamples));
    CHECK(cudaMalloc((void **)&d_loss, sizeof(float) * m_nsamples));
    CHECK(cudaMalloc((void **)&d_clusterSize, sizeof(int) * m_numClusters));

    CHECK(cudaMemcpy(d_data, v_data, datamem, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_clusters, m_clusters, clustermem, cudaMemcpyHostToDevice));

    float lastLoss = 0;
    for (int i=0; i<m_maxIters; ++i) {
        this->getDistance(d_data);
        this->updateClusters(d_data);
        CHECK(cudaMemcpy(h_loss, d_loss, sampleClassmem, cudaMemcpyDeviceToHost));
        this->m_optTarget = h_loss[0];
        if (std::abs(lastLoss - this->m_optTarget) < this->m_eplison) break;
        lastLoss = this->m_optTarget;
        std::cout << "Iters: " << i+1 << "  current loss : " << m_optTarget << std::endl;
    }
    
    CHECK(cudaMemcpy(m_clusters, d_clusters, clustermem, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_sampleClasses, d_sampleClasses, sampleClassmem, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_distances, d_distances, distmem, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_clusters));
    CHECK(cudaFree(d_sampleClasses));
    CHECK(cudaFree(d_distances));
    CHECK(cudaFree(d_minDist));
    CHECK(cudaFree(d_loss));
    CHECK(cudaFree(d_clusterSize));
    delete[] h_clusterSize;
    delete[] h_loss;
}
```

### 4.5 开始聚类
运行 `fit` 函数后，迭代 `22` 轮后相邻的两次迭代结果 `loss` 绝对值之差小于 `m_eplison`，模型收敛，运行时间约 `759.44 ms`。为什么这里只需要 `22` 轮迭代就收敛了而前面C++代码需要 `33` 轮，笔者猜测这应该是GPU单精度计算误差导致，事实上根据迭代记录，其实在第17轮开始就已经基本收敛了，只是我们 `m_eplison` 定的比较小，于是在 1e8 次单精度运算后误差超过了阈值。  
可见基于CUDA的并行计算要比基于循环的C++代码快了大概 `23` 倍，已经是一个可观的提升了。
```cuda
Using CUDA:
*********starting fitting*********
Iters: 1  current loss : 1.4382e+07
Iters: 2  current loss : 1.09722e+07
Iters: 3  current loss : 1.08641e+07
Iters: 4  current loss : 1.08437e+07
Iters: 5  current loss : 1.08365e+07
Iters: 6  current loss : 1.08326e+07
Iters: 7  current loss : 1.08303e+07
Iters: 8  current loss : 1.08287e+07
Iters: 9  current loss : 1.08277e+07
Iters: 10  current loss : 1.0827e+07
Iters: 11  current loss : 1.08266e+07
Iters: 12  current loss : 1.08264e+07
Iters: 13  current loss : 1.08262e+07
Iters: 14  current loss : 1.08261e+07
Iters: 15  current loss : 1.0826e+07
Iters: 16  current loss : 1.0826e+07
Iters: 17  current loss : 1.08259e+07
Iters: 18  current loss : 1.08259e+07
Iters: 19  current loss : 1.08259e+07
Iters: 20  current loss : 1.08259e+07
Iters: 21  current loss : 1.08259e+07
Iters: 22  current loss : 1.08259e+07
Time = 759.44 ms.
********* final clusters**********
clusters 1
0.00123018  -0.0041137  0.00163218  -0.00432372  -0.000945806  0.00308277  0.00370208  -0.15029  -0.00257875  -0.000995057  0.00116238  0.00162593  -0.00347802  -0.671157  -0.000107295  0.00241036  0.00294409  -0.607338  -2.57106e-07  0.00317934  0.00337062  0.00254363  0.00276771  -0.00102519  -0.00107145  -0.00223194  0.00122808  -1.06103  1.7726e-05  0.000150998  0.17382  0.00166559  0.00184371  0.00137186  0.00169097  -0.00062256  -0.000492505  -0.00163299  -0.0010392  -0.00113833  -0.000679217  0.000422278  0.00306267  8.49941e-05  0.00323317  0.00161266  0.00334968  0.00355647  -0.00307676  -0.00116147  -0.00221317  9.51576e-05  0.0353675  -0.0033152  -0.0014949  0.00409992  0.00177637  0.0017666  -8.12544e-05  0.00355678  0.00035303  -0.00177558  0.00489658  -0.475213  -0.00162473  -0.00221053  -0.00138114  1.02816  -0.000449021  0.001067  0.00446234  -0.00163512  -0.00143369  0.00272584  -0.00100722  -0.00516115  -0.00237272  -0.00261138  0.00147283  -0.00222827  -0.00582426  -0.000778003  -0.0021275  -0.000506351  -0.825543  -0.00103254  -0.00115987  0.0032816  -0.000154275  -0.00054638  0.00266763  -0.000390763  -0.0027186  0.00148753  -0.00187763  0.00249815  0.000333597  -0.555523  0.00063068  -0.00208013
clusters 2
-0.00181535  -0.00197782  -0.000225024  -0.00177978  -0.00194357  0.0008172  0.000940936  -4.80256  0.000235074  -0.00196392  0.000320223  0.0043042  0.000853165  -2.68272  -0.000401361  -0.00192916  0.00294155  -0.315917  0.00275689  -0.00130876  0.00436599  0.000374993  -0.000172491  -0.000704537  0.00178661  -0.00171318  -0.00109704  0.869553  -0.00144477  -0.000108804  -1.31922  -0.00380671  -0.000284446  -0.00286004  0.00425439  -0.00401153  -0.000849683  -0.00231506  0.00163766  0.00172354  -0.00168308  -0.000679854  -0.00196785  0.00115584  0.00414018  -0.00216509  0.00275576  -0.00195483  0.00140989  0.000712338  0.00194839  0.00271  -1.76096  -0.00099947  -0.000909499  -0.00098245  -0.00145953  -0.00103133  -0.00257677  -0.00399408  0.000361122  0.00295089  -0.00114412  -0.954675  -0.000423194  0.000259006  0.000255573  0.582677  -0.00465551  0.00315604  0.0002401  0.000477747  -0.00255822  0.00295519  0.0027962  0.00347227  -0.00392339  0.000771538  -0.00261917  0.000359551  -0.00119779  0.00164642  -0.00274816  -0.00424702  -1.02355  0.00281609  0.00259559  -0.00547654  0.00225441  0.00191445  0.00237061  0.00247349  0.00193345  5.02093e-05  0.00153661  -4.61675e-05  -0.00164144  -1.1324  -0.00102164  -0.00132161
clusters 3
0.00104871  -0.00388688  0.0011054  0.0019854  -0.00100557  0.000342949  -0.000565209  -1.32753  0.00266834  0.00065333  -0.00145518  -3.18092e-05  0.000404591  0.54714  -0.000696519  -0.00177883  0.000880592  -0.0550648  0.00248429  -0.00171618  -0.00128511  -0.0005937  -0.00828466  -0.000678516  -0.00457646  0.0020884  -0.00140919  1.22616  0.00118976  0.000162946  -0.780997  -0.00318463  -0.00336879  0.000252832  0.00176585  0.00205918  0.00253899  0.0004084  3.00026e-05  0.000985757  0.0014166  0.000581707  -0.00203361  -0.000207115  -0.000133765  0.000542376  -0.00120722  0.00150407  0.00151074  0.0024296  0.00108831  0.00374465  -0.654936  0.00270912  0.00487302  0.00167395  0.00156632  0.0026716  0.00061462  -9.24657e-06  -0.00264763  -0.00290026  -0.000751565  1.14771  -0.00108072  0.00201891  -0.0015465  0.134038  -0.000376355  0.00527833  5.72629e-05  -0.00287477  0.000429819  0.00140474  -0.000317671  -3.114e-05  -0.00029828  0.000248583  -0.00167626  -0.00187673  -0.000550209  -0.00253926  -0.00151382  0.000728666  0.982889  0.00157814  -0.00167135  -0.00200255  0.000731197  0.00031886  0.00179399  0.00105034  -0.00323367  -0.00384662  -0.00327942  0.0014008  -0.000952957  0.249197  -0.000147461  0.00312724
clusters 4
0.00329106  0.000275235  0.00173659  0.00130285  -0.00121463  -0.00467991  0.0010435  3.88175  0.00115023  -0.000291634  -0.00201259  9.16404e-05  0.000913863  2.09039  -0.0019596  0.000593348  0.00171809  0.0464497  0.00203357  -0.00283912  0.00185615  -0.00386599  0.00373049  -0.000386479  0.00178729  -0.00298412  -2.05272e-05  -1.38833  0.00123477  -0.000339517  0.866498  0.00057938  0.00180402  0.00019624  -0.00157562  -0.00192844  0.00168405  -0.000292741  -0.00233711  0.00160762  0.00148108  0.000369695  0.00639402  -0.00116059  0.00318428  0.00210023  -0.00116693  -0.00275107  0.000513095  -0.0033453  -0.000726905  0.00104311  0.815323  -0.00141417  -0.00504613  3.88838e-05  -0.000983991  -0.00132187  0.00436054  0.00129583  0.00237567  0.00222531  0.000985423  0.891325  0.000619051  0.00269165  0.00410931  0.162949  0.000653176  0.00132925  0.000762549  0.00145699  -0.000917212  0.000545157  -0.00135832  2.26041e-05  0.000330361  0.0015791  0.000908565  -0.000357039  -0.00100437  -0.00237288  -0.00011506  0.000256538  1.01316  -0.000209342  -0.00092681  -0.000293742  0.00312878  -0.00397126  0.00345725  0.000473708  4.65667e-06  -0.00324465  -0.00037823  -0.000623574  -0.000564681  1.02339  0.00120382  -0.000410744
sampleClasses_10 1
1  0  2  1  0  3  2  0  2  2
```

## 5 基于 Python scikit-learn 库的 Kmeans 实现
sklearn 库是一个Python机器学习最常用的第三方库，里面封装了大量的机器学习模型可供直接调用，并且API接口统一，上手简单。在 sklearn 库中有 Kmeans 模型的API，下面给出简单实例代码：
```python
import numpy as np
from sklearn.cluster import KMeans

xy = np.loadtxt("sample_1e6_fea_100_class_4_lable_1_ninfo_8.csv", delimiter=",")
clusters = xy[[1, 3, 6, 8], :-1]

%%time
model = KMeans(n_clusters=4, verbose=1, init=clusters)
model.fit(xy[:, :-1])
print(model.labels_[:10])
"""
# output:
Initialization complete
Iteration 0, inertia 208029133.74340856
Iteration 1, inertia 121657238.56283979
Iteration 2, inertia 119114598.7570485
Iteration 3, inertia 118622118.65031119
Iteration 4, inertia 118446750.62386386
Iteration 5, inertia 118355720.17504436
Iteration 6, inertia 118300281.55464649
Iteration 7, inertia 118264039.94947146
Iteration 8, inertia 118240199.91468692
Iteration 9, inertia 118224723.1885456
Iteration 10, inertia 118214754.50177184
Iteration 11, inertia 118208411.6566767
Iteration 12, inertia 118204244.10556875
Iteration 13, inertia 118201492.41313204
Iteration 14, inertia 118199701.13865753
Iteration 15, inertia 118198516.21804495
Iteration 16, inertia 118197696.48761667
Iteration 17, inertia 118197156.68854257
Iteration 18, inertia 118196809.15596466
Iteration 19, inertia 118196610.40712865
Iteration 20, inertia 118196491.81284113
Iteration 21, inertia 118196417.76360503
Converged at iteration 21: center shift 0.0001008382370309347 within tolerance 0.00013575864017093938.
Wall time: 3.64 s
array([1, 0, 2, 1, 0, 3, 2, 0, 2, 2])
"""
```
可以看到使用 sklearn 库进行 KMeans 聚类大概需要 `3.64 s`，速度快于我们简单循环的C++代码，慢于CUDA代码。但是由于第三方库的存在，其实现过程难度大大降低，主体代码就两行，而写C++要500多行。。。

## 6 小结
本文内容总结如下：
- 如果核函数的计算之间互不影响，可以通过流的方式让核函数并行执行，但是流的生成与销毁也是有开销的，比如本文使用流并行执行 `calDistKernel` 函数，但实际上由于 `calDistKernel` 函数执行太快，最后加上流的生成与销毁消耗的时间，最终反而要比循环执行要慢 100 多毫秒。
- 主机端与设备端之间数据传输是经由 PCIe 传递，而常用的连接 GPU 和 CPU 内存的 PCIe x16 Gen3 仅有 16GB/s 带宽，是 GPU 显存带宽的几十分之一。所以要获得可观的 GPU 加速，就必须尽量缩减数据传输所花的时间比例，有时即使某些计算在 GPU 中效率不高，也尽量在 GPU 中计算，避免过多的数据经 PCIe 传递。
- 人生苦短，我用 Python。

## 附录
主函数及计时函数代码：
```cuda
void timing(
    float *data, 
    int *label, 
    float *clusters, 
    const int numClusters, 
    const int n_features, 
    const int n_samples,
    const int method) {
    
    Kmeans *model;
    switch (method)
    {
    case 0:
        model = new Kmeans(numClusters, n_features, clusters, n_samples, 50, 0.1);
        break;
    case 1:
        model = new KmeansGPU(numClusters, n_features, clusters, n_samples, 50, 0.1);
        break;
    default:
        break;
    }

    std::cout << "*********starting fitting*********" << std::endl;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    model->fit(data);

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    
    printf("Time = %g ms.\n", elapsedTime);

    std::cout << "********* final clusters**********" << std::endl;
    printVecInVec<float>(model->m_clusters, 4, 100, "clusters");
    std::cout << "*********    accuracy  **********" << std::endl;
    std::cout << "model accuracy : " << model->accuracy(label) << std::endl;
    printVecInVec<int>(model->m_sampleClasses, 1, 10, "sampleClasses_10");

    delete model;
}


int main() {
    int N = 0;
    int n_features = 100;
    const int bufferSize = 1000000 * n_features;
    float *data = new float[bufferSize];
    int *label = new int[bufferSize];
    readCoordinate(data, label, n_features, N);
    std::cout << "num of samples : " << N << std::endl;

    int cidx[] = {1, 3, 6, 8};
    int numClusters = 4;
    float clusters[400] = {0};
    for (int i=0 ; i<numClusters; ++i) {
        for (int j=0; j<n_features; ++j) {
            clusters[i * n_features + j] = data[cidx[i] * n_features + j];
        }
    }
    std::cout << "********* init clusters **********" << std::endl;
    printVecInVec<float>(clusters, 4, 100, "clusters");

    std::cout << "Using CPU:" << std::endl;
    timing(data, label, clusters, numClusters, n_features, N, 0);
    std::cout << "Using CUDA:" << std::endl;
    timing(data, label, clusters, numClusters, n_features, N, 1);

    delete[] data;
    delete[] label;

    return 0;
}
```