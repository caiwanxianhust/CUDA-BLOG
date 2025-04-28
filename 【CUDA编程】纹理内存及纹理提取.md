#! https://zhuanlan.zhihu.com/p/688899761
![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/GJUG0H1sS5qAxgP6xzdfAUic6fRibC7gvYTNskHDibj3jmDVibHdnI0esy8RlIkbnmzGPeImaTJpJAdceAPOicUWENw/640?wx_fmt=jpeg&from=appmsg)

# 【CUDA编程】纹理内存和纹理提取

**写在前面**：本文主要介绍了纹理内存和纹理提取的相关概念和使用方法，说实话笔者平常并未实际使用过纹理内存，本文所有的内容全部来自官方文档，再结合笔者的理解进行阐述。如有错漏之处，请读者们务必指出，感谢！

## 1 纹理内存

CUDA 支持 GPU 上用于图形访问纹理和表面内存的纹理硬件子集，从纹理或表面内存读取数据相比于从全局内存读取可以获得不少性能提升。

CUDA 中的纹理内存是一种专门的、高度优化的内存类型，主要用于图像渲染程序中基于 GPU 对数据进行加速载入和渲染等工作。纹理内存可以用于过滤和采样图像，这对于图形渲染和计算机视觉应用非常关键。

纹理内存空间驻留在设备内存中并缓存在纹理缓存中，因此纹理提取仅在缓存未命中时从设备内存读取一次内存，否则只需从纹理缓存读取一次。纹理缓存针对 2D 空间局部性进行了优化，因此同一个 warp 中的线程读取 2D 空间中地址相邻的纹理或表面内存将获得最佳性能。此外，它是为流式读取设计的，具有恒定的延迟。缓存命中时会减少 DRAM 带宽需求，但不会减少读取延迟。

## 2 纹理提取
纹理内存可以通过纹理函数在 Kernel 中读取，调用这些函数读取纹理的过程称为**纹理提取**（texture fetch）。CUDA 提供了两种不同的 API 来访问纹理内存：**纹理引用**和**纹理对象**，其中纹理引用在 CUDA 12.0 及后续版本中已经不再支持，推荐使用纹理对象进行纹理提取，纹理对象是 CUDA 针对纹理引用缺点而提出的升级版，其作用和纹理引用完全一致，但是使用方法更加灵活。

纹理对象中可以指定纹理提取过程中的基本信息、返回值类型、寻址方式和过滤模式等信息，下面来看一下纹理对象 API。

### 2.1 纹理对象 API
用户通过 `cudaCreateTextureObject()` API 创建纹理对象，创建时还需要指定 `cudaResourceDesc` 资源描述和 `cudaTextureDesc` 纹理描述信息，其中纹理描述属性如下：

```cuda
struct cudaTextureDesc
{
    enum cudaTextureAddressMode addressMode[3];
    enum cudaTextureFilterMode  filterMode;
    enum cudaTextureReadMode    readMode;
    int                         sRGB;
    int                         normalizedCoords;
    unsigned int                maxAnisotropy;
    enum cudaTextureFilterMode  mipmapFilterMode;
    float                       mipmapLevelBias;
    float                       minMipmapLevelClamp;
    float                       maxMipmapLevelClamp;
};
```
`cudaTextureDesc` 的主要属性解释如下：
- `addressMode` 指定寻址方式。
- `filterMode` 指定过滤模式。
- `readMode` 指定读模式。
- `normalizedCoords` 指定纹理坐标是否需要标准化。
- `sRGB`、`maxAnisotropy`、`mipmapFilterMode`、`mipmapLevelBias`、`minMipmapLevelClamp` 和 `maxMipmapLevelClamp` 请参阅的参考手册。

可以看到，`cudaTextureDesc` 纹理描述中有不少纹理提取相关的专业名词，下面将对这些名词进行介绍。

### 2.2 纹理（texture）
纹理（texture），即要提取的纹理内存。纹理对象在运行时创建，并在创建纹理对象时指定纹理。

### 2.3 纹理维度
维度确定了纹理是使用一个纹理坐标的一维数组、使用两个纹理坐标的二维数组还是使用三个纹理坐标的三维数组。数组的元素称为 texels，是纹理元素的缩写。纹理的宽度、高度和深度是指数组在每个维度上的大小。最大纹理宽度、高度和深度与设备计算能力相关，具体可以查阅官方手册。

### 2.4 纹理元素类型
纹理元素的类型（type）仅限于基本整型和单精度浮点类型以及从基本类型派生的内置向量类型，如 `float2`、`float4`、`int2` 等。

### 2.5 读模式（read mode）
读模式（read mode）表示纹理提取结果的表示方式，枚举值为 `cudaReadModeNormalizedFloat`（浮点类型）、`cudaReadModeElementType`（元素类型）。

如果是 `cudaReadModeNormalizedFloat`，那么即使 texel 的类型是 $16$ 位或 $8$ 位整数类型，则纹理提取返回的值实际上也是作为浮点类型返回的，并且整数类型的全范围映射到浮点区间，无符号整数类型映射到区间 $[0.0 , 1.0]$ 表示，有符号整数类型映射到区间 $[-1.0, 1.0]$ 表示；例如，值为 `0xff` 的无符号 $8$ 位纹理元素读取为 $1$。

如果是 `cudaReadModeElementType`，则不执行转换。

### 2.6 坐标标准化（normalizedCoords）
默认情况下，使用 $[0, N-1]$ 范围内的浮点坐标（通过 Texture Functions 的函数）引用纹理，其中 $N$ 是与坐标对应的维度中纹理的大小。例如，大小为 $64 \times 32$ 的纹理将分别使用 $x$ 和 $y$ 维度的 $[0, 63]$ 和 $[0, 31]$ 范围内的坐标进行引用。标准化纹理坐标导致坐标被指定在 $[0.0,1.0-1/N]$ 范围内，而不是 $[0,N-1]$，所以相同的 $64 \times 32$ 纹理将在 $x$ 和 $y$ 维度的 $[0.0,1.0-1/N]$ 范围内被标准化坐标定位。如果纹理坐标独立于纹理大小，则归一化纹理坐标自然适合某些应用程序的要求。

### 2.7 寻址方式（addressing mode）
纹理提取时使用超出范围的坐标调用设备函数是有效的，寻址模式定义了在这种情况下会发生什么。默认寻址模式是将坐标限制在有效范围内：$[0, N)$ 用于非归一化坐标，$[0.0, 1.0)$ 用于归一化坐标。如果指定了边框模式，则纹理坐标超出范围的纹理提取将返回 $0$。对于归一化坐标，还可以使用环绕模式和镜像模式。使用环绕模式时，每个坐标 $x$ 都转换为 $frac(x)=x - floor(x)$，其中 $floor(x)$ 是不大于 $x$ 的最大整数。使用镜像模式时，如果 $floor(x)$ 为偶数，则每个坐标 $x$ 转换为 $frac(x)$，如果 $floor(x)$ 为奇数，则转换为 $1-frac(x)$。寻址模式被指定为一个大小为 $3$ 的数组，其第一个、第二个和第三个元素分别指定第一个、第二个和第三个纹理坐标的寻址模式；寻址模式枚举值为`cudaAddressModeBorder`、`cudaAddressModeClamp`、`cudaAddressModeWrap`和`cudaAddressModeMirror`； 其中 `cudaAddressModeWrap` 和 `cudaAddressModeMirror` 仅支持标准化纹理坐标。

### 2.8 过滤模式（filtering mode）
过滤模式（filtering mode）指定如何根据输入纹理坐标计算纹理提取时返回的值，是纹理提取的核心逻辑。过滤模式的枚举值为 `cudaFilterModePoint` （最近点采样）或 `cudaFilterModeLinear` （线性过滤）。其中线性纹理过滤模式只适用于返回值为浮点类型的纹理提取，它在相邻纹素之间执行低精度插值。启用线性纹理过滤模式后，将读取纹理提取位置周围的 texels，并根据纹理坐标落在 texels 之间的位置对纹理提取的返回值进行插值。对一维纹理进行简单线性插值，对二维纹理进行双线性插值，对三维纹理进行三线性插值。最近点采样模式，则返回值是最接近输入纹理坐标的 texel。

我们把绑定到纹理引用的纹理表示为一个数组 `T`，则：

- 一维纹理：$N$ 个纹理元素（texel）；
- 二维纹理：$N \times M$ 个纹理元素；
- 三维纹理：$N \times M \times L$ 个纹理元素。

#### 2.8.1 最近点采样
在这种过滤模式下，纹理提取获取的值是：
- 一维纹理：$tex(x) = T[i]$；
- 二维纹理：$tex(x, y) = T[i, j]$；
- 三维纹理：$tex(x, y, z) = T[i, j, k]$。

其中，$i=floor(x),\ j=floor(y),\ k=floor(z)$。

下图展示了 $N=4$ 的一维纹理使用最近点采样的提取结果。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qaVc6w896HIFoHlvlld3HS1cmHfCf5Q4ZhVfGcn6eIjECsdnJBBEM6icDkJdqFo8Kk2uXsgIylURg/640?wx_fmt=png&amp;from=appmsg)

从图中可以看出，原理就是把距离输入的纹理坐标最近点的纹理元素返回，当然返回值还会受读模式和纹理元素类型控制，对于整数纹理，可以选择返回映射到区间 `[0, 1]` 或 `[-1, 1]` 间的浮点数。

#### 2.8.2 线性过滤

线性纹理过滤仅适用于返回类型为浮点类型的提取场景，该过滤模式下纹理提取获取的值为：
- 一维纹理：$tex(x)=(1-\alpha)T[i]+{\alpha}T[i+1]$；
- 二维纹理：$tex(x,y)=(1-\alpha)(1-\beta)T[i,j]+\alpha(1-\beta)T[i+1,j]+(1-\alpha){\beta}T[i,j+1]+\alpha{\beta}T[i+1,j+1]$；
- 三维纹理：

$$
tex(x,y,z)= (1-\alpha)(1-\beta)(1-\gamma)T[i,j,k] + \alpha(1-\beta)(1-\gamma)T[i+1,j,k]  \\
+ (1-\alpha)\beta(1-\gamma)T[i,j+1,k] + \alpha\beta(1-\gamma)T[i+1,j+1,k]  \\
+ (1-\alpha)(1-\beta){\gamma}T[i,j,k+1] + \alpha(1-\beta){\gamma}T[i+1,j,k+1]  \\
+ (1-\alpha)\beta{\gamma}T[i,j+1,k+1] + \alpha\beta{\gamma}T[i+1,j+1,k+1] 
$$

其中：
- $i=floor(x_B),\ \alpha=frac(x_B),\ x_B = x - 0.5$；
- $j=floor(y_B),\ \beta=frac(y_B),\ y_B = y - 0.5$；
- $k=floor(z_B),\ \gamma=frac(z_B),\ z_B = z - 0.5$；

$\alpha$、$\beta$ 和 $\gamma$ 以 9-bit 定点数制存储，带有 8 位小数值。

从公式可以看出，启用线性纹理过滤模式后，将读取纹理提取位置周围的 texels，并根据纹理坐标落在 texels 之间的位置对纹理提取的返回值进行插值。对一维纹理进行简单线性插值，对二维纹理进行双线性插值，对三维纹理进行三线性插值。

下图展示了 $N=4$ 的一维纹理使用线性过滤的提取结果。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qaVc6w896HIFoHlvlld3HSmgOYgyOfr2YPKyYoQbF9EqwlDgPviaWicYul5M4j4go6bdzAfJROPskQ/640?wx_fmt=png&amp;from=appmsg)

从图中可以看出，在线性过滤模式下，即使输入的坐标与纹理元素坐标相等，纹理提取的值也不会是纹理元素本身，而是向右偏移了 $0.5$。如果需要**当输入的坐标与纹理元素坐标相等时，纹理提取的值等于纹理元素**，那么各维度输入坐标需要加上 $0.5$ 再进行纹理提取。

### 3 纹理提取示例
常规的纹理内存应用流程主要分为如下几个步骤：
- 分配主机内存并初始化数据，这一步的目的主要是获取数据源，不是必须的；
- 分配 CUDA 数组，拷贝主机数据到 CUDA 数组；
- 设置资源类型；
- 设置纹理对象属性；
- 创建纹理对象；
- 启动 Kernel，在 Kernel 中进行纹理提取。

#### 3.1 主机端的内存分配和源数据初始化
下面给出代码示例，首先是主机端的内存分配和源数据初始化。假设源数据是一个 `32*32` 的二维矩阵，我们简单将矩阵元素按索引进行初始化。

```cuda
constexpr int height = 32;
constexpr int width = 32;

// 分配主机内存，并初始化
float *h_data = new float[height * width];
for (int i = 0; i < height * width; ++i) {
    h_data[i] = i;
}
printMat(h_data, width, height, "h_data");

/*
h_data
0       1       2       3       4       5       6       7       8       9       10      11      12      13      14      15      16      17      18      19      20      21      22      23      24      25      26      27      28      29      30      31
32      33      34      35      36      37      38      39      40      41      42      43      44      45      46      47      48      49      50      51      52      53      54      55      56      57      58      59      60      61      62      63
64      65      66      67      68      69      70      71      72      73      74      75      76      77      78      79      80      81      82      83      84      85      86      87      88      89      90      91      92      93      94      95
...
*/
```

#### 3.2 CUDA 数组
分配 CUDA 数组并将数据从主机端拷贝到 CUDA 数组中。CUDA 数组是针对纹理提取优化的特殊内存布局，由一系列一维、二维或三维的元素组成，每个元素有 $1$、$2$ 或 $4$ 个分量，支持有符号或无符号 $8$ 位、$16$ 位或 32 位整数、$16$ 位浮点数、或 32 位浮点数等多种类型。CUDA 数组只能由 Kernel 通过纹理提取或表面内存的读取和写入来访问，因此也属于设备端的内存，需要通过 `cudaMallocArray` API 进行创建并使用 `cudaMemcpy2DToArray` API 传输数据。

```cuda
// 分配 cuda 数组
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat);
cudaArray_t cuArray;
CHECK_CUDA_ERROR(cudaMallocArray(&cuArray, &channelDesc, width, height));

// 拷贝主机数据到 cuda 数组
constexpr size_t spitch = width * sizeof(float);
CHECK_CUDA_ERROR(cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(float), height, cudaMemcpyHostToDevice));
```

#### 3.3 设置资源类型
前面介绍过，纹理对象创建时需要传入资源描述句柄和纹理描述句柄，这里资源描述句柄主要作用是指定数据源类型（CUDA 数组）并将 CUDA 数组赋值给资源描述句柄的相关属性。 
```cuda
// 设置资源类型
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc)); 
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cuArray;
```

#### 3.4 设置纹理描述
纹理描述句柄中需要指明纹理提取的基本属性，包括寻址模式、过滤模式、坐标是否标准化等，这里我们暂且指定为线性过滤并且坐标不进行标准化。
```cuda
// 设置纹理属性
cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeWrap;
texDesc.addressMode[1] = cudaAddressModeWrap;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;
```

#### 3.5 纹理提取
CUDA 通过 `cudaCreateTextureObject` API 创建纹理对象，创建成功后纹理对象即可作为 Kernel 参数传入 Kernel 在设备中进行纹理提取，具体如下：
```cuda
// 创建纹理对象
cudaTextureObject_t texObj = 0;
CHECK_CUDA_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
```
在线性过滤模式中，前面介绍过，即使输入的坐标与纹理元素坐标相等，纹理提取的值也不会是纹理元素本身，而是向右偏移了 $0.5$。如果需要**当输入的坐标与纹理元素坐标相等时，纹理提取的值等于纹理元素**，那么各维度输入坐标需要加上 $0.5$ 再进行纹理提取。那么为了验证这个情况，我们不妨将 Kernel 中的坐标加上 $0.5$。

```cuda
__global__ void accessTextureKernel(float *out, cudaTextureObject_t texObj, int width, int height)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    float u = idx + 0.5f;
    float v = idy + 0.5f;
    out[idy * width + idx] = tex2D<float>(texObj, u, v);
}

// 启动 kernel
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
accessTextureKernel<<<block, grid>>>(d_buf, texObj, width, height);
CHECK_CUDA_ERROR(cudaGetLastError());

// 拷贝数据回主机
CHECK_CUDA_ERROR(cudaMemcpy(h_data, d_buf, height * width * sizeof(float), cudaMemcpyDeviceToHost));
CHECK_CUDA_ERROR(cudaDeviceSynchronize());

printMat(h_data, width, height, "out");

/*
out
0       1       2       3       4       5       6       7       8       9       10      11      12      13      14      15      16      17      18      19      20      21      22      23      24      25      26      27      28      29      30      31
32      33      34      35      36      37      38      39      40      41      42      43      44      45      46      47      48      49      50      51      52      53      54      55      56      57      58      59      60      61      62      63
64      65      66      67      68      69      70      71      72      73      74      75      76      77      78      79      80      81      82      83      84      85      86      87      88      89      90      91      92      93      94      95
*/
```
从打印的结果可知，输入坐标加上 $0.5$ 后纹理提取的结果就是纹理元素的值。

## 4 小结
本文主要对纹理内存的基本概念、纹理提取的计算逻辑进行了介绍，并给出了一般性的纹理内存应用示例。


