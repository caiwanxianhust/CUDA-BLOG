#! https://zhuanlan.zhihu.com/p/685152552
![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qOuFVvibRGAiaf5MwEHJtsRJLuebFyQdu9d6DqEMZR1ZdRBu6hoGwCye3kpOpjnKogBe09FEFOqvTg/640?wx_fmt=png)
# 【CUDA编程】Faster Transformer v3.0 源码详解

**写在前面**：本文将对 Faster Transformer v3.0 版本源码进行解读，重点介绍该版本基于前面 3 个版本的优化内容，剖析源码作者优化意图，为了便于交流讨论，除公众号：**后来遇见AI** 以外，本文也将在知乎进行发布，欢迎各位读者阅读并给出意见。此外，由于笔者近期忙于工作、软考、写作等诸多事宜，这将是 Faster Transformer 源码详解系列的最后一篇文章，后续版本的源码将不再进行介绍。

## 1 v3.0 版本发布背景
在 FasterTransformer v1.0 中，Nvidia 提供了一个高度优化的 BERT Transformer Encoder 模块，主要应用于序列标注推理场景，笔者针对源码的实现逻辑和优化技巧进行了深度剖析，有兴趣的读者可以移步——[【CUDA编程】Faster Transformer v1.0 源码详解](https://mp.weixin.qq.com/s/8P8n8XAdBcEYqwaKQ49-SQ)。  

在 FasterTransformer v2.0 中，Nvidia 添加了一个高度优化的 Decoder 模块和一套推理方案 Decoding 模型。其中，Decoder 相当于我们常说的 decoder layer；而 Decoding 则包含了整个解码的流程，包括词嵌入、位置编码、解码层和束搜索等过程，相当于我们常说的 decoder model。同样，笔者针对 v2.0 版本新增的内容进行了优化解读，有兴趣的读者可以移步——[【CUDA编程】Faster Transformer v2.0 源码详解](https://mp.weixin.qq.com/s/TkrszG84qRM2ULNqhBJ99Q)。

在 FasterTransformer v2.1 中，官方主要添加了 3 块优化内容。第一点是考虑到 PyTorch 的用户越来越多，官方添加了对 PyTorch 的支持。第二个特点是支持 [Effective Transformer](https://github.com/bytedance/effective_transformer)，该优化思路来自字节跳动算法团队，计算模型中去除了 encoder 输入的无效填充，从而降低了计算开销。第三，除了使用束搜索进行解码外，还提供了基于采样的解码策略。除此之外，Nvidia 还对 Encoder、Decoder 和 beam search 等诸多模块的内核进行了优化，进一步提高了 FasterTransformer 的推理速度。有兴趣的读者可以移步——[【CUDA编程】Faster Transformer v2.1 源码详解](https://mp.weixin.qq.com/s/mofyXsnduNzrU9RjZM4cvw)。

在 FasterTransformer v3.0 中，Nvidia 针对 Encoder 提供了基于 cuBLASLt 的 INT8 量化实现，该实现同样支持 Effective Transformer。通过 INT8 量化的 Encoder 可以高效地利用图灵架构 GPU 中的 INT8 Tensor Core，在保证低精度损失的前提下，取得好的加速比（对比 FP16 运算精度而言），要注意的是 FasterTransformer v3.0 仅支持在计算能力不低于 7.5 的设备上运行。

本文主要聚焦于 v3.0 源码的 3 个方面进行解读：INT8 量化、利用 cuBLASLt 及 INT8 Tensor Core 加速矩阵乘法、相应内核优化，针对其他未发生变更的内容，有兴趣的读者可以阅读笔者的前三篇文章。

## 2 整体架构
同前面三个版本一样，v3.0 的底层由 CUDA、cuBLAS、cuBLASLt 实现，提供 C++ API 和 TensorFlow/PyThorch OP。用户可以将它们集成到 TensorFlow、PyTorch 或其他在本机 C++ 中构建的推理服务代码中。此外官方还提供了一些简单的示例代码来演示如何使用 Encoder、Decoder 以及在 C++、TensorFlow 和 PyTorch 中执行 Decoding 过程。下面是整体架构图：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qOuFVvibRGAiaf5MwEHJtsRJLuebFyQdu9d6DqEMZR1ZdRBu6hoGwCye3kpOpjnKogBe09FEFOqvTg/640?wx_fmt=png)

源码地址如下，有兴趣的读者可以前往下载：
> https://github.com/NVIDIA/FasterTransformer/tree/v3.0/

## 3 INT8 量化
NVIDIA GPU 从图灵架构开始支持 INT8 Tensor Core，可以大幅提高神经网络 INT8 推理速度和吞吐量。 INT8 推理需要先将神经网络模型量化，因此官方发布了基于 TensorFlow 和 PyTorch 量化工具，用于生成 INT8 量化模型以方便部署。该量化工具集成了多种量化校准算法及两种量化方法（PTQ 和 QAT），用于实现模型的量化及导出到 TensorRT 和 FasterTransformer 3.0，该工具不仅可以保证推理精度，而且可以满足 INT8 加速的需求。

### 3.1 什么是 INT8 ？
随着以 Transformer 为基础语言模型规模的快速增大，对 GPU 显存和算力的需求越来越大，如果降低显存占用、提升计算效率就显得愈发重要。这两个目标都与模型的参数类型息息相关，参数类型占用空间越小，显存占用自然也就越少，计算起来也就更快。

模型的参数类型通常为 `fp32`、`fp16`、`tf32` 或 `bfloat16` 等，浮点数存储规则如下图所示：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5om6pWchp6mdnwrfpSUZ1QKnVfJaOzQxw1HgSq0k46Vg5WzgZsibeCFicJd84iavo5SuL3vurRHBrctg/640?wx_fmt=png&amp;from=appmsg)

- FP32：即我们常用的 `float` 类型，标准的 IEEE 32 位浮点表示，指数 8 位，尾数 23 位，符号 1 位，可以表示大范围的浮点数。大部分硬件都支持 FP32 运算指令。
- FP16：即我们常用的 `half` 类型，指数 5 位，尾数 10 位，符号 1 位。FP16 数字的数值范围远低于 FP32，存在上溢（当用于表示非常大的数时）和下溢（当用于表示非常小的数时）的风险，可以通过缩放损失（loss scaling）来缓解这个问题。
- bfloat16（BF16）：指数 8 位（与 FP32 相同），尾数 7 位，符号 1 位。这意味着 BF16 可以保留与 FP32 相同的动态范围。但是相对于 FP16，损失了 3 位精度。因此，在使用 BF16 精度时，大数值绝对没有问题，但是精度会比 FP16 差。
- TF32：使用 19 位表示，结合了 BF16 的范围和 FP16 的精度，是计算数据类型而不是存储数据类型，目前使用范围较小。

通常在模型训练阶段，为了保证精度，权重主要使用 FP32 类型，而在推理阶段，将训练好的权重转化为 FP16 类型后，模型的精度往往与使用 FP32 相差不大。这意味着在推理阶段，如果权重类型使用 FP16，那么只需要使用一半 GPU 显存就能获得相同的结果，同时 FP16 也意味着更高的计算效率，体现在 FasterTransformer 中就是官方往往也会给出配套的 `half` 数据类型的 Kernel 实现。

既然可以将 32 位的数据类型缩减到 16 位，那么按照这个思路是否还能进一步缩减到 8 位呢？这就是我们常说的量化技术，最常见的就是 INT8 量化。

INT8，即 `int8_t` 类型，在头文件 stdint.h 中被定义为 `signed char` 类型，由 1 个符号位和 7 个数值位表示，范围为 `[-127,127]` 区间内的整数。

### 3.2 量化方法
INT8 量化，简单来说就是把浮点数 $x$ 通过缩放因子 $scale$ 和基准值 $z$ 缩放到 `[-127,127]` 区间内的整数 $x_q$ 的过程。

量化过程：
$$
x_q = clip(round(\frac{x}{scale}) + z)
$$

反量化过程：
$$
x = (x_q - z)\cdot scale
$$

根据 $z$ 是否等于 0，将 INT8 量化方法分为两种：对称量化和非对称量化。

当 $z = 0$ 时，浮点数 $\alpha$ 和 $-\alpha$ 两个绝对值相同的值在量化后也将映射到两个绝对值相同的 INT8 整数，称为对称（Symmetric）量化。针对神经网络模型，由于激活值和权重的分布通常在宏观角度上是以 0 对称的，所以直接使用对称量化不会有太大的精度损失。

当 $z \neq 0$ 时，浮点数 $\alpha$ 和 $\beta$ 两个绝对值不同的值在量化后可能会映射到两个绝对值相同的 INT8 整数，称为非对称（Symmetric）量化。非对称量化通过基准值偏移，针对浮点数分布不与 0 对称的场景，可以保留更好的精度。

在 FasterTransformer v3.0 源码中使用的就是对称量化，因为对称量化相比于非对称量化实现起来相对简单很多。以乘法运算为例，非对称量化由于偏移项的存在会多出 3 项，这给反量化也增加了难度。

对称量化的乘法：
$$
R_1 R_2 = s_1 Q_1 s_2 Q_2 = s_1 s_2 Q_1 Q_2
$$

非对称量化的乘法：
$$
R_1 R_2 = s_1 (Q_1 - z_1) s_2 (Q_2 - z_2) = s_1 s_2 Q_1 Q_2 - s_1 s_2 z_2 Q_1 - s_1 s_2 z_1 Q_2 + s_1 s_2 z_1 z_2
$$

### 3.3 量化粒度
量化粒度（Quantization Granularity）主要分为两种：per-tensor 和 per-channel。per-tensor 就是整个 tensor 使用同一个缩放因子进行量化，相当于整个 tensor 一起量化；per-channel 即每个 channel 使用同一个缩放因子进行量化，相当于整个 tensor 会被分为 num_channel 组分别量化，不同的 channel 之间缩放因子不同，这里的 channel 是卷积网络的一个概念，其他网络也适用。

很明显 per-channel 量化要更为复杂一些，那么为什么要选择 per-channel 量化呢？一般来说神经网络权重中不同 channel 的数值分布差异很大，其最大的绝对值差别非常大，如果都使用同一个缩放因子，那么可能很大一部分 channel 的信息在量化后全部损失掉了。

一般来说，如果我们选择 per-tensor 量化，那么激活值和权重都会使用 per-tensor 这种方式量化；而如果我们选择 per-channel 量化，那么通常我们只对权重进行 per-channel 量化，而激活值仍然保持 per-tensor 量化方式。

总的来说，两种量化粒度有如下特点。
- per-tensor：实现简单，性能更高。
- per-channel：精度更好。

### 3.4 如何确定缩放因子
选择了使用对称量化方法进行量化后，我们还剩一个值需要确定，即缩放因子 $scale$，通常有两种方法来确定 $scale$：训练后量化（PTQ）、量化感知训练（QAT）。

训练后量化通常会使用一个校准（Calibration）的过程，利用校准数据在模型上进行前向传播，在前向传播的过程中会统计权重、激活值的信息，最后根据统计信息来确定 $scale$ 值。

量化感知训练会在训练的同时来确定 $scale$ 值，使得神经网络对量化后的激活值、权重适应性更强。

两种确定缩放因子的方式特点总结如下：
| PTQ | QAT |
|:----:|:----:|
|只需要前向传播|需要训练|
|速度快|速度慢|
|精度稍低|精度高|

### 3.5 校准方法
#### 3.5.1 最大值校准
在前向传播过程中搜集 tensor 的浮点数绝对值的最大值。
$$
scale = \frac{MaxAbsVal}{127.0}
$$

最大值校准在权重量化时效果比较好，但是有时在激活值量化时效果比较差。

激活值的范围与输入数据是息息相关的，有时一些输入数据可能会导致激活值产生比较严重的离群值，如果使用最大值校准，可能会导致严重的精度损失。通常要结合下面三种校准器：百分位校准器（Pencentile Calibration）、MSE 校准器、Entropy 校准器。

#### 3.5.2 Pencentile 校准器
统计 tensor 中浮点数的直方图，选择一个百分位阈值对应的浮点数作为最大值进行校准，计算出缩放因子。关于百分位阈值，我们通常使用 99%、99.9%、99.99%、99.999 等。

```
percentile_calibrate(input_tensor, percentile)
{
    collect histogramof input_tensor;
    choose threshold value to keep percentile% of values;
    calculate scalevalue from threshold;
    return scale;
}
```

#### 3.5.3 MSE 校准器
统计 tensor 中浮点数的直方图，根据不同阈值计算量化前后的 MSE（均方误差），找到 MSE 最小的阈值对应的浮点数作为最大值进行校准，计算出缩放因子。总的来说，目的是希望量化前后平均的变化数值最小。

均方误差计算公式：
$$
MSE = \frac{1}{n}\sum_{i=1}^{n} {(Y_i - \hat{Y}_i)^2}
$$

```
mse_calibrate(input tensor) 
{
    collect histogram of input tensor;  
    
    for different threshold value {
        calculate MSE between input tensor before and after quantization;
    }
    choose threshold that minimize MSE; 
    
    calculate scale value from threshold; 
    return scale;
}
```

#### 3.5.4 Entropy 校准器
统计 tensor 中浮点数的直方图，根据不同阈值计算量化前后的 KL 散度，找到 MSE 最小的阈值对应的浮点数作为最大值进行校准，计算出缩放因子。总的来说，目的是希望量化前后的概率分布变化最小。

均方误差计算公式：
$$
P、Q 两个分布的 KL 散度 = \sum {P \log \frac{P}{Q}}
$$

```
entroy_calibrate(input tensor) 
{
    collect histogram of input tensor;  
    
    for different threshold value {
        calculate KL_divergence between data distribution before and after quantization;
    }
    choose threshold that minimize KL_divergence; 
    
    calculate scale value from threshold; 
    return scale;
}
```

下面这张图是 BERT-Encoder 的其中某一层激活值在不同校准器下生成的阈值。一般来说，阈值比较小，表示范围比较小，精度表现比较好。一般情况下，Entropy 校准器比较适合 CV 相关的网络，而百分位校准器、MSE 校准器比较适合 BERT 这类模型。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qWsexf6fYF1XicBS7DaptrWERChPBHJv18hVmyXdGpiaQ4FyqMJ9SZFHbiaPicpN4IlOxtFH2dVvibuyg/640?wx_fmt=png&amp;from=appmsg)

### 3.6 Faster Transformer 的 Encoder 中针对 INT8 的一些 tricks
Faster Transformer v3.0 中基于 INT8 量化的应用主要在于 gemm 的计算使用 INT8 量化，所有的量化与反量化都围绕 gemm 进行。
- 在使用 cuBLASLt API 时对矩阵的尺寸（`m`、`n`、`k`）有要求，为了避免为 transformed 矩阵申请 buffer，会确保 `n % 8 == 0` 以及 `k % 32 == 0`。
- 对权重矩阵提前 transform。对于推理框架来说，权重矩阵实际上是固定的，可以提前准备好所有权重矩阵相关的前置工作。
- 把矩阵乘法中 `A、C` 矩阵的 transform 融合到其他 Kernel 中，而不是使用 cuBLASLt API cublasLtMatrixTransform()。
- 把量化和反量化操作融合到其他 Kernel 中。

关于 INT8 量化和反量化，源码中定义了一个 `amaxList` 变量用来存储各个激活值和权重的 scale 值相关的信息，以便在量化和反量化时随时取用。

```cuda
const float *amaxList;
```

其中前 `80` 个元素为激活值的 scale 值相关的信息，每 `4` 个元素为一组，分别存储了：最大值、最大值除以 `127.0f`、最大值除以 `127.0f` 后再除以 `127.0f`，`127.0f` 除以最大值等四项指标。根据激活值的使用顺序，按如下顺序存储：

| tensor 名 | 起止索引 |
|:----:|:----:|
|`input_amax`|0-3|
|`Qbias_amax`|4-7|
|`Kbias_amax`|8-11|
|`Vbias_amax`|12-15|
|`Softmax_amax`|16-19|
|`bmm2_amax`|20-23|
|`ProjBiasNorm_amax`|24-27|
|`F1Bias_amax`|28-31|
|`F2BiasNorm_amax`|32-35|
|保留空间|36-79|

在索引大于 `80` 的位置可以用来存储权重的 scale 值相关的信息，例如：
- `query_weight_amax_list`
- `key_weight_amax_list`
- `value_weight_amax_list`
- `proj_weight_amax_list`
- `FC1_weight_amax_list`
- `FC2_weight_amax_list`

由于权重是固定的，其所有前置工作都应该提前完成并存储起来用于实时计算时取用，所以这里索引大于 `80` 的位置的元素也将在 `OpenMultiHeadAttention` 类初始化时从外部传入。

## 4 Kernel 融合思路
下图是一个不使用 INT8 量化的 Transformer Encoder 部分的计算流程图。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qWsexf6fYF1XicBS7DaptrW3W8Wa7wwpVia0MhctSLOr9zw1AXSicQrmluIJtyGlFCpcdyeqtjgKgVw/640?wx_fmt=png&amp;from=appmsg)

通常情况下，对于 gemm 运算我们调用 cuBlas API 进行计算，而把两个 gemm 运算之间的其他计算工作都融合到一个 kernel 中。当加入 INT8 量化以后，在 gemm 前后需要进行量化和反量化操作，因此可以得到如下计算流程图。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qWsexf6fYF1XicBS7DaptrWNJZaTnNTDaQBpU7ObLqEia5NkSC3GzhZ0zpoFx28n0xlkkUDLr96rlg/640?wx_fmt=png&amp;from=appmsg)

图中绿色的框代表量化与反量化操作，可以看到大部分 Kernel 中都融合了该操作。而对于残差结构的 Kernel 而言，其输入 tensor 都是 fp32/fp16 类型，从 Kernel 融合的角度，其实可以把反量化操作也融合到残差 Kernel 中，这样的话在增加 Kernel 融合程度的同时还有一个好处。在把反量化融合到残差 Kernel 之前，每层 Encoder layer 的 from_tensor 在计算之前都要先进行 INT8 量化，因为残差结构要用到原始的 from_tensor，而把反量化融合到残差 Kernel 之后，残差结构的 Kernel 直接使用 from_tensor 的量化结果即可，每层 Encoder layer 计算前无需再显式量化一次，只需要最后一层输出的时候做一次反量化即可，每层最开始的量化操作时间就被隐藏了，具体计算流程图如下：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qWsexf6fYF1XicBS7DaptrWlwcJlU8GD8N6DUdtpeefGXTO2PRPVxH3sHicTfmDYu67TGuo1K3Ufwg/640?wx_fmt=png&amp;from=appmsg)

针对反量化操作是否融合到残差 Kernel，Nvidia 官方源码都给出了相应的实现，具体通过 `int8_mode` 参数来控制。

## 5 内核优化
### 5.1 量化 kernel
前面介绍过，当反量化操作没有融合到残差 Kernel 时，每层 Encoder layer 在计算前都要进行一次量化，这部分逻辑放在 `BertEncoderTransformer` 类的 `initialize` 函数中，通过调用 `quantized_kernelLauncher` 函数来启动 `quantized_kernel`，代码如下：

```cuda
template <typename T>
__global__
void quantized_kernel(int8_t *dst, const T* src, const int size, const float* scale_ptr)
{
  int tid = (blockIdx.x*blockDim.x + threadIdx.x) << 2;
  if (tid < size){
    const float scale = __ldg(scale_ptr);
    char4 tmp;
    tmp.x = float_to_int8_rn(static_cast<float>(__ldg(&src[tid]))*scale);
    tmp.y = float_to_int8_rn(static_cast<float>(__ldg(&src[tid+1]))*scale);
    tmp.z = float_to_int8_rn(static_cast<float>(__ldg(&src[tid+2]))*scale);
    tmp.w = float_to_int8_rn(static_cast<float>(__ldg(&src[tid+3]))*scale);
    char4 *dst_ptr4 = (char4 *)dst;
    dst_ptr4[tid >> 2] = tmp;
  }
}

template <typename T>
void quantized_kernelLauncher(int8_t* dst, const T * src, const int size, const float* scale_ptr, cudaStream_t stream)
{
   assert(size % (4 * 64) == 0);
   dim3 grid((size+255)/256);
   dim3 block(64);
   quantized_kernel<T><<<grid, block, 0, stream>>>(dst, src, size, scale_ptr);
}
```

通过 `quantized_kernel` 的执行配置，可以发现，每个 block 量化 `256` 个元素，很明显这是一个 per-tensor 量化粒度。每个 block 中有 `64` 个线程，每个线程量化 `4` 个元素。这里要注意的是有一个 `scale_ptr` 参数，这个参数表示缩放因子，但是与我们前面介绍的缩放因子有所不同，这个缩放因子取用的是 `127.0f/amax`，这样的话直接用量化前的浮点值乘以 `scale_ptr` 中存储的值再取整就得到量化结果。

在 `quantized_kernel` 中，每个线程量化连续的 `4` 个元素，这里调用了一个 `float_to_int8_rn` 函数完成 `float` 到 `int8_t` 类型的转换，该函数是通过 PTX 代码实现的，有兴趣的读者可以去看源码，这里就不解释了。为什么一次要量化 `4` 个元素？因为很明显这个量化操作主要的瓶颈在于内存带宽，一次性量化 `4` 个元素可以平衡一下指令吞吐量和内存吞吐量。此外，源码在这里定义了一个 `char4` 类型变量 `tmp`，我们知道 `char` 类型和 `int8_t` 类型一样都占 `8` 个字节，所以可以用 `char` 对象来存储 `int8_t` 对象，`char4` 类型中存储了 `4` 个 `char` 对象，这样的话把 `int8_t` 指针转为 `char4` 指针后，相当于一次性写入了 `4` 个 `int8_t` 数据，这是一种向量化数据加载的写法，可以一定程度上提升内存带宽。

### 5.2 INT8 矩阵乘法
v3.0 的矩阵乘法全部基于 INT8 Tensor Core 进行，即矩阵 `A` 和 `B` 的类型为 `int8_t`，而矩阵 `C` 的类型为 `int32_t`，官方关于矩阵乘法提供了两个函数实现：`cublasLtMM_withAlgo` 和 `cublasLtMM_withAlgo_int8IO`，区别就是矩阵 `C` 的类型有所不同，前者是 `int32_t`，后者是 `int8_t`。

这两个函数的核心逻辑都是调用 cuBLASLt 的 `cublasLtMatmul` 进行计算，在调用前需要先创建一系列描述矩阵信息的句柄 `cublasLtMatrixLayout_t` 以及描述矩阵乘法运算的句柄 `cublasLtMatmulDesc_t`，其中涉及一些内存布局、数据类型的设置请读者直接参考源代码和 cuBLASLt 官方文档，这里不对 API 的使用进行介绍。有兴趣的读者可以参阅笔者的另一篇文章，其中演示了如何调用 cuBLASLt 的 `cublasLtMatmul` API 计算矩阵乘法：[【CUDA编程】使用 cuBLASLt 中的 cublasLtMatmul API 进行矩阵乘法](https://mp.weixin.qq.com/s/syWVFmIhlui72br2TVzomA) 

代码过长，这里就不贴了，读者可以自行阅读 github 上的源码。

### 5.3 add_QK_bias_transform 核函数

`add_QK_bias_transform` 核函数在 MultiHeadAttention 中 Q、K、V 矩阵乘以相关权重矩阵之后被调用，其中包含了反量化、加偏置项、量化以及转置等操作。此外，如果使用了 Effective Transformer 的情况下，那还需要把填充值恢复了，此时应该调用 `add_QK_bias_transform_rebuild_padding`，仅仅是加了一个恢复填充值的操作，所以这里只对 `add_QK_bias_transform` 进行介绍。

`add_QK_bias_transform` 核函数的 `grid_size` 为 `batch_size * seq_len * 2`，即整个网格分为两部分，前 `batch_size * seq_len` 个 block 处理矩阵 Q，后 `batch_size * seq_len` 个 block 处理矩阵 K。`block_size` 为 `head_num * size_per_head / 4`，这里要注意不能超过 `1024`。

```cuda
//add_QK_bias_transform for batch int8 cublasLtMatmul & per axis quantization for weight
//1.add QK bias
//2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = batch_size * seq_len, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
//only for int32 input & int8 output
//seq_len, size_per_head must be a multiple of 32
//grid.x = batch_size * seq_len * 2;
//block.x = head_num * size_per_head / 4;
//using char4
template <typename T>
__global__
void add_QK_bias_transform(int8_t *q_buf_, int8_t *k_buf_, const int32_t* Q, const T* bias_Q, 
                           const int32_t* K, const T* bias_K, const int m, const int batch_size, 
                           const int seq_len, const int head_num, const int size_per_head, int stride, 
                           const float * q_weight_amax, const float *q_input_deQFactor_div127_ptr, const float * k_weight_amax, 
                           const float *k_input_deQFactor_div127_ptr, const float *q_output_scale_ptr, const float *k_output_scale_ptr)
{
  const int32_t* data_ptr;
  char4* buf_ptr4;
  const T* bias_ptr;
  const float* weight_amax;
  int qk_id = blockIdx.x / m;

  data_ptr = qk_id == 0 ? Q : K;
  buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
  bias_ptr = qk_id == 0 ? bias_Q : bias_K;
  const float input_deQFactor_div127 = qk_id == 0 ? __ldg(q_input_deQFactor_div127_ptr) : __ldg(k_input_deQFactor_div127_ptr);
  weight_amax = qk_id == 0 ? q_weight_amax : k_weight_amax;
  const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

  int threadIdx4 = threadIdx.x << 2;
  int batch_id = (blockIdx.x % m) / seq_len;
  int head_id = threadIdx4 / size_per_head;
  int id_in_head = threadIdx4 % size_per_head;
  int word_id = blockIdx.x % seq_len;

  int data_id = (((threadIdx4 >> 5) << 5)*m + ((blockIdx.x%m) << 5) + (threadIdx4&31));

  float scale;
  float tmp;
  char4 tmp4;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.x = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4)* input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.y = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.z = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.w = float_to_int8_rn(tmp*output_scale);


  //row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major

  int row_id = word_id;
  int col_id = id_in_head;
  //new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
  int new_col = col_id >> 5;
  int new_row = (qk_id != 1) ?
                  //COL32
                  ((row_id << 5) + (col_id&31))
               :
                  //COL4
                  ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                  ////row_id%2 is even row, otherwise odd row
                  ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                  (
                  ((((row_id >> 3) << 3) + ((row_id&1) << 2) + ((col_id&31) >> 3)) << 5) +
                  ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                  ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                  (((((col_id&7) >= 4)?4:0) + ((row_id&7) >> 1)) << 2) +
                  ////col_id%4 is the id of 4 cols
                  (col_id&3)
                  )
                  ;
  buf_ptr4[(((batch_id*head_num + head_id) * stride + (new_col << 5)*seq_len + new_row) >> 2)] = tmp4;
}
```

核函数中首先定义了几个临时变量：
- `data_ptr`：用来存储 Q 或 K 矩阵，这里的 Q 或 K 指的是 MultiHeadAttention 中 from_tensor 乘以权重矩阵后的矩阵，是 `int32_t` 类型。
- `buf_ptr4`：用来存储输出矩阵。由于最终 Q 或 K 要经过量化，所以其元素会变成 `int8_t` 类型，我们知道 `int8_t` 类型其实就是 `char` 型，占 `1` 个字节，这里 `buf_ptr4` 使用 `char4` 类型的目的是一次性加载 `4` 个元素，提升内存带宽。
- `bias_ptr`：存储 Q 或 K 对应的偏置项。
- `weight_amax`：存储 Q 或 K 对应的权重矩阵的最大值。
- `qk_id`：用于区分当前 block 处理的矩阵到底是 Q 还是 K，直接用 `blockIdx.x` 除以 `m` 即可，等于 `0` 是 Q，等于 `1` 是 K。
- `input_deQFactor_div127`：用于存储 Q 或 K 反量化时 from_tensor 分量的 scale 值，要注意的是 Q 和 K 都是 from_tensor 经过线性变换后的矩阵，所以其反量化时 from_tensor 分量的 scale 值相同，而权重分量的 scale 值不同。这里 `input_deQFactor_div127` 取的是 `amax/127.0f/127.0f`。
- `output_scale`：存储 Q 或 K 加偏置之后量化的 scale 值。

我们把 Q 或 K 矩阵中每 hidden_dim 个元素（即 `head_num * size_per_head` 个元素）称为一行，由于 `block_size` 为 `head_num * size_per_head / 4`，也就是说每个线程要处理 `4` 个元素，所以干脆为了索引方便又临时定义了一个 `threadIdx4` 变量，它是 block 内线程编号的 `4` 倍，便于直接对应一行元素中每 `4` 个元素的起始位置。

接下来就是找到当前线程要处理的 Q 或 K 矩阵中的元素索引，当前的 Q 或 K 矩阵其实是 from_tensor 经过矩阵乘法运算后的矩阵，而调用 cuBLASLt 的 `cublasLtMatmul` 函数时对输出矩阵的内存排序方式进行了指定，为 `CUBLASLT_ORDER_COL32`，这种排序方式与传统行主序、列主序有所不同，有兴趣的读者可以阅读笔者的上一篇文章或者调用 API 自行打印试试，这里直接说结论。

假设矩阵 `A` 的形状为 `[96, 64]`，其元素值如下：
```
array([[   0,    1,    2, ...,   61,   62,   63],
       [  64,   65,   66, ...,  125,  126,  127],
       [ 128,  129,  130, ...,  189,  190,  191],
       ...,
       [5952, 5953, 5954, ..., 6013, 6014, 6015],
       [6016, 6017, 6018, ..., 6077, 6078, 6079],
       [6080, 6081, 6082, ..., 6141, 6142, 6143]])
```
按行主序存储时，其实就是从 `0~6143` 共 `96*64` 个元素连续存储。

将其转换为 `CUBLASLT_ORDER_COL32` 格式后，按如下方式存储：
```
    [   0,    1,    2, ...,   29,   30,   31,
        64,   65,   66, ...,   93,   94,   95,
        128,  129,  130, ...,  157,  158,  159,
       ...,
        6080, 6081, 6082, ..., 6109, 6110, 6111,
        32,   33,   34, ...,   61,   62,   63,
       ...,
        6112, 6113, 6114, ..., 6141, 6142, 6143  ]
```
可以发现，截至当前 `CUBLASLT_ORDER_COL32` 格式相对于最原始的行主序格式的区别只是把矩阵的列按 `32` 分片存储了，相当于：
```
A_col32 = np.concatenate([A[:, :32], A[:, 32:]], axis=0)
```
相当于在原矩阵中用下图的方式索引：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qq77XYnvmmxPgkEOQT154DHxFxicna9zxzGvHfb2YDV0rzFNNOm4MMh2lWUyIsfKBDibmn4OLCa2CA/640?wx_fmt=png&amp;from=appmsg)

所以，如果要在 `A_col32` 中索引原矩阵中的元素 `A[i, j]`，应该用如下方式计算索引偏移量：
```
idx = ((j >> 5) << 5) * m + (i << 5) + (j & 0x1f)
```
所以，元素索引 `data_id` 应该采用如下方式计算：
```
int data_id = (((threadIdx4 >> 5) << 5)*m + ((blockIdx.x%m) << 5) + (threadIdx4&31));
```

找到对应元素后，先进行反量化，通过下面对称量化的矩阵乘法公式可以看出，反量化时直接除以缩放因子即可，激活值采用的是 per-tensor 量化粒度，所以整个矩阵的缩放因子是一样的，现在需要做的是确认激活值和权重值的缩放因子。
$$
R_1 R_2 = s_1 Q_1 s_2 Q_2 = s_1 s_2 Q_1 Q_2
$$

以 Q 矩阵为例先看下量化后的矩阵乘法公式，用 `A` 表示 from_tensor：
$$
\frac{127}{amax} A * \frac{127}{weight\_amax} W = Q
$$
现在要对 Q 进行反量化，那么直接让 Q 除以两个缩放因子即可，即。
$$
dq\_Q = \frac{amax}{127} \cdot \frac{weight\_amax}{127} Q 
$$

正好 `input_deQFactor_div127` 取的是 `amax/127.0f/127.0f`，所以再乘一个 `weight_amax` 即可，反量化后的元素值计算方式如下：
```cuda
scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
```

反量化完成后，直接进行加偏置和量化操作。因为是一次处理 `4` 个元素，所以另外三个元素也用一样的方式计算完成后都存到 `tmp4` 变量中。

熟悉 MutiheadAttention 的读者知道，下一步就要对 Q 和 K 矩阵进行转置了，即要把 `[batch_size, seq_len, head_num, size_per_head]` 转置为 `[batch_size, head_num, seq_len, size_per_head]`。除此之外，官方源码中还把调用 cuBLASLt 的 `cublasLtMatmul` 函数前的 transform 操作也融合到了 Kernel 中。

首先我们来看转置操作，对于转置操作，其实相当于是把中间两个维度 `seq_len` 和 `head_num` 互换了顺序，所以转置以后 `row_id` 为 `word_id`，`col_id` 为 `id_in_head`。

然后再看 transform 操作，从 Attention 公式 $Softmax( \frac{QK^T}{\sqrt{d_k}}) V$ 可以看出 Q 和 K 矩阵分别作为矩阵乘法的左矩阵和右矩阵， cuBLASLt 库的 `cublasLtMatmul` 函数要求左矩阵的内存顺序必须为 `CUBLASLT_ORDER_COL32`，右矩阵的内存顺序为 `CUBLASLT_ORDER_COL4_4R2_8C` 且为转置形式。cuBLASLt 库也专门提供了 `cublasLtMatrixTransform` API 用于 transform 操作，而 v3.0 源码中为了提升 Kernel 融合程度，将这个 transform 操作融合到了 `add_QK_bias_transform` 中。

对于 Q 矩阵，最终矩阵的形状为 `[batch_size, head_num, seq_len, size_per_head]`，假设只考虑转置，那么新的索引应该为：
```cuda
(batch_id * head_num + head_id) * seq_len * size_per_head + word_id * size_per_head + id_in_head
```
再来考虑 transform，由于公式 $Softmax( \frac{QK^T}{\sqrt{d_k}}) V$ 中的矩阵乘法主要涉及的是 `[seq_len, size_per_head]` 两个维度，则上述公式的第一部分无需改变（相当于新矩阵的 `batch_size` 变成了 `batch_size * head_num`），第二部分和第三部分应当按 `CUBLASLT_ORDER_COL32` 进行索引，计算方式类似 `data_id`：
```cuda
(batch_id * head_num + head_id) * seq_len * size_per_head + ((id_in_head >> 5) << 5) * seq_len + (word_id << 5) + (id_in_head & 31) 
```
最后考虑到向量化存储数据，即一次性写入 `4` 个元素，索引值需要再除以 `4`，就得到 Q 矩阵的新索引。
```cuda
(((batch_id*head_num + head_id) * stride + (new_col << 5)*seq_len + new_row) >> 2)
```

对于 K 矩阵，最终矩阵的形状为 `[batch_size, head_num, seq_len, size_per_head]`，假设只考虑转置，那么新的索引应该为：
```cuda
(batch_id * head_num + head_id) * seq_len * size_per_head + word_id * size_per_head + id_in_head
```
再来考虑 transform，同样索引第一部分无需改变，但后面两个部分将需要按照内存顺序 `CUBLASLT_ORDER_COL4_4R2_8C` 进行索引，为了能够正确索引，我们首先需要明白内存顺序 `CUBLASLT_ORDER_COL4_4R2_8C` 具体是怎么存储数据的，官方文档描述如下：
> Data is ordered in column-major ordered tiles of composite tiles with total 32 columns and 8 rows. A tile is composed of interleaved inner tiles of 4 columns within 4 even or odd rows in an alternating pattern. The leading dimension is the stride (in elements) to the beginning of the first 32 column x 8 row tile for the next 32-wide group of columns. For example, if the matrix has 33 columns and 1 row, the leading dimension must be at least (32 * 8) * 1 = 256.

当然这个文档看起来不那么直观，大概只能确定一点，主维度为 `32 * roundoff(m, 8)`，为了清晰明白的观察到索引变化，建议读者们自己调用 `cublasLtMatrixTransform` API 将一个普通行主序的矩阵转换为 `CUBLASLT_ORDER_COL4_4R2_8C` 打印出来看看。要注意的是 `cublasLtMatrixTransform` API 目前暂不支持将 `int32_t` 的矩阵转换为 `CUBLASLT_ORDER_COL4_4R2_8C`。这里笔者直接贴出来笔者对于 `CUBLASLT_ORDER_COL4_4R2_8C` 的理解，见下图。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qDPxX44ia0NaSKm3vsHvmjngOTRpjupNpicvppEA9C22fFibb1lR3f6bSn93CyTicbJ4x8VNgN1F1ibsw/640?wx_fmt=png&amp;from=appmsg)

图中每个矩形代表原行主序矩阵中连续存储的 `4` 个元素，从图中不难理解，新矩阵主维度为 `32 * roundoff(m, 8)`，我们把一个主维度的元素组成的矩阵姑且称之为一个**子矩阵**（后面会用到），而在一个主维度的内部按 `8 * 32` 的小矩阵存储，在小矩阵内部的具体存储顺序笔者标注在矩阵内部，把 `8 * 32` 的矩阵成为一个**小矩阵**。不难发现小矩阵内部又先分成两部分：偶数行和奇数行，奇偶行内部又是按 `4 * 4` 的更小矩阵存储的，图中的 `0，1,2,3` 四个编号对应的 就是一个 `4 * 4` 矩阵。

现在不考虑 `batch_size` 维度，假设原矩阵维度为 `[seq_len, size_per_head]`，在原矩阵中位置为 `word_id` 行`id_in_head` 列的元素，应该怎样计算它在新矩阵中的索引？

显然首先我们要先考虑这个元素属于哪个**子矩阵**，并加上子矩阵的偏移量，假设 `id_in_head` 为 `51`，显然前面还有一个子矩阵，偏移量就为 `32 * seq_len`：

```cuda
((id_in_head >> 5) << 5) * seq_len
```

然后我们需要计算元素在**子矩阵**内部的索引，先看它属于哪个**小矩阵**，加上小矩阵的偏移量，假设 `word_id` 为 `17`，**小矩阵**的行数为 `8`，显然前面还有 `2` 个**小矩阵**，偏移量为 `8 * 32` ：

```cuda
((row_id >> 3) << 3) << 5
```

接着看在小矩阵内部元素位于奇数行还是偶数行，偶数行在前，奇数行在后，假设 `word_id` 为 `17`，那显然是在奇数行，偏移量为 `4 * 32`。

```cuda
((row_id&1) << 2) << 5
```

确定奇偶行后，就需要知道元素前面还有多少个 `4 * 4` 矩阵，假设 `id_in_head` 为 `51`，这显然前面还有 `(51&31) >> 2` 个 `4 * 4` 矩阵，那么偏移量就为 `((51&31) >> 2) << 4`。也可以以 `8 * 4` 矩阵为单位，则偏移量为 `((51&31) >> 3) << 5 + (((51&7) >= 4)?4:0) << 2`，即偏移量计算公式为：

```cuda
(((id_in_head&31) >> 3) << 5) + ((((id_in_head&7) >= 4)?4:0) << 2)
```

在 `4 * 4` 矩阵内部索引就比较简单了，首先使用 `(row_id&7) >> 1)` 确定行数，然后使用 `(col_id&3)` 确定列数，偏移量计算如下：

```cuda
((((row_id&7) >> 1)) << 2) + (col_id&3)
```

最后总的偏移量通过如下方式计算：

```cuda
int row_id = word_id;
int col_id = id_in_head;
int new_col = (col_id >> 5);
int new_row = //COL4
                ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                ////row_id%2 is even row, otherwise odd row
                ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                (
                ((((row_id >> 3) << 3) + ((row_id&1) << 2) + ((col_id&31) >> 3)) << 5) +
                ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                (((((col_id&7) >= 4)?4:0) + ((row_id&7) >> 1)) << 2) +
                ////col_id%4 is the id of 4 cols
                (col_id&3)
                )
                ;
int dst_idx = (batch_id * head_num + head_id) * seq_len * size_per_head + (new_col << 5) * seq_len + new_row;
```
最终考虑到 `batch_size` 维度以及向量化存储数据后索引计算如下：
```cuda
(((batch_id*head_num + head_id) * stride + (new_col << 5)*seq_len + new_row) >> 2)
```

### 5.4 add_V_bias_transform 核函数

`add_V_bias_transform` 核函数与前面的 `add_QK_bias_transform` 相似，在 MultiHeadAttention 中 Q、K、V 矩阵乘以相关权重矩阵之后被调用，其中包含了反量化、加偏置项、量化以及转置等操作。此外，如果使用了 Effective Transformer 的情况下，那还需要把填充值恢复了，此时应该调用 `add_V_bias_transform_rebuild_padding`，仅仅是加了一个恢复填充值的操作，所以这里只对 `add_V_bias_transform` 进行介绍。

`add_V_bias_transform` 核函数的 `grid` 设置为三维，`[size_per_head/32, seq_len/32, batch_size * head_num]`。`block` 为两维，设置为 `[8, 32]`。这样设置大有深意，`grid` 的第三维表示转置之后矩阵的前两维 `batch_size` 和 `head_num`；第一维 `size_per_head/32` 结合 `block` 的第一维 `8`，可以锁定 `size_per_head` 个元素中每 `4` 个元素的起始索引，这里由于 Kernel 中使用了向量化加载数据，每次加载 `4` 个元素，所以 `8` 正好够用；第二维 `seq_len/32` 结合 `block` 的第二维 `32` 可以锁定 `seq_len` 个元素中每个元素的索引，对应了每个 `word_id`。

```cuda
//input matrix a matrix of m = batch_size*seq_len , n = head_num*size_per_head, CUBLASLT_ORDER_COL32
//output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len , CUBLASLT_ORDER_COL4_4R2_8C
//only for int32_t Input int8_t Output
//seq_len, size_per_head must be a multiple of 32
//grid = (size_per_head/32, seq_len/32, batch_size*head_num)
//block = (8, 32);
//using char4
//per axis quantization for weight
template <typename T>
__global__
void add_V_bias_transform(int8_t *v_buf_, const int32_t *V, const T *V_bias, const int batch_size, const int seq_len, 
                          const int head_num, const int size_per_head, int stride, const float* weight_amax, 
                          const float *input_deQFactor_div127_ptr, const float *out_scale_ptr)
{
  const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
  const float out_scale = __ldg(out_scale_ptr);
  __shared__ int8_t shm[32][33];
  const int32_t* data_ptr = V;
  char4* buf_ptr4 = (char4*) v_buf_;
  const T* bias_ptr = V_bias;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  int word_id = (blockIdx.y << 5) + threadIdx.y;
  int id_in_size = (blockIdx.x << 5) + threadIdx4;

  //for V layout (batch_size*seq_len, head_num*size_per_head)
  int col = head_id*size_per_head + id_in_size;
  int row = batch_id*seq_len + word_id;
  int inIdx = (((col >> 5) << 5)*batch_size*seq_len + ((row << 5) + (col&31)));
  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;
  
  float tmp;
  float scale;

  //const half2* bias_ptr2 = (const half2*)bias_ptr;
  //half2 tmp2;

  //tmp2 = __ldg(&bias_ptr2[col >> 1]);
  
  scale = __ldg(data_ptr + inIdx) * __ldg(weight_amax + col) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr + col));//(tmp2.x);
  shm[sh_row][sh_col] = float_to_int8_rn(tmp*out_scale);
  
  scale = __ldg(data_ptr + inIdx + 1) * __ldg(weight_amax + col + 1) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr+col+1));//(tmp2.y);
  shm[sh_row][sh_col+1] = float_to_int8_rn(tmp*out_scale);
  
  //tmp2 = __ldg(&bias_ptr2[(col >> 1) + 1]);

  scale = __ldg(data_ptr+inIdx+2) * __ldg(weight_amax+col+2) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr+col+2));//(tmp2.x);
  shm[sh_row][sh_col+2] = float_to_int8_rn(tmp*out_scale);
  
  scale = __ldg(data_ptr+inIdx + 3) * __ldg(weight_amax+col+3) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr+col+3));//(tmp2.y);
  shm[sh_row][sh_col+3] = float_to_int8_rn(tmp*out_scale);

  __syncthreads();

  //for dst of (size_per_head, seq_len)
  word_id = (blockIdx.y << 5) + threadIdx4;
  id_in_size = (blockIdx.x << 5) + threadIdx.y;
  col = (word_id >> 5);
  row = (
        //COL4
        ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
        ////id_in_size%2 is even row, otherwise odd row
        ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
        ((((id_in_size >> 3) << 3) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
        ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
        ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
        (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
        ////word_id%4 is the id of 4 cols
        (word_id&3)
        );
        
  char4 dataTmp;
  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];
  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}
```
核函数中首先定义了几个临时变量：
- `data_ptr`：用来存储 V 矩阵，这里的 V 指的是 MultiHeadAttention 中 from_tensor 乘以权重矩阵后的矩阵，是 `int32_t` 类型。
- `buf_ptr4`：用来存储输出矩阵。由于最终 V 要经过量化，所以其元素会变成 `int8_t` 类型，前面介绍过使用 `char4` 类型的目的是一次性加载 `4` 个 `int8_t` 元素，提升内存带宽。
- `bias_ptr`：存储 V 对应的偏置项。
- `input_deQFactor_div127`：用于存储 V 反量化时 from_tensor 分量的 scale 值。这里 `input_deQFactor_div127` 取的是 `amax/127.0f/127.0f`。
- `out_scale`：存储 V 加偏置之后量化的 scale 值。
- `shm[32][33]`：根据 `block_size` 参数我们可以知道，每个 block 实际是处理了 `32` 个 `word`，针对每个 `word` 只处理了其中 `32` 个元素（一次处理 `4` 个）。这个 `shm` 变量存的就是每个元素反量化、加偏置项、量化后的信息，二级数组定义为 `shm[32][33]` 而不是 `shm[32][32]` 的目的是通过加入偏移量防止**Bank Conflict**，这是影响共享内存性能的主要的因素。

然后根据内建变量计算矩阵各维度的索引 `batch_id`、`head_id`、`word_id`、`id_in_size`。当前线程处理的元素索引 `inIdx` 与 `add_QK_bias_transform` 核函数中的 `data_id` 的计算方式一样，对于形状为 `[batch_size*seq_len, head_num*size_per_head]` 的矩阵 V 来说，其按 `CUBLASLT_ORDER_COL32` 顺序存储后的元素索引计算方式如下：
```cuda
int col = head_id*size_per_head + id_in_size;
int row = batch_id*seq_len + word_id;
int inIdx = (((col >> 5) << 5)*batch_size*seq_len + ((row << 5) + (col&31)));
```
找到对应元素后，先进行反量化，具体原理在 `add_QK_bias_transform` 核函数中介绍过，反量化后的元素值计算方式如下：
```cuda
scale = __ldg(data_ptr + inIdx) * __ldg(weight_amax + col) * input_deQFactor_div127;
```
反量化完成后，直接进行加偏置和量化操作。因为是一次处理 `4` 个元素，所以另外三个元素也用一样的方式计算完成后都存到 `shm` 中。计算完成后通过 `__syncthreads()` 同步 block 内所有线程，确保 block 内处理的 `32` 个 `word` 都以完成计算。

下一步就要对 V 矩阵进行转置了，即要把 `[batch_size, seq_len, head_num, size_per_head]` 转置为 `[batch_size, head_num, seq_len, size_per_head]`。除此之外，官方源码中还把调用 cuBLASLt 的 `cublasLtMatmul` 函数前的 transform 操作也融合到了 Kernel 中，由于 `cublasLtMatmul` 函数仅支持 NT 乘法，所以还需要把 V 矩阵再转置一次，转置为 `[batch_size, head_num, size_per_head, seq_len]`。

这也是本 Kernel 实现中最难理解的部分，既然难理解，我们不妨先不要看代码，假设按照我们正常的思路接下来的转置加 transform 应该怎么写（暂不考虑向量化加载数据），显然可以得到如下代码：
```cuda
//for dst of (size_per_head, seq_len)
int row_id = id_in_size;
int col_id = word_id;
int new_col = (col_id >> 5);
int new_row =  //COL4
                ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                ////row_id%2 is even row, otherwise odd row
                ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                (
                ((((row_id >> 3) << 3) + ((row_id&1) << 2) + ((col_id&31) >> 3)) << 5) +
                ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                (((((col_id&7) >= 4)?4:0) + ((row_id&7) >> 1)) << 2) +
                ////col_id%4 is the id of 4 cols
                (col_id&3)
                )
                ;
int dst_idx = (batch_id * head_num + head_id) * seq_len * size_per_head + (new_col << 5) * size_per_head + new_row;
```

上面的代码思路很清晰，既然要转置，就把 `row_id` 和 `col_id` 互换，分别为 `id_in_size` 和 `word_id`，然后计算 transform 后的索引，但是这样做有个问题，从 `dst_idx` 的公式可以发现，最后一项 `new_row` 是随着 `threadIdx.y` 增长而增长的，而 `threadIdx.x` 的增长体现在 `word_id` 的增长，而 `word_id` 在倒数第二项，这就间接导致 `dst_idx` 将是随 `threadIdx.x` 跳跃性增长的，很显然这就是**全局内存的非合并写入**。**全局内存的非合并写入**是我们在 CUDA 开发中的大忌，要尽量避免。

如何解决这个问题？其中一种方法就是引入共享内存，由于共享内存带宽非常高，可以先把全局内存的数据合并读取并写入共享内存，再从共享内存中以转置的方式读取数据再合并写入全局内存，至于共享内存的读写就无需考虑合并不合并的问题了。总之，矩阵转置 Kernel 使用共享内存实现合并读写是一个比较经典的做法。这也解释了为什么在 `add_QK_bias_transform` 核函数中没有使用共享内存，而 `add_V_bias_transform` 核函数中使用共享内存 `shm` 的原因。

想通了这一点，再来看源码就容易理解了，把握一点即可，数据往共享内存写入的时候正常写，从共享内存读取以及计算目标索引的时候把 `threadIdx.x` 与 `threadIdx.y` 相互调换即可。

总的来说，这个 Kernel 实现理解起来还是比较复杂的，笔者的建议还是那句话，对于非 CUDA 大佬，没必要把所有操作都融合到 add_bias Kernel，不如直接 cuBLAS API 用起来，官方库的性能也不低，维护起来也很方便，官方这个源码，可读性、可维护性让新手望而却步。

### 5.5 softmax 核函数

获取 Q\K\V 矩阵之后就是 $QK^T$ 矩阵乘法计算，这里不再赘述，获取到 $QK^T$ 结果后，需要对其进行 softmax 操作，关于 softmax 由于涉及 INT8 量化，官方这次也提供了新的 Kernel。

在 softmax 之前，源码中根据 `seq_len` 的长度分别走了不同的分支，每个分支下都有一个相应的 softmax Kernel 实现。

```cuda
grid.x = seq_len;
grid.y = batch_size;
grid.z = head_num;

if (seq_len <= 32){
  if (batch_size * head_num > 960)
    grid.x = ceil(float(seq_len)/32.0f);  // 1
  block.x = (seq_len + 31)/32*32;         // 32
  softmax_COL32_LE32<<<grid, block, 0, stream>>>((int8_t*)qk_buf_, qk_int_buf_, attr_mask, batch_size, head_num, 
                                                  seq_len, float(scalar), q_buf_addBias_amax_ptr + 1, k_buf_addBias_amax_ptr + 1, 
                                                  qk_afterSM_amax_ptr, seq_len*head_num, seq_len*seq_len);
}
else if (seq_len <= 64){
  assert(seq_len % 2 == 0);
  block.x = (seq_len/2 + 31)/32*32;
  if (batch_size * head_num > 960)
    grid.x = ceil(float(seq_len)/32.0f);
  softmax_COL32_LE64<<<grid, block, 0, stream>>>((int8_t*)qk_buf_, qk_int_buf_, attr_mask, batch_size, head_num, 
                                                  seq_len, float(scalar), q_buf_addBias_amax_ptr + 1, k_buf_addBias_amax_ptr + 1, 
                                                  qk_afterSM_amax_ptr, seq_len*head_num, seq_len*seq_len);
}
else
{
  assert(seq_len % 4 == 0);
  block.x = (seq_len/4 + 31)/32*32;
  softmax_COL32<<<grid, block, 0, stream>>>((int8_t*)qk_buf_, qk_int_buf_, attr_mask, batch_size, head_num, 
                                            seq_len, float(scalar), q_buf_addBias_amax_ptr + 1, k_buf_addBias_amax_ptr + 1, 
                                            qk_afterSM_amax_ptr, seq_len*head_num, seq_len*seq_len);
}
```
首先明确待计算的 $QK^T$ 矩阵的形状为 `[batch_size, head_num, seq_len, seq_len]`，先看 `seq_len` 在 `[1, 32]` 区间的情况，此时 `seq_len` 比较小，`block_size` 为 `32`，每个 block 处理一行数据，如果 `batch_size * head_num > 960`，为了不浪费计算资源，可以把 `grid.x` 设置为 `1`，此时每个 block 需要处理一个 `seq_len * seq_len` 个元素。

核函数内首先是对一些量化、反量化的系数进行读取和计算，定义了几个临时变量：
- `amax`：存储量化要使用的 softmax 矩阵的最大值。
- `scalar1`：其中 `scalar1a` 是 Attention 计算的缩放系数，`scalar1b`、`scalar1c` 是 Q、K 矩阵对应的量化系数用于反量化的。
- `qual`：标识当前线程标号是否小于 `seq_len`，值为 `false` 时，当前线程计算结果无意义。

当 `batch_size * head_num > 960` 时，此时 `gridDim.x` 为 `1`，`blockIdx.x` 只能取到 `0`，所以不同行的元素需要循环计算，在每个循环内计算当前行的 `seq_len` 个元素的 softmax 结果。当 `batch_size * head_num <= 960` 时，此时 `gridDim.x` 为 `seq_len`，循环次数实际只有一次，每个 block 处理一行数据。综合来看，一次循环只会处理一行元素，所以我们只需要关注循环体内部的逻辑。

矩阵 $QK^T$ 是通过 cuBLASLt 的 `cublasLtMatmul` API 计算得到的，按 `CUBLASLT_ORDER_COL32` 顺序存储，所以先计算其索引 `inIdx`，计算方式上面介绍过：
```cuda
int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len) +
                (threadIdxx & 0xffffffe0) * seq_len +
                (seq_id << 5) + (threadIdxx & 31);
```
通过索引得到元素值，然后乘以 `scalar1` 进行反量化和修正，此时元素值已经转化为正常 `fp32` 类型，可以正常参与后续计算。拿到元素值后还需要考虑 mask 操作，mask 矩阵是一个形状为 `[batch_size, seq_len, seq_len]` 的矩阵，所以对于某一行来说其偏移量为 `batch_id * seq_len * seq_len`，加上线程标号就得到了 mask 矩阵对应当前元素位置的索引 `mask_id`。如果 mask 矩阵中索引出来的值为 `0`，则给 `mask_val` 赋一个很小的值最后加在 `floatTmp` 上使 `floatTmp` 值很小，以致最终这个 softmax 分量趋于 `0`；如果索引出来的值为 `1`，则 mask 不干预后续计算。

每个线程拿到处理后的 `floatTmp` 值后，进行一次块内规约，即可求出当前行的最大值 `max_val`。这里考虑到 `seq_len` 不超过 `32`，所以一个 block 实际只有一个 warp，源码做了一个判断，当 `blockDim.x <= 32` 时，用束内规约代替块内规约。

接下来就是常规的计算 softmax 操作，加了一个防数值溢出的机制，公式如下：
$$
y_i = softmax(x_i) = \frac{e^{x_i - x_{max}}}{\sum _{j=0}^{n-1} e^{x_j - x_{max}}}，其中i,j=0,1,2...,n-1
$$

让 `floatTmp` 减去 `s_max` 并求其指数值，然后对 `floatTmp` 再一次块内规约求出当前行的和 `sum_val`，此时再让 `floatTmp` 除以 `s_sum` 即可得到 softmax 值。但是因为后续还要进行量化操作，所以就把量化操作融合到这里，先把 `s_sum` 赋值为 `127.0f/sum_val/amax`，再让 `floatTmp` 乘以 `s_sum` 就能得到 INT8 量化后的 softmax 值。

注意这里得到 INT8 量化后的 softmax 值之后，考虑到后面 $QK^T$ 矩阵还要作为左矩阵参与矩阵乘法，所以也要按 `CUBLASLT_ORDER_COL32` 顺序存储 softmax 值，因此输出矩阵的索引也要使用 `inIdx`。

再来讨论 `seq_len` 在 `(32, 64]` 区间的情况，核函数为 `softmax_COL32_LE64`，`block.x` 设置为 `32`。与 `softmax_COL32_LE32` 唯一的不同点在于采用了 `char2` 类型一次性加载 `2` 个 `int8_t` 元素。

同样地，当 `seq_len > 64` 时，核函数为 `softmax_COL32`，与 `softmax_COL32_LE32` 唯一的不同点在于采用了 `char4` 类型一次性加载 `4` 个 `int8_t` 元素。

### 5.6 transpose_COL32_kernel 核函数 

softmax 计算完成之后就是一个 softmax 矩阵与 $V^T$ 矩阵的乘法操作，这里源码是通过 cuBLASLt 的 `cublasLtMatmul` API 计算的，不再赘述。矩阵乘法计算完成后，结果矩阵（称为 attention out 矩阵）的形状为 `[batch_size, head_num, seq_len, size_per_head]`，还需要对其进行**多头合并**，即将其转置为 `[batch_size * seq_len, head_num * size_per_head]`，期间还需要进行反量化、量化操作。

```cuda
//src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
//dst is of m = batch_size*seq_len, k(n) = head_num*size_per_head, CUBLASLT_ORDER_COL32
//grid(seq_len, batch_size)
//block(size_per_head/4, head_num)
//assume size_per_head is multiples of 32
__global__
void transpose_COL32_kernel(int8_t* dst, const int32_t* src, const int batch_size, const int seq_len, const int head_num, 
                            const int size_per_head, const float *v_buf_addBias_deQFactor, const float* qk_afterSM_deQFactor, const float* out_scale_ptr, 
                            const int batch_size_x_seq_len, const int seq_len_x_size_per_head)
{
  const float scale = __ldg(v_buf_addBias_deQFactor) * __ldg(qk_afterSM_deQFactor) * __ldg(out_scale_ptr);
  int threadIdx4 = threadIdx.x << 2;
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.y;
  //get the (row, col) output layout of m*k
  //m = batch_size*seq_len
  //k = head_num*size_per_head
  int mk_row = batch_id*seq_len + seq_id;
  int mk_col = head_id*size_per_head + threadIdx4;
  //get the (row, col) layout of COL32; leading dimension = 32*m = 32*batch_size*seq_len
  int COL32_row = (mk_row << 5) + (mk_col&31);
  int COL32_col = mk_col >> 5;
  int outIdx = ((COL32_col << 5)*batch_size_x_seq_len + COL32_row) >> 2;

  //get the (row, col) input layout of m'*k'
  //m' = seq_len
  //k' = size_per_head
  mk_row = seq_id;
  mk_col = threadIdx4;
  //get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
  COL32_row = (mk_row << 5) + (mk_col&31);
  COL32_col = mk_col >> 5;

  int inIdx = (batch_id*head_num + head_id)*seq_len_x_size_per_head + (COL32_col << 5 )*seq_len + COL32_row;
  char4 tmp;
  tmp.x = float_to_int8_rn(__ldg(src+inIdx)*scale);
  tmp.y = float_to_int8_rn(__ldg(src+inIdx+1)*scale);
  tmp.z = float_to_int8_rn(__ldg(src+inIdx+2)*scale);
  tmp.w = float_to_int8_rn(__ldg(src+inIdx+3)*scale);
  char4 *dst_ptr4 = (char4 *)dst;
  dst_ptr4[outIdx] = tmp;
}

void transpose_COL32_kernelLauncher(int8_t* dst, const int* src, const int batch_size, const int seq_len, const int head_num, 
                                    const int size_per_head, const float *v_buf_addBias_deQFactor, const float* qk_afterSM_deQFactor, 
                                    const float* out_scale_ptr, cudaStream_t stream){
  assert(size_per_head%32==0);
  transpose_COL32_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head/4, head_num), 0, stream>>>(dst, src, batch_size, seq_len, head_num, size_per_head, v_buf_addBias_deQFactor, qk_afterSM_deQFactor, out_scale_ptr, batch_size*seq_len, seq_len*size_per_head);
}
```

Kernel 执行配置：`grid` 设置为二维，`[seq_len, batch_size]`。`block` 为两维，设置为 `[size_per_head/4, head_num]`。

核函数内部首先定义了几个临时变量：
- `scale`：反量化与量化系数，由 `3` 部分组成，softmax 矩阵的反量化系数、V 矩阵的反量化系数、转置后矩阵的量化系数。
- `threadIdx4`：`4` 倍的 `threadIdx.x` 很明显这里核函数要使用 `char4` 类型一次写入 `4` 个 `int8` 元素。
- `batch_id`、`seq_id`、`head_id`：矩阵各维度索引。

首先 attention out 矩阵的内存顺序为 `CUBLASLT_ORDER_COL32`，假设要将其转置为 `[batch_size * seq_len, head_num * size_per_head]`，那么转置后输出矩阵的索引也应当按 `CUBLASLT_ORDER_COL32` 顺序来计算，即先确定转置后的矩阵列索引 `mk_col`，根据列索引加上一个主维度为单位的偏移量 `(((mk_col) >> 5) << 5)*batch_size_x_seq_len`，再结合行索引确定在一个主维度内的偏移量 `(mk_row << 5) + (mk_col&31)`，最后考虑到一次写入 `4` 个元素，索引除以 `4` 就得到当前线程处理的元素在输出矩阵中的索引 `outIdx`。

同样的步骤我们也需要确定内存顺序为 `CUBLASLT_ORDER_COL32`，形状为 `[batch_size, head_num, seq_len, size_per_head]` 的输入矩阵的元素索引，要注意的是，这个矩阵的形状为 `[seq_len, size_per_head]`，`batch_size` 实际为 `batch_size * head_num`，这个前面介绍过多次，直接写结果：
```cuda
int inIdx = (batch_id*head_num + head_id)*seq_len_x_size_per_head + ((threadIdx4 >> 5) << 5 )*seq_len + (seq_id << 5) + (threadIdx4&31);
```

根据 `inIdx` 取值后乘以 `scale` 就完成了反量化与量化操作，然后按 `outIdx` 写入 INT8 量化后的结果即可。

至此，MutiHeadAttention 相关的 Kernel 已介绍完毕。

### 5.7 add_bias_input_layernorm_COL32_mixIntI_int8O 核函数

MutiHeadAttention 计算完成后还需要对 attention 的结果进行一次线性变换，其中矩阵乘法部分使用 cuBLASLt 库完成，add bias 部分则与 layernormlization、残差操作融合到一个 Kernel 中。

当 `int8_mode_` 为 `1` 时，不采用量化残差思路，此时调用 `add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher` 函数；当 `int8_mode_` 为 `1` 时，采用量化残差思路，此时调用 `add_bias_input_layernorm_COL32_mixIntI_int8O_kernelLauncher` 函数。在本文第 4 节笔者介绍了二者的不同之处，这两个函数区别只是后者的 input_tensor 是 `int8_t` 类型，在计算前需要先反量化，而前者的 input_tensor 直接是 `fp32` 或 `fp16` 类型，直接可以参与计算。这里只对后者进行解读，且只解读数据类型为 `fp32` 的情况，`fp16` 同理无需赘述。

在 `add_bias_input_layernorm_COL32_mixIntI_int8O_kernelLauncher` 函数中调用了 `add_bias_input_layernorm_COL32_mixIntI_int8O` 核函数，`grid_size` 为 `batch_size * seq_len`（不考虑 Effective Transformer 机制，这个机制在笔者前面的文章有详细介绍，这里就不纠结了），`block_size` 为 `hidden_dim / 4`，这显然 Kernel 内部又采用 `char4` 类型一次加载 `4` 个 `int8_t` 元素了。

```cuda
//input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
//using char4
template <typename T>
__global__
void add_bias_input_layernorm_COL32_mixIntI_int8O(int8_t* output, const int32_t* input1, const int8_t* input2, const T* bias, const T* gamma, 
                                                  const T* beta, int m, int n, const float* weight_amax, const float *input1_deQFactor_div127_ptr, 
						  const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  const float input1_deQFactor_div127 = __ldg(input1_deQFactor_div127_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  const float output_scale = __ldg(output_scale_ptr);
  int col_start = threadIdx.x << 2;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out[4];
  int input1Idx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31));
  int outIdx = input1Idx >> 2;
  char4 *outTmpPtr = (char4*)output;
  char4 *input2TmpPtr = (char4*)input2;
  char4 input2Tmp = __ldg(input2TmpPtr+outIdx);
  
  int col_start_tmp = col_start;
  local_out[0] = static_cast<float>(input2Tmp.x)*input2_deQFactor + static_cast<float>(__ldg(input1+input1Idx))*__ldg(weight_amax+col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[1] = static_cast<float>(input2Tmp.y)*input2_deQFactor + static_cast<float>(__ldg(input1+input1Idx+1))*__ldg(weight_amax+col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[2] = static_cast<float>(input2Tmp.z)*input2_deQFactor + static_cast<float>(__ldg(input1+input1Idx+2))*__ldg(weight_amax+col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[3] = static_cast<float>(input2Tmp.w)*input2_deQFactor + static_cast<float>(__ldg(input1+input1Idx+3))*__ldg(weight_amax+col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start_tmp));


  mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out[0] = local_out[0] - s_mean;
  local_out[1] = local_out[1] - s_mean;
  local_out[2] = local_out[2] - s_mean;
  local_out[3] = local_out[3] - s_mean;
  variance = blockReduceSum<float>(local_out[0] * local_out[0] +
                                   local_out[1] * local_out[1] +
                                   local_out[2] * local_out[2] +
                                   local_out[3] * local_out[3]
                                  );
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

  col_start = col_start+1;
  local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);
  
  col_start = col_start+1;
  local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);
  
  col_start = col_start+1;
  local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);
  
  outTmpPtr[outIdx] = input2Tmp;
}
```
核函数内部首先定义了几个临时变量：
- `input1_deQFactor_div127`：`input1` 矩阵的反量化系数，`input1` 矩阵就是 attention 结果经过矩阵乘法后的结果矩阵，我们姑且称之为 attn_matmul 矩阵。
- `input2_deQFactor`：`input2` 矩阵的反量化系数，`input2` 矩阵就是每一层的 from_tensor。
- `output_scale`：结果矩阵的量化系数。
- `col_start`：`4` 倍的 `threadIdx.x`，对应 `hidden_dim` 维度每 `4` 个元素的起始索引，很明显这里核函数要使用 `char4` 类型一次写入 `4` 个 `int8_t` 元素。

首先明确 `input1` 和 `input2` 矩阵的形状均为 `[batch_size * seq_len, hidden_dim]` 且按 `CUBLASLT_ORDER_COL32` 顺序存储，所以两个矩阵的索引是一一对应的，按照当前的线程网络布局，`blockIdx.x ` 对应 `row_id`，`col_start` 对应 `col_id`，可以很容易计算出 `input1Idx` 和 `outIdx`，其中 `input1Idx` 是用来存取 `int32_t` 类型的 `input1` 矩阵，而 `outIdx` 用来存取 `char4` 类型的元素实际对应的是 `int8_t` 类型的矩阵。

接下来要计算 attn_matmul 矩阵反量化、from_tensor 反量化、 attn_matmul 矩阵加偏置项、add、layerNorm 等操作。不妨把 `local_out` 的计算公式拆开来看：

- from_tensor 反量化：`static_cast<float>(input2Tmp.x)*input2_deQFactor`
- attn_matmul 矩阵反量化：`static_cast<float>(__ldg(input1+input1Idx))*__ldg(weight_amax+col_start_tmp)*input1_deQFactor_div127`，这里由于 attn_matmul 矩阵是 attention out 矩阵乘了一个权重矩阵得到的，所以反量化的时候还需要考虑权重矩阵的量化系数，即 `__ldg(weight_amax+col_start_tmp)`，从取值可以看出，这是 per-channel 量化。
- attn_matmul 矩阵加偏置项：`static_cast<float>(__ldg(bias+col_start_tmp))`

整体加起来就是 add 操作，按索引循环 `4` 次就把一组元素的以上操作计算完毕存到临时数组 `local_out` 中，现在还差一个 layerNorm 操作，这里补充说明一下 layerNorm 的公式：
$$
variance = \frac{\sum (x_i - mean)^2}{n}
\\
y_i = \frac{x_i - mean}{\sqrt{variance}} \cdot \gamma + \beta
$$

其中，`mean` 的计算很容易，块内规约求和然后除以数量 `n` 即可。`variance` 的计算则要先计算元素与 `mean` 的差值的平方，对差值的平方进行规约然后除以数量 `n`。源码中的 `s_variance` 对应的是 $\frac{1}{\sqrt{variance}}$。

按照上述公式计算完毕后把结果存入 `input2Tmp` 再写入 `output` 矩阵，至此核函数结束。

### 5.8 add_bias_act_COL32_int32I_int8O 核函数

第一次 add & layerNorm 之后，就是前馈神经网络的部分，即两层全连接神经网络，单层可以拆分为：GEMM、add bias、activation 三个操作，由于 GEMM 前后的量化机制，所以可以把两个 GEMM 间的反量化、add bias、activation 融合到一个 Kernel，即 `add_bias_act_COL32_int32I_int8O`。

在 `add_bias_act_COL32_int32I_int8O_kernelLauncher` 函数中调用了 `add_bias_act_COL32_int32I_int8O` 核函数，`grid_size` 为 `batch_size * seq_len`（不考虑 Effective Transformer 机制，这个机制在笔者前面的文章有详细介绍，这里就不纠结了），`block_size` 为 `hidden_dim / 4`，注意这里的 `hidden_dim` 实际是 `4 * head_num * size_per_head`，是之前 `hidden_dim` 的 `4` 倍，输入矩阵的维度为 `[batch_size * seq_len, 4 * head_num * size_per_head]`，这种网格设计显然 Kernel 内部又采用 `char4` 类型一次加载 `4` 个 `int8_t` 元素了。

```cuda
//add bias to matrix of m * n, CUBLASLT_ORDER_COL32
//grid, thread = (m), (n/4)
//using char4
//for per-axis-quantization weight
template <typename T>
__global__
void add_bias_act_COL32_int32I_int8O(int8_t *out, const int32_t* input, const T* bias, const int m, const int n, 
                                     const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr)
{

  const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
  const float out_scale = __ldg(out_scale_ptr);
 
  int col_start = threadIdx.x << 2;
  char4 *outTmpPtr = (char4 *)out;
  char4 tmp;
  int inIdx = (col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start&31);
  int outIdx = inIdx >> 2;
  float val;
  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.x = float_to_int8_rn(val*out_scale);
 
  col_start = col_start + 1;
  inIdx = inIdx + 1;
  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.y = float_to_int8_rn(val*out_scale);
  
  col_start = col_start + 1;
  inIdx = inIdx + 1;
  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.z = float_to_int8_rn(val*out_scale);

  col_start = col_start + 1;
  inIdx = inIdx + 1;
  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.w = float_to_int8_rn(val*out_scale);

  outTmpPtr[outIdx] = tmp;
}
```

核函数内部首先定义了几个临时变量：
- `input1_deQFactor_div127`：`input` 矩阵的反量化系数，`input` 矩阵就是 FC1 GEMM 的结果矩阵。
- `output_scale`：结果矩阵的量化系数。
- `col_start`：`4` 倍的 `threadIdx.x`，对应 `hidden_dim` 维度每 `4` 个元素的起始索引，很明显这里核函数要使用 `char4` 类型一次写入 `4` 个 `int8_t` 元素。

首先明确 `input` 矩阵的形状为 `[batch_size * seq_len, hidden_dim]` 且按 `CUBLASLT_ORDER_COL32` 顺序存储，按照当前的线程网络布局，`blockIdx.x ` 对应 `row_id`，`col_start` 对应 `col_id`，可以很容易计算出 `inputIdx` 和 `outIdx`，其中 `inputIdx` 是用来存取 `int32_t` 类型的 `input` 矩阵，而 `outIdx` 用来存取 `char4` 类型的元素实际对应的是 `int8_t` 类型的 `out` 矩阵。

对 `input` 矩阵的反量化需要考虑两个量化系数，一个是 `input1_deQFactor_div127`，另一个跟 FC1 GEMM 的权重矩阵有关，为 `__ldg(weight_amax+col_start)`，表示权重矩阵每个 channel 的最大值。让 `input` 矩阵的元素乘以这两个量化系数就得到反量化的结果，然后再加上偏置项 `__ldg(bias+col_start)`，得到矩阵线性变换后的结果，至此还差一个激活函数运算。这里采用 `gelu` 激活函数，显然这是一个 element-wise 操作。计算完成后将结果 `val` 乘以量化系数 `output_scale` 量化为 INT8 范围即可。

### 5.9 反量化、转置
前馈神经网络第二层 Dense 结束后紧接着又是一个 add & layerNorm 操作，复用 `add_bias_input_layernorm_COL32_mixIntI_int8O_kernelLauncher` 函数即可，然后对当前 transformer 层数进行判断，如果是最后一层，还需要进行反量化、转置等操作。源码中使用 `FT_transformC_kernelLauncher`、`transposeMatrix_kernelLauncher`、`dequantized_kernelLauncher`，三个函数完成这个过程。

其中 `FT_transformC_kernelLauncher` 函数将调用一个 `FT_transformC` 函数将按 `CUBLASLT_ORDER_COL32` 顺序排列的矩阵转化为列主序矩阵，`transposeMatrix_kernelLauncher` 再将列主序转换为行主序矩阵，`dequantized_kernelLauncher` 再将矩阵进行反量化。这个步骤其实和 `BertEncoderTransformer` 中的 `genTransATensorAndInt8TensorForFirstLayer` 函数的步骤正好相反，`genTransATensorAndInt8TensorForFirstLayer` 函数是在第一层 transformer 之前执行的，正好相互呼应，有始有终。

关于反量化 Kernel，其逻辑和量化 Kernel 一样，普通的 element-wise 操作，笔者不再进行介绍。

而关于转置和 transform 操作，笔者也不打算介绍其 Kernel 逻辑，没有太多的技巧，请读者结合内存顺序自行理解。但是笔者这里给出一个可以灵活转换内存顺序的 CUDA 代码实现，该实现调用了 cuBLASLt 库中的 `cublasLtMatrixTransform` API，可以实现 `CUBLASLT_ORDER_COL`、`CUBLASLT_ORDER_ROW`、`CUBLASLT_ORDER_COL32`、`CUBLASLT_ORDER_COL4_4R2_8C` 等 `cublasLtOrder_t` 类型的顺序互转。

```cuda
template <typename T, cudaDataType_t cuda_type>
void matTransformV2(T *dst, T *src, int batch_size, int64_t stride, int m, int n, 
                    int lda, int lda_trans, cublasLtOrder_t &src_order, 
                    cublasLtOrder_t &dst_order, cublasOperation_t &opTrans, cudaStream_t stream)
{
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);

  // cudaDataType_t Atype = CUDA_R_8I;
  cublasLtMatrixLayout_t Adesc = nullptr;

  CHECKSTATUS(cublasLtMatrixLayoutCreate(&Adesc, cuda_type, m, n, lda));
  CHECKSTATUS(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &src_order, sizeof(src_order)));

  cublasLtMatrixLayout_t AtransformDesc = nullptr;

  if (opTrans == cublasOperation_t::CUBLAS_OP_T) {
    CHECKSTATUS(cublasLtMatrixLayoutCreate(&AtransformDesc, cuda_type, n, m, lda_trans));
  }
  else {
    CHECKSTATUS(cublasLtMatrixLayoutCreate(&AtransformDesc, cuda_type, m, n, lda_trans));
  }

  if (batch_size > 1) {
    CHECKSTATUS(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECKSTATUS(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECKSTATUS(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
    CHECKSTATUS(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
  }
  
  CHECKSTATUS(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &dst_order, sizeof(dst_order)));

  cublasLtMatrixTransformDesc_t transformDesc = nullptr;
  cudaDataType_t transformScaleType = CUDA_R_32F;
  float transformAlpha = 1.0f;
  float transformBeta = 0.0f;
  CHECKSTATUS(cublasLtMatrixTransformDescCreate(&transformDesc, transformScaleType));
  CHECKSTATUS(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTrans, sizeof(opTrans)));
 
  CHECKSTATUS(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, src, Adesc, &transformBeta, 
    nullptr, nullptr, dst, AtransformDesc, stream));
  
  if (Adesc) CHECKSTATUS(cublasLtMatrixLayoutDestroy(Adesc));
  if (AtransformDesc) CHECKSTATUS(cublasLtMatrixLayoutDestroy(AtransformDesc));
  if (transformDesc) CHECKSTATUS(cublasLtMatrixTransformDescDestroy(transformDesc));
  if (ltHandle) CHECKSTATUS(cublasLtDestroy(ltHandle));
}
```

下面也给出一个普通行主序转 `CUBLASLT_ORDER_COL4_4R2_8C` 顺序的代码示例。
```cuda
void testcol44r28c() 
{
    cublasLtOrder_t order_col = CUBLASLT_ORDER_COL;
    cublasLtOrder_t order_row = CUBLASLT_ORDER_ROW;
    cublasLtOrder_t order_col32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_col4_4r2_8c = CUBLASLT_ORDER_COL4_4R2_8C;

    constexpr int batch_size = 1;
    constexpr int m = 64;
    constexpr int n = 128;
    constexpr int num_elements = batch_size * m * n;
    int8_t h_a[num_elements];
    for (int i=0; i<num_elements; ++i) h_a[i] = (int8_t)(i % 128);
    printMat(h_a, m, n, "h_a");

    cudaStream_t master_stream;
    CHECK(cudaStreamCreate(&master_stream));

    int8_t *d_a;
    int8_t *a_transform;
    CHECK(cudaMallocAsync((void **)&d_a, sizeof(int8_t) * num_elements * 2, master_stream));
    a_transform = d_a + num_elements;

    CHECK(cudaMemcpyAsync(d_a, h_a, sizeof(int8_t) * num_elements, cudaMemcpyHostToDevice, master_stream));

    int lda_transform = 32 * roundoff(m, 8);
    cublasOperation_t opTrans = CUBLAS_OP_N;
    constexpr cudaDataType_t a_type = CUDA_R_8I;

    matTransformV2<int8_t, a_type>(a_transform, d_a, batch_size, m * n, m, n, 
                                    n, lda_transform, order_row, order_col4_4r2_8c, opTrans, master_stream);
    CHECK(cudaMemcpyAsync(h_a, a_transform, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost, master_stream));
    CHECK(cudaStreamSynchronize(master_stream));
    printMat(h_a, m*n/32, 32, "a_transform");

    CHECK(cudaFreeAsync(d_a, master_stream));
    CHECK(cudaStreamDestroy(master_stream));
}
```
要注意的是，对于行主序的矩阵 `A(m,n)`，其主维度为 `n`，转换为 `CUBLASLT_ORDER_COL4_4R2_8C` 顺序后，其主维度应该为 `32 * roundoff(m, 8)`，转换为 `CUBLASLT_ORDER_COL32` 顺序后其主维度应该为 `32 * m`。

至此，FasterTransformer v3.0 的 Encoder 部分介绍完毕，Decoder 部分在本次版本中没有进行变更，有兴趣的读者可以阅读笔者对 v2.1 版本的源码解读：[【CUDA编程】Faster Transformer v2.1 源码详解](https://mp.weixin.qq.com/s/mofyXsnduNzrU9RjZM4cvw)。。

## 6 小结
本文主要内容分为 3 个方面：INT8 量化、利用 cuBLASLt 及 INT8 Tensor Core 加速矩阵乘法、内核优化，总结如下：
- 详细阐述了 INT8 量化的背景和原理，由于篇幅有限，量化感知训练（QAT）部分没有进行介绍。
- 介绍了 v3.0 版本的 Kernel 融合思路，这也是 v3.0 源码的整体逻辑框架。
- 针对 FasterTransformer v3.0 Encoder 部分的 Kernel 进行逐一解读，详细介绍了其实现原理和计算思路。
- 利用 cuBLASLt 及 INT8 Tensor Core 加速矩阵乘法部分由于篇幅有限，没有细讲，但是在笔者上一篇文章中进行了详细介绍。
- 形象化地描述了 cuBALSLt 的两种主要内存顺序，并给予内存顺序给出了索引的计算方式。
- 关于转置和 transform 操作，给出了一个可以灵活转换内存顺序的 CUDA 代码实现，该实现调用了 cuBLASLt 库中的 `cublasLtMatrixTransform` API，是笔者比较建议的一种方式。当然，可能比直接融合在 Kernel 里效率也许稍低，但是代码的可读性和可维护性非常好。