#! https://zhuanlan.zhihu.com/p/669957986
# 【CUDA编程】束内洗牌函数（Warp Shuffle Functions）

**写在前面**：本文主要介绍了 cuda 编程中束内洗牌函数的计算逻辑和使用方法，大部分内容来自于 CUDA C Programming Guide 的原文，笔者尽量以简单朴实话对原文进行阐述，由于笔者水平有限，文章若有错漏之处，欢迎读者批评指正。

## 1 束内洗牌函数的应用背景
束内洗牌函数（Warp Shuffle Functions）是设备端的内置函数，与束内表决函数一样，主要用于进行一些 warp 内数据交换操作。但有两个主要区别：没有数据的规约处理功能、交换的数据通常大于 1-bit。

同样地，束内洗牌函数在写程序的时候也不是必须的，开发人员完全可以不借助它写出功能完备的代码。通常，不使用 warp shuffle 的时候，需要通过共享内存进行 warp 内线程间的数据交换，但是 warp shuffle 的效率要比共享内存高出不少，所以束内洗牌函数的应用场景非常广泛，比如常用的束内规约算法，大都借助了束内洗牌函数进行计算。

## 2 函数描述
内部函数 `__shfl_sync()` 允许 warp 中的线程之间交换变量，而无需使用共享内存。交换同时发生在 warp 中的所有活动线程（使用 `mask` 指定），根据数据类型移动每个线程 $4$ 或 $8$ 个字节的数据。

warp 中的线程称为通道（lanes），并且每个通道具有介于 `0` 和 `warpSize-1`（包括）之间的索引，称之为通道 ID。当前支持四种源通道（source-lane）寻址模式：
- `__shfl_sync()`：从索引通道直接复制。
- `__shfl_up_sync()`：从相对于调用者 ID 较低的通道复制。
- `__shfl_down_sync()`：从相对于调用者 ID 较高的通道复制。
- `__shfl_xor_sync()`：基于自身通道 ID 的按位异或（XOR）从通道复制。

线程只能从另一个参与执行 `__shfl_sync()` 命令的活动线程读取数据。如果目标线程处于非活动状态，则检索到的值未定义。

所有 `__shfl_sync()` 函数都采用一个可选的 `width` 参数，该参数会改变函数的执行逻辑。`width` 的值必须取 $2$ 的自然数次幂（如 $1$、$2$、$4$、$8$、$16$、$32$）；如果 `width` 不是 $2$ 的自然数次幂，或者是大于 `warpSize` 的数字，则结果未定义。如果 `width` 小于 `warpSize`，则 warp 的每个子部分都作为一个单独的实体，姑且称之为 **warp 内的子线程组**，子线程组内起始逻辑通道 ID 为 $0$。简单来说，`width` 参数的作用是将 warp 划分为多个独立计算的子线程组，增加束内洗牌函数的灵活性。

四个束内洗牌函数的计算思路总结如下：
1. 通过 `mask` 参数确定参与计算的线程。
2. 根据 `width` 参数将 warp 划分为 $1$ 个或多个子线程组，在组内对线程重新从 $0$ 开始编号，记为 `srcLaneId`，即每个组内的计算是独立的。
3. 结合函数名和第三个参数 `srcLane`、`delta` 或 `laneMask` 确定当前线程具体是从组内哪个线程复制数据，对应的目标线程的标号记为 `tgtLaneId`。
4. 被 `mask` 指定的线程返回组内对应的 `tgtLaneId` 标号的线程中的变量 `var` 的值，其余线程返回 $0$。

`__shfl_sync()`：`tgtLaneId` 等于 `srcLane`。如果 `srcLane` 在 `[0:width-1]` 范围之外，则 `tgtLaneId` 等于 `srcLane%width`。简单来说，在组内把标号 `srcLane` 线程的变量 `var` 的数据广播给了其他线程。

如下图中，当调用 `shfl_sync(0xffffffff,x,2)` 时，默认 `width` 等于 `warpSize`（即 $32$），则标号为 $2$ 的线程向标号为 $0 \sim 31$ 的线程广播了其变量 `x` 的值；当调用 `shfl_sync(mask,x,2,16)` 时，warp 被划分为两个宽度为 $16$ 的子线程组，则标号为 $2$ 的线程向标号为 $0 \sim 15$ 的线程广播了其变量 `x` 的值，warp 内通道 ID 为 $18$ 的线程（在 warp 的子线程组内标号也为 $2$）向通道 ID 为 $16 \sim 31$ 的线程广播了其变量 `x` 的值。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5psEJ51sC2MnS4kYyPImLibU2pKYKGaWEnic6tTQAVUhiaU0vphyHExLwIWoGyG8xcjZlApllicyq5Aaw/640?wx_fmt=png&amp;from=appmsg)

`__shfl_up_sync()`：`tgtLaneId` 等于 `(srcLaneId - delta)`。本质上把子线程组内通道 ID 较低的线程中的变量广播到了 ID 较高的线程。由于每个子线程组要单独计算，所以每个子线程组内较低通道 ID 的线程将无法获得其他线程的数据（组内的线程标号不会小于 $0$）。

如下图中，当调用 `__shfl_up_sync(0xffffffff, x, 2)` 时，默认 `width` 等于 `warpSize`（即 $32$），则标号为 $2 \sim 31$ 的线程分别获得标号为 $0 \sim 29$ 的线程中变量 `x` 的值；当调用 `__shfl_up_sync(mask, x, 2, 16)` 时，warp 被划分为两个宽度为 $16$ 的子线程组，则标号为 $2 \sim 15$ 的线程分别获得标号为 $0 \sim 13$ 的线程中变量 `x` 的值，warp 内通道 ID 为 $18 \sim 31$ 的线程（在 warp 的子线程组内标号为 $2 \sim 15$）分别获得通道 ID 为 $16 \sim 29$ 的线程中变量 `x` 的值。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5psEJ51sC2MnS4kYyPImLibUZuECHxl1lJy7KJQUAFR4qdqrs0JiaZMNOZuEsGmWfXVCdpGrhpT6OWg/640?wx_fmt=png&amp;from=appmsg)

`__shfl_down_sync()`：`tgtLaneId` 等于 `(srcLaneId + delta)`。本质上把通道 ID 较高的线程中的变量广播到了 ID 较低的线程。由于每个子线程组要单独计算，所以每个子线程组内较高通道 ID 的线程将无法获得其他线程的数据（组内的线程标号不会大于 `width`）。

如下图中，当调用 `__shfl_down_sync(0xffffffff, x, 2)` 时，默认 `width` 等于 `warpSize`（即 $32$），则标号为 $0 \sim 29$ 的线程分别获得标号为 $2 \sim 31$ 的线程中变量 `x` 的值；当调用 `__shfl_down_sync(mask, x, 2, 16)` 时，warp 被划分为两个宽度为 $16$ 的子线程组，则标号为 $0 \sim 13$ 的线程分别获得标号为 $2 \sim 15$ 的线程中变量 `x` 的值，warp 内通道 ID 为 $16 \sim 29$ 的线程（在 warp 的子线程组内标号为 $0 \sim 13$）分别获得通道 ID 为 $18 \sim 31$ 的线程中变量 `x` 的值。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5psEJ51sC2MnS4kYyPImLibU76ibebxcZOTQKibIs6QgcvtaFmDQrjfSEI4RSQnib22nTUiaqRmfC3HEgw/640?wx_fmt=png&amp;from=appmsg)

`__shfl_xor_sync()`：`tgtLaneId` 等于 `(srcLaneId ^ laneMask)`。通过对调用者的通道 ID 进行按位异或来计算目标线程的通道 ID。

如下图中，当调用 `__shfl_xor_sync(0xffffffff, x, 1)` 时，默认 `width` 等于 `warpSize`（即 $32$），对 $0 \sim 31$ 和 $1$ 进行按位异或计算得到 $1$、$0$、$3$、$2$、$5$、$4$、$\ldots$、$29$、$28$、$31$、$30$，各自通道根据计算结果获得目标通道的变量 `x` 的值；同样地，当调用 `__shfl_xor_sync(0xffffffff, x, 3)` 时，默认 `width` 等于 `warpSize`（即 $32$），对 $0 \sim 31$ 和 $3$ 进行按位异或计算得到 $3$、$2$、$1$、$0$、$7$、$6$、$\ldots$、$31$、$30$、$29$、$28$，各自通道根据计算结果获得目标通道的变量 `x` 的值。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5psEJ51sC2MnS4kYyPImLibUDSqicibsOP0ClXaiaaNiaxhhNLtuvYiamibOqKvqeydj9d1cKIvg4PJ92icvg/640?wx_fmt=png&amp;from=appmsg)

## 3 示例
### 3.1 在 warp 内广播单个变量的值

```cuda
#include <stdio.h>

__global__ void bcast(int arg) {
    int laneId = threadIdx.x & 0x1f;
    int value;
    // Note unused variable for all threads except lane 0
    if (laneId == 0)        
        value = arg;       
    // Synchronize all threads in warp, and get "value" from lane 0
    value = __shfl_sync(0xffffffff, value, 0);   
    if (value != arg)
        printf("Thread %d failed.\n", threadIdx.x);
}

int main() {
    bcast<<< 1, 32 >>>(1234);
    cudaDeviceSynchronize();

    return 0;
}
```
以上代码首先在寄存器内定义一个变量 `value`，随后在 warp 内通道 ID 为 $0$ 的线程中将 `value` 赋值为 `arg`，此时其他 $31$ 个线程的 `value` 还未进行初始化。然后每个线程中调用 `value = __shfl_sync(0xffffffff, value, 0)`，相当于把 ID 为 $0$ 的线程中的 `value` 的值广播到其他 $31$ 个线程并赋值给 `value`。

### 3.2 在含有 8 个线程的子线程组内进行包含扫描

```cuda
#include <stdio.h>

__global__ void scan4() {
    int laneId = threadIdx.x & 0x1f;
    // Seed sample starting value (inverse of lane ID)
    int value = 31 - laneId;

    // Loop to accumulate scan within my partition.
    // Scan requires log2(n) == 3 steps for 8 threads
    // It works by an accumulated sum up the warp
    // by 1, 2, 4, 8 etc. steps.
    for (int i=1; i<=4; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        int n = __shfl_up_sync(0xffffffff, value, i, 8);
        if ((laneId & 7) >= i)
            value += n;
    }

    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    scan4<<< 1, 32 >>>();
    cudaDeviceSynchronize();

    return 0;
}
```
上述代码首先在寄存器内定义一个变量 `value` 并将其初始化为 `(31 - laneId)`，也就是说在 warp 内各线程的 `value` 值分别为 $31$、$30$、$\ldots$、$1$、$0$。随后通过调用 `__shfl_up_sync(0xffffffff, value, i, 8)` 分别在 $4$ 个子线程组（每个子线程组包含 $8$ 个线程）内实现了 Hillis Steele Scan 算法，从而组内包含扫描任务。

### 3.3 束内规约

```cuda
#include <stdio.h>

__global__ void warpReduce() {
    int laneId = threadIdx.x & 0x1f;
    // Seed starting value as inverse lane ID
    int value = 31 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);

    // "value" now contains the sum across all threads
    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    warpReduce<<< 1, 32 >>>();
    cudaDeviceSynchronize();

    return 0;
}
```
上述代码，基于折半规约的思想，调用 `__shfl_xor_sync(0xffffffff, value, i, 32)` 将 warp 内 `value` 的值规约求和到通道 ID 为 $0$ 的线程中。

## 4 小结
相比使用共享内存进行线程间数据交换，束内洗牌函数具有如下特点：
- 不需要为参与数据交换的 warp 分配共享内存，这样可以减少共享内存的使用。
- warp shuffle 可以直接交换，不需要进行显式同步。
- warp shuffle 的效率要高于基于共享内存的数据交换，因此除非 warp shuffle 满足不了计算诉求，否则都应该使用 warp shuffle 而不是共享内存。