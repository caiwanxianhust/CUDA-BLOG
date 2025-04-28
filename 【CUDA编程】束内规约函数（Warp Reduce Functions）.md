#! https://zhuanlan.zhihu.com/p/670534280
# 【CUDA编程】束内规约函数（Warp Reduce Functions） 

**写在前面**：本文主要介绍了 cuda 编程中束内规约函数的计算逻辑和使用方法，大部分内容来自于 CUDA C++ Programming Guide 的原文，笔者尽量以简单朴实话对原文进行阐述，由于笔者水平有限，文章若有错漏之处，欢迎读者批评指正。

## 1 束内规约函数的应用背景
束内规约函数（Warp Reduce Functions）是 Nvidia 新引入的一种设备端的内置函数，顾名思义，主要用于进行 warp 内数据的规约操作。此类函数仅受计算能力 $8.x$ 及以上的 GPU 设备支持。

同样地，束内规约函数在写程序的时候也不是必须的，开发人员完全可以不借助它写出功能完备的代码，事实上由于这类函数引入时间较晚，在绝大部分应用程序中都没有被使用。通常我们在进行规约操作的时候，需要通过共享内存进行 warp 内线程间的通信（参见笔者另一篇文章：[【CUDA编程】束内规约与块内规约问题](https://zhuanlan.zhihu.com/p/652255520)），结合折半规约的思想进行规约操作。随着束内规约函数的出现，使得规约操作变得简单粗暴，直接调用内置函数即可，性能要比共享内存高出不少，所以束内规约函数的应用场景非常广泛，随着 GPU 设备的更新肯定会大量使用。

## 2 函数列表

内部函数 `__reduce_sync(unsigned mask, T value)` 在对 `mask` 中指定的线程进行同步之后，在不同的线程中的变量 `value` 执行规约操作。对于 add、min、max 等操作，`T` 可以是无符号的或有符号的类型，而对于 and、or、xor 等操作，`T` 只能是无符号的类型。目前官方提供的函数列表如下：

```cuda
// add/min/max
unsigned __reduce_add_sync(unsigned mask, unsigned value);
unsigned __reduce_min_sync(unsigned mask, unsigned value);
unsigned __reduce_max_sync(unsigned mask, unsigned value);
int __reduce_add_sync(unsigned mask, int value);
int __reduce_min_sync(unsigned mask, int value);
int __reduce_max_sync(unsigned mask, int value);

// and/or/xor
unsigned __reduce_and_sync(unsigned mask, unsigned value);
unsigned __reduce_or_sync(unsigned mask, unsigned value);
unsigned __reduce_xor_sync(unsigned mask, unsigned value);
```

## 3 计算逻辑

束内规约函数（Warp Reduce Functions）按照操作类型主要分为以下两类：
- `__reduce_add_sync`、`__reduce_min_sync`、`__reduce_max_sync`：返回对 `mask` 中指定的线程中的变量 `value` 的值进行加法（add）、最小（min）或最大（max）规约操作的计算结果。
- `__reduce_and_sync`、`__reduce_or_sync`、`__reduce_xor_sync`：返回对 `mask` 中指定的线程中的变量 `value` 的值进行逻辑与（AND）、逻辑或（OR）或逻辑异或（XOR）等规约操作的计算结果。

以上函数参数中的掩码 `mask` 指定了 warp 中参与调用的线程。掩码中的每个 bit 代表了一个参与调用的线程，其位置对应于线程在 warp 中的编号（`lane_id`）。掩码中指定的所有活动线程必须使用相同的掩码执行相同的内部代码，否则结果未定义。

## 4 示例代码

以下代码演示了 kernel 中调用束内规约函数的样例，由于笔者手头暂时没有 $8.x$ 及以上的 GPU 设备，所以无法得到实际计算结果，有条件的读者可以尝试。 
```cuda
#include <stdio.h>


__global__ void testWarpReduce() {

    int a = threadIdx.x;
    int ret_add = __reduce_add_sync(0xffffffff, a);
    int ret_min = __reduce_min_sync(0xffffffff, a);
    int ret_max = __reduce_max_sync(0xffffffff, a);

    unsigned int b = a & 1;
    unsigned int ret_and = __reduce_and_sync(0xffffffff, b);
    unsigned int ret_or = __reduce_or_sync(0xffffffff, b);
    unsigned int ret_xor = __reduce_xor_sync(0xffffffff, b);

    printf("threadId: %d  reduce_add: %d  reduce_min: %d  reduce_max: %d  reduce_and: %x  reduce_or: %x  reduce_xor: %x\n",
        threadIdx.x, ret_add, ret_min, ret_max, ret_and, ret_or, ret_xor);

}

int main() {
    testWarpReduce<<<1, 32>>>();
    return 0;
}
```