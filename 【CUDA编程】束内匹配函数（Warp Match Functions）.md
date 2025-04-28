#! https://zhuanlan.zhihu.com/p/670290502
# 【CUDA编程】束内匹配函数（Warp Match Functions）

**写在前面**：本文主要介绍了 cuda 编程中束内匹配函数的计算逻辑和使用方法，大部分内容来自于 CUDA C Programming Guide 的原文，笔者尽量以简单朴实话对原文进行阐述，由于笔者水平有限，文章若有错漏之处，欢迎读者批评指正。

## 1 束内匹配函数

在计算能力 $7.x$ 及以上的 GPU 设备中，Nvidia 提供了一类名为 Warp Match Functions 的内置函数，该函数在设备端执行，主要用于在 warp 中的线程之间执行变量的广播和比较操作。

## 2 函数列表

目前官方提供了如下两个束内匹配函数：
```cuda
unsigned int __match_any_sync(unsigned mask, T value);
unsigned int __match_all_sync(unsigned mask, T value, int *pred);
```

类型 `T` 可以是 `int`、`unsigned int`、`long`、`unsigned long`、`long long`、`unsigned long long`、`float` 或 `double` 等类型。

## 3 计算逻辑

内部函数 `__match_sync()` 允许在对 `mask` 中指定的线程进行同步之后，在不同的线程之间广播和比较一个值 `value`。
- `__match_any_sync`：返回一个 $32$ bit 无符号整数，将 `mask` 中指定的线程中的变量 `value` 与当前线程中的 `value` 比较，把 `value` 值与当前线程相同的线程对应的 bit 位设置为 $1$，注意，当前线程对应的 bit 位也设置为 $1$。例如，在 $0$ 号线程内执行此函数，假设 $0$ 号线程的 `value` 等于 $5$，比较后发现只有 $1$ 号线程的 `value` 等于 $5$，而其他线程的 `value` 等于 $6$，则对于 $0$ 号和 $1$ 号线程来说，`__match_any_sync` 返回值为 `0x3`，而其他线程中 `__match_any_sync` 返回 `fffffffc`。
- `__match_all_sync`：如果 `mask` 中指定的所有线程中的变量 `value` 值都相同，则返回 `mask`，`pred` 设置为 `true`；否则返回 $0$，`pred` 设置为 `false`。 

新的 `*_sync` 型束内匹配内部函数采用一个掩码 `mask` 指定 warp 中参与调用的线程。掩码中的每个 bit 代表了一个参与调用的线程，其位置对应于线程在 warp 中的编号（`lane_id`）。掩码中指定的所有活动线程必须使用相同的掩码执行相同的内部代码，否则结果未定义。

## 4 示例代码

```cuda
#include <stdio.h>


__global__ void testWarpMatch() {
    int a = threadIdx.x > 1 ? 6 : 5;
    int pred;
    unsigned int ret_all = __match_all_sync(0xffffffff, a, &pred);
    unsigned int ret_any = __match_any_sync(0xffffffff, a);
    printf("threadId: %d  match_all: %x  match_any: %x  pred: %d\n", threadIdx.x, ret_all, ret_any, pred);

}


int main() {
    testWarpMatch<<<1, 32>>>();

    return 0;
}

/**
PS E:\IT\cudaLearning\warpFunc> .\warpMatch.exe
threadId: 0  match_all: 0  match_any: 3  pred: 0
threadId: 1  match_all: 0  match_any: 3  pred: 0
threadId: 2  match_all: 0  match_any: fffffffc  pred: 0
threadId: 3  match_all: 0  match_any: fffffffc  pred: 0
threadId: 4  match_all: 0  match_any: fffffffc  pred: 0
threadId: 5  match_all: 0  match_any: fffffffc  pred: 0
threadId: 6  match_all: 0  match_any: fffffffc  pred: 0
threadId: 7  match_all: 0  match_any: fffffffc  pred: 0
threadId: 8  match_all: 0  match_any: fffffffc  pred: 0
threadId: 9  match_all: 0  match_any: fffffffc  pred: 0
threadId: 10  match_all: 0  match_any: fffffffc  pred: 0
threadId: 11  match_all: 0  match_any: fffffffc  pred: 0
threadId: 12  match_all: 0  match_any: fffffffc  pred: 0
threadId: 13  match_all: 0  match_any: fffffffc  pred: 0
threadId: 14  match_all: 0  match_any: fffffffc  pred: 0
threadId: 15  match_all: 0  match_any: fffffffc  pred: 0
threadId: 16  match_all: 0  match_any: fffffffc  pred: 0
threadId: 17  match_all: 0  match_any: fffffffc  pred: 0
threadId: 18  match_all: 0  match_any: fffffffc  pred: 0
threadId: 19  match_all: 0  match_any: fffffffc  pred: 0
threadId: 20  match_all: 0  match_any: fffffffc  pred: 0
threadId: 21  match_all: 0  match_any: fffffffc  pred: 0
threadId: 22  match_all: 0  match_any: fffffffc  pred: 0
threadId: 23  match_all: 0  match_any: fffffffc  pred: 0
threadId: 24  match_all: 0  match_any: fffffffc  pred: 0
threadId: 25  match_all: 0  match_any: fffffffc  pred: 0
threadId: 26  match_all: 0  match_any: fffffffc  pred: 0
threadId: 27  match_all: 0  match_any: fffffffc  pred: 0
threadId: 28  match_all: 0  match_any: fffffffc  pred: 0
threadId: 29  match_all: 0  match_any: fffffffc  pred: 0
threadId: 30  match_all: 0  match_any: fffffffc  pred: 0
threadId: 31  match_all: 0  match_any: fffffffc  pred: 0
 */
```