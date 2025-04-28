#! https://zhuanlan.zhihu.com/p/669917716
# 【CUDA编程】束内表决函数（Warp Vote Function）

**写在前面**：本文主要介绍了 cuda 编程中束内表决函数的计算逻辑和使用方法，大部分内容来自于 CUDA C Programming Guide 的原文，笔者尽量以简单朴实话对原文进行阐述，由于笔者水平有限，文章若有错漏之处，欢迎读者批评指正。

## 1 束内表决函数的应用背景
束内表决函数（Warp Vote Function）是设备端的内置函数，主要用于进行一些 warp 内数据交换操作，总的来说具有两个功能：1-bit 数据交换、1-bit 数据规约。为什么说是 1-bit，后面讲到函数计算逻辑时读者们自然会明白。

需要注意的是，束内表决函数在写程序的时候不是必须的，至少笔者阅读的一些大厂和 Nividia 官方的源码中很少看到应用，但是有了这个函数，可以让开发人员多一种选择，没准在某个场景恰好用得上。为什么说用得少？一方面是这个系列的内置函数出得晚，早期束内数据交换主要使用共享内存，另一方面是可以进行束内数据交换的函数是在太多，比如它的兄弟系列，如束内规约函数、束内匹配函数、束内洗牌函数等等。

## 2 函数列表
目前官方主要提供了以下 4 种束内表决函数：
```
int __all_sync(unsigned mask, int predicate);
int __any_sync(unsigned mask, int predicate);
unsigned __ballot_sync(unsigned mask, int predicate);
unsigned __activemask();
```

弃用通知：`__any`、`__all` 和 `__ballot` 这三个函数在 CUDA 9.0 中已针对所有设备弃用。

删除通知：在计算能力不低于 $7.x$ 的设备上，`__any`、`__all` 和 `__ballot` 这三个函数不再可用，而应使用与它们的对应的同步型的变体（加了 `_sync` 后缀的）。

## 3 计算逻辑

束内表决函数允许给定 warp 中的线程在 warp 内进行规约和广播操作。这些函数将 warp 中每个线程的 `int` 型表达式 `predicate` 作为输入，并将这些值与 $0$ 进行比较。比较的结果通过以下方式之一在 warp 的活动线程中组合（规约），向每个参与线程广播单个返回值：
- `__all_sync(unsigned mask, predicate)`：评估 `mask` 中所有未退出线程的 `predicate`，当且仅当 `predicate` 对所有线程的评估结果都为非零时，才返回非零值。
- `__any_sync(unsigned mask, predicate)`：评估 `mask` 中所有未退出线程的 `predicate`，只要 `predicate` 对其中任意一个线程的评估结果为非零时，即返回非零值。
- `__ballot_sync(unsigned mask, predicate)`：返回一个 $32$ 位无符号整数，代表了 warp 内变量 `predicate` 的非零值分布情况。当且仅当 warp 的第 N 个线程的 `predicate` 为非零并且第 N 个线程处于活动状态时，该 $32$ 位整数的第 N 位为 $1$。
- `__activemask()`：返回一个 $32$ 位无符号整数，代表了 warp 内所有线程的当前活动状态。当 warp 中的第 N 条线程调用 `__activemask()` 时处于活动状态，则第 N 位设置为 $1$，否则则为非活动线程，相应位置设置为 $0$。已经退出程序的线程总是被标记为非活动的。

虽然表达式 `predicate` 是 `int` 型，有 32-bit，但是最终都是跟 $0$ 比较，所以编译器会在需要的时候自动转换为 1-bit 型，实际硬件上比较的其实是 1-bit 类型。`all` 版本判断它们是否全部为真，而 `any` 版本则判断它们是否至少有 $1$ 个为真，很多代码需要这种操作的，例如可以快速的判断 warp 整体（或者某些部分）是否需要进行某种操作，如果整体需要或者不需要，可以快速的越过一些片段，而不需要在片段内部逐步处理之类的。

对于 `__all_sync`、`__any_sync` 和 `__ballot_sync`，必须传递一个掩码 `mask` 来指定 warp 中参与调用的线程。掩码中的每个 bit 代表了一个参与调用的线程，其位置对应于线程在 warp 中的编号（`lane_id`）。掩码中指定的所有活动线程必须使用相同的掩码执行相同的内部代码，否则结果未定义。

## 4 代码示例

```cuda
#include <stdio.h>


__global__ void testWarpVote() {
    int a = threadIdx.x & 1;
    int ret_all = __all_sync(0xffffffff, a);
    int ret_any = __any_sync(0xffffffff, a);
    unsigned int ret_ballot = __ballot_sync(0xffffffff, a);
    unsigned int ret_activemask = __activemask();
    printf("threadId: %d  all: %d  any: %d  ballot: %x  activemask: %x\n", threadIdx.x, ret_all, ret_any, ret_ballot, ret_activemask);
}


int main() {

    testWarpVote<<<1, 32>>>();

    return 0;
}

/**
threadId: 0  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 1  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 2  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 3  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 4  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 5  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 6  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 7  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 8  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 9  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 10  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 11  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 12  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 13  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 14  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 15  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 16  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 17  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 18  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 19  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 20  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 21  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 22  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 23  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 24  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 25  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 26  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 27  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 28  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 29  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 30  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
threadId: 31  all: 0  any: 1  ballot: aaaaaaaa  activemask: ffffffff
 */
```