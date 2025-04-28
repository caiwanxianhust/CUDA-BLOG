#! https://zhuanlan.zhihu.com/p/678283126
![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/GJUG0H1sS5qAxgP6xzdfAUic6fRibC7gvYTNskHDibj3jmDVibHdnI0esy8RlIkbnmzGPeImaTJpJAdceAPOicUWENw/640?wx_fmt=jpeg&amp;from=appmsg)

# 【CUDA编程】数学函数（Mathematical Functions）

**写在前面**：本文是笔者手稿的第 13 章的内容，主要介绍了在 CUDA 程序开发过程中数学函数的用法及其精度范围，选择合适的数学函数，对应用程序性能的提升有重要意义。如有错漏之处，请读者们务必指出，感谢！

CUDA 参考手册（NVIDIA CUDA Reference Manual）列出了设备代码中支持的 C/C++ 标准库数学函数的所有函数及其描述，以及所有内部函数（仅在设备代码中支持）。

本章提供了其中一些函数的准确性指标，具体指标使用 ULP 进行量化。
关于 ULP 的定义，请参阅 Jean-Michel Muller 的论文 On the definition of ulp(x)，这里给出论文链接：https://hal.inria.fr/inria-00070503/document

设备代码中支持的数学函数不设置全局 `errno` 变量，也不进行异常报错。因此，如果需要错误诊断机制，用户应该对函数的输入和输出进行额外的检查。应用程序在使用数学函数时，应当保证指针参数的有效性，不能传入野指针。同时，也不允许将未初始化的参数传递给数学函数，这会导致未定义的行为：函数内联在用户程序中，因此受到编译器优化的影响。

## 1 标准函数 
本节中介绍的函数既可在设备代码中使用，也可在主机代码中使用。

本节中指明了每个函数在设备上执行时的误差界限（error bounds），以及在主机不提供函数的情况下在主机上执行时的误差界限。

误差界限是基于大量的测试用例确定的，并不是一个绝对的界限。

### 1.1 单精度浮点函数

加法和乘法符合 IEEE 标准，因此最大误差为 $0.5$ ulp。

将单精度浮点操作数舍入为整数的推荐方法是 `rintf()`，而不是 `roundf()`。 原因是 `roundf()` 映射到设备上的 $4$ 条指令序列，而 `rintf()` 映射到单个指令。类似地，`truncf()`、`ceilf()` 和 `floorf()` 等函数也都只映射到一条指令。

表 \ref{tab:Single-Precision-Functions-with-Maximum-ULP-Error} 展示了单精度数学标准库函数的最大 ULP 误差。最大误差表示为正确舍入的单精度结果与 CUDA 库函数返回的结果之间的差值的绝对值，单位为 ulps。

| 函数 | 最大 ULP 误差 |
|---|---|
| `x + y` | $0$（IEEE-754 round-to-nearest-even） |
| `x * y` | $0$（IEEE-754 round-to-nearest-even） |
| `x / y` | 使用 `-prec-div=true` 编译且计算能力大于等于 2 时为 $0$ ，否则为 $2$ |
| `1 / x` | 使用 `-prec-div=true` 编译且计算能力大于等于 2 时为 $0$ ，否则为 $1$ |
| `rsqrtf(x)`、`1/sqrtf(x)` | $2$，仅当编译器自动将 `1/sqrtf(x)` 转换为 `rsqrtf(x)` 时对 `1/sqrtf(x)` 适用 |
| `sqrtf(x)` | 使用 `-prec-div=true` 编译时为 $0$ ，否则计算能力大于等于 5.2 时为 $1$，否则旧架构为 $3$ |
| `cbrtf(x)` | $1$ |
| `rcbrtf(x)` | $1$ |
| `hypotf(x, y)` | $3$ |
| `rhypotf(x, y)` | $2$ |
| `norm3df(x, y, z)` | $3$ |
| `rnorm3df(x, y, z)` | $2$ |
| `norm4df(x, y, z, t)` | $3$ |
| `rnorm4df(x, y, z, t)` | $2$ |
| `normf(dim,arr)` | 由于使用了快速算法，但由于四舍五入导致精度损失，因此无法提供误差范围。 |
| `rnormf(dim,arr)` | 由于使用了快速算法，但由于四舍五入导致精度损失，因此无法提供误差范围。 |
| `expf(x)` | $2$ |
| `exp2f(x)` | $2$ |
| `exp10f(x)` | $2$ |
| `expm1f(x)` | $1$ |
| `logf(x)` | $1$ |
| `log2f(x)` | $1$ |
| `log10f(x)` | $2$ |
| `log1pf(x)` | $1$ |
| `sinf(x)` | $2$ |
| `cosf(x)` | $2$ |
| `tanf(x)` | $4$ |
| `sincosf(x, sptr, cptr)` | $2$ |
| `sinpif(x)` | $1$ |
| `cospif(x)` | $1$ |
| `sincospif(x, sptr, cptr)` | $1$ |
| `asinf(x)` | $2$ |
| `acosf(x)` | $2$ |
| `atanf(x)` | $2$ |
| `atan2f(x)` | $3$ |
| `sinhf(x)` | $3$ |
| `coshf(x)` | $2$ |
| `tanhf(x)` | $2$ |
| `asinhf(x)` | $3$ |
| `acoshf(x)` | $4$ |
| `atanhf(x)` | $3$ |
| `powf(x, y)` | $4$ |
| `erff(x)` | $2$ |
| `erfcf(x)` | $4$ |
| `erfinvf(x)` | $2$ |
| `erfcinvf(x)` | $4$ |
| `erfcxf(x)` | $4$ |
| `normcdff(x)` | $5$ |
| `normcdfinvf(x)` | $5$ |
| `lgammaf(x)` | $6$ |
| `tgammaf(x)` | $5$ |
| `fmaf(x, y, z)` | $0$ |
| `frexpf(x, exp)` | $0$ |
| `ldexpf(x, exp)` | $0$ |
| `scalbnf(x, n)` | $0$ |
| `scalblnf(x, l)` | $0$ |
| `logbf(x)` | $0$ |
| `ilogbf(x)` | $0$ |
| `j0f(x)` | $\|x\| < 8$ 时为 $9$；其他情况下最大绝对误差为 $2.2\times10^{-6}$ |
| `j1f(x)` | $\|x\| < 8$ 时为 $9$；其他情况下最大绝对误差为 $2.2\times10^{-6}$ |
| `jnf(n, x)` | $n = 128$ 时最大绝对误差为 $2.2\times10^{-6}$ |
| `y0f(x)` | $\|x\| < 8$ 时为 $9$；其他情况下最大绝对误差为 $2.2\times10^{-6}$ |
| `y1f(x)` | $\|x\| < 8$ 时为 $9$；其他情况下最大绝对误差为 $2.2\times10^{-6}$ |
| `ynf(n, x)` | $\|x\| < n$ 时为 `ceil(2 + 2.5n)`；其他情况下最大绝对误差为 $2.2\times10^{-6}$ |
| `cyl_bessel_i0f(x)` | $6$ |
| `cyl_bessel_i1f(x)` | $6$ |
| `fmodf(x, y)` | $0$ |
| `remainderf(x, y)` | $0$ |
| `remquof(x, y, iptr)` | $0$ |
| `modff(x, iptr)` | $0$ |
| `fdimf(x, y)` | $0$ |
| `truncf(x)` | $0$ |
| `roundf(x)` | $0$ |
| `rintf(x)` | $0$ |
| `nearbyintf(x)` | $0$ |
| `ceilf(x)` | $0$ |
| `floorf(x)` | $0$ |
| `lrintf(x)` | $0$ |
| `lroundf(x)` | $0$ |
| `llrintf(x)` | $0$ |
| `llroundf(x)` | $0$ |


### 1.2 双精度浮点函数

推荐使用 `rint()` 函数将双精度浮点数舍入为整数，而不是 `round()`，后者映射到设备上的 $5$ 条指令序列，而前者只映射到单个指令。类似地，`trunc()`、`ceil()` 和 `floor()` 也都映射到一条指令。

下表展示了双精度数学标准库函数的最大 ULP 误差。最大误差表示为正确舍入的双精度结果与 CUDA 库函数返回的结果之间的差值的绝对值，单位为 ulps。

| 函数 | 最大 ULP 误差 |
|---|---|
| `x + y` | $0$（IEEE-754 round-to-nearest-even） |
| `x * y` | $0$（IEEE-754 round-to-nearest-even） |
| `x / y` | $0$（IEEE-754 round-to-nearest-even） |
| `1 / x` | $0$（IEEE-754 round-to-nearest-even） |
| `sqrt(x)` | $0$（IEEE-754 round-to-nearest-even） |
| `rsqrt(x)` | $1$ |
| `cbrt(x)` | $1$ |
| `rcbrt(x)` | $1$ |
| `hypot(x, y)` | $2$ |
| `rhypot(x, y)` | $1$ |
| `norm3d(x, y, z)` | $2$ |
| `rnorm3d(x, y, z)` | $1$ |
| `norm4d(x, y, z, t)` | $2$ |
| `rnorm4d(x, y, z, t)` | $1$ |
| `norm(dim, arr)` | 由于使用了快速算法，但由于四舍五入导致精度损失，因此无法提供误差范围。 |
| `rnorm(dim, arr)` | 由于使用了快速算法，但由于四舍五入导致精度损失，因此无法提供误差范围。 |
| `exp(x)` | $1$ |
| `exp2(x)` | $1$ |
| `exp10(x)` | $1$ |
| `expm1(x)` | $1$ |
| `log(x)` | $1$ |
| `log2(x)` | $1$ |
| `log10(x)` | $1$ |
| `log1p(x)` | $1$ |
| `sin(x)` | $2$ |
| `cos(x)` | $2$ |
| `tan(x)` | $2$ |
| `sincos(x, sptr, cptr)` | $2$ |
| `sinpi(x)` | $2$ |
| `cospi(x)` | $2$ |
| `sincospi(x, sptr, cptr)` | $2$ |
| `asin(x)` | $2$ |
| `acos(x)` | $2$ |
| `atan(x)` | $2$ |
| `atan2(x)` | $2$ |
| `sinh(x)` | $2$ |
| `cosh(x)` | $1$ |
| `tanh(x)` | $1$ |
| `asinh(x)` | $3$ |
| `acosh(x)` | $3$ |
| `atanh(x)` | $2$ |
| `pow(x, y)` | $2$ |
| `erf(x)` | $2$ |
| `erfc(x)` | $5$ |
| `erfinv(x)` | $5$ |
| `erfcinv(x)` | $6$ |
| `erfcx(x)` | $4$ |
| `normcdf(x)` | $5$ |
| `normcdfinv(x)` | $8$ |
| `lgamma(x)` | $4$ |
| `tgamma(x)` | $10$ |
| `fma(x, y, z)` | $0$ （IEEE-754 round-to-nearest-even） |
| `frexp(x, exp)` | $0$ |
| `ldexp(x, exp)` | $0$ |
| `scalbn(x, n)` | $0$ |
| `scalbln(x, l)` | $0$ |
| `logb(x)` | $0$ |
| `ilogb(x)` | $0$ |
| `j0(x)` | $\|x\| < 8$ 时为 $7$；其他情况下最大绝对误差为 $5\times10^{-12}$ |
| `j1(x)` | $\|x\| < 8$ 时为 $7$；其他情况下最大绝对误差为 $5\times10^{-12}$ |
| `jn(n, x)` | $n = 128$ 时最大绝对误差为 $5\times10^{-12}$ |
| `y0(x)` | $\|x\| < 8$ 时为 $7$；其他情况下最大绝对误差为 $5\times10^{-12}$ |
| `y1(x)` | $\|x\| < 8$ 时为 $7$；其他情况下最大绝对误差为 $5\times10^{-12}$ |
| `yn(n, x)` | $\|x\| > 1.5n$ 时最大绝对误差为 $5\times10^{-12}$ |
| `cyl_bessel_i0(x)` | $6$ |
| `cyl_bessel_i1(x)` | $6$ |
| `fmod(x,y)` | $0$ |
| `remainder(x, y)` | $0$ |
| `remquo(x, y, iptr)` | $0$ |
| `modf(x, iptr)` | $0$ |
| `fdim(x, y)` | $0$ |
| `trunc(x)` | $0$ |
| `round(x)` | $0$ |
| `rint(x)` | $0$ |
| `nearbyint(x)` | $0$ |
| `ceil(x)` | $0$ |
| `floor(x)` | $0$ |
| `lrint(x)` | $0$ |
| `lround(x)` | $0$ |
| `llrint(x)` | $0$ |
| `llround(x)` | $0$ |


## 2 内部函数 

本节中介绍的函数只能在设备代码中使用。

这些内部函数的功能与标准库中对应函数相同，但是它们映射的本机指令更少，在舍弃部分精度的基础上提供了更快的计算速度。这些函数以 `__` 作为前缀，例如 `__sinf(x)`。编译器有一个选项 `-use_fast_math`，指定该选项后将在编译时强制下表中的每个函数编译为其对应的内部函数。

内部函数除了会降低函数的计算结果的精度外，还可能在一些特殊情况下与标准函数存在差异。所以推荐通过调用内联函数来选择性地替换标准数学函数，具体是否替换需要用户根据实际任务权衡。

| 函数操作 | 设备函数 |
|---|---|
| `x / y` | `__fdividef(x, y)` |
| `sinf(x)` | `__sinf(x)` |
| `cosf(x)` | `__cosf(x)` |
| `tanf(x)` | `__tanf(x)` |
| `sincosf(x, sptr, cptr)` | `__sincosf(x, sptr, cptr)` |
| `logf(x)` | `__logf(x)` |
| `log2f(x)` | `__log2f(x)` |
| `log10f(x)` | `__log10f(x)` |
| `expf(x)` | `__expf(x)` |
| `exp10f(x)` | `__exp10f(x)` |
| `powf(x, y)` | `__powf(x, y)` |


### 2.1 单精度浮点函数

`__fadd_[rn,rz,ru,rd]()` 和 `__fmul_[rn,rz,ru,rd]()` 分别被编译器映射到加法和乘法运算，但是并不会使用 FMAD（浮点乘加）指令。相比之下，由 `*` 和 `*` 运算符表示的加法和乘法操作通常会被编译器替换为 FMAD 指令。这 $4$ 个后缀分别代表不同的舍入模式，具体介绍如下：

- `_rn`：舍入到最接近的偶数。
- `_rz`：向零舍入。
- `_ru`：向上舍入（到正无穷大）。
- `_rd`：向下舍入（到负无穷大）。


浮点除法的精度取决于代码的编译选项是使用 `-prec-div=false` 还是 `-prec-div=true`。使用 `-prec-div=false` 编译代码时，使用除法 `/` 运算符和 `__fdividef(x,y)` 精度相同，但当 $2^{126} < |y| < 2^{128}$ 时，`__fdividef(x,y)` 提供的结果为零，而 `/` 运算符提供的正确结果在下表中规定的精度范围内。此外，当 $2^{126} < |y| < 2^{128}$ 时，如果 `x` 为无穷大，则 `__fdividef(x,y)` 返回 `NaN`（相当于无穷大乘以零），而 `/` 运算符返回无穷大。另一方面，当编译选项使用 `-prec-div=true` 或根本没有任何 `-prec-div` 相关选项编译代码时，`/` 运算符符合 IEEE 标准，因为 `-prec-div` 的默认值为 `true`。

| 函数 | 误差界限 |
|---|---|
| `__fadd_[rn,rz,ru,rd](x,y)` | IEEE-compliant |
| `__fsub_[rn,rz,ru,rd](x,y)` | IEEE-compliant |
| `__fmul_[rn,rz,ru,rd](x,y)` | IEEE-compliant |
| `__fmaf_[rn,rz,ru,rd](x,y,z)` | IEEE-compliant |
| `__frcp_[rn,rz,ru,rd](x)` | IEEE-compliant |
| `__fsqrt_[rn,rz,ru,rd](x)` | IEEE-compliant |
| `__frsqrt_rn(x)` | IEEE-compliant |
| `__fdiv_[rn,rz,ru,rd](x,y)` | IEEE-compliant |
| `__fdividef(x,y)` | $2^{-126} < |y| < 2^{126}$ 时最大 ulp 误差为 $2$ |
| `__expf(x)` | 最大 ulp 误差为 `2 + floor(abs(1.173 * x))` |
| `__exp10f(x)` | 最大 ulp 误差为 `2 + floor(abs(1.173 * x))` |
| `__logf(x)` | 当 $x \in [0.5,2]$ 时，最大绝对值误差为 $2^{-21.41}$，否则最大 ulp 误差为 $3$ |
| `__log2f(x)` | 当 $x \in [0.5,2]$ 时，最大绝对值误差为 $2^{-22}$，否则最大 ulp 误差为 $2$ |
| `__log10f(x)` | 当 $x \in [0.5,2]$ 时，最大绝对值误差为 $2^{-24}$，否则最大 ulp 误差为 $3$ |
| `__sinf(x)` | 当 $x \in [-\pi,\pi]$ 时，最大绝对值误差为 $2^{-21.41}$，否则误差更大 |
| `__cosf(x)` | 当 $x \in [-\pi,\pi]$ 时，最大绝对值误差为 $2^{-21.19}$，否则误差更大 |
| `__sincosf(x,sptr,cptr)` | 与 `__sinf(x)` 和 `__cosf(x)` 相同 |
| `__tanf(x)` | 相当于 `__sinf(x) * (1/__cosf(x))` |
| `__powf(x,y)` | 相当于 `exp2f(y * __log2f(x))` |


### 2.2 双精度浮点函数

`__dadd_rn()` 和 `__dmul_rn()` 分别被编译器映射到加法和乘法运算，但是并不会使用 FMAD（浮点乘加）指令。相比之下，由 `*` 和 `*` 运算符表示的加法和乘法操作通常会被编译器替换为 FMAD 指令。

| 函数 | 误差界限 |
|---|---|
| `__dadd_[rn,rz,ru,rd](x,y)` | IEEE-compliant |
| `__dsub_[rn,rz,ru,rd](x,y)` | IEEE-compliant |
| `__dmul_[rn,rz,ru,rd](x,y)` | IEEE-compliant |
| `__fma_[rn,rz,ru,rd](x,y,z)` | IEEE-compliant |
| `__ddiv_[rn,rz,ru,rd](x,y)(x,y)` | IEEE-compliant，要求计算能力大于 2 |
| `__drcp_[rn,rz,ru,rd](x)` | IEEE-compliant，要求计算能力大于 2 |
| `__dsqrt_[rn,rz,ru,rd](x)` | IEEE-compliant，要求计算能力大于 2 |



