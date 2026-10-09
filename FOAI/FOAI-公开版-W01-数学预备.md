---
title: FOAI W01｜数学预备
course: FOAI
week: 1
type: lecture
tags:
  - AI-MSc
  - TB1
  - FOAI
  - 数学基础
source: 课程材料（Conor Houghton、Roberto Colomboni）的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: draft
publication: pending
updated: 2026-10-09
---

# FOAI W01｜数学预备（Mathematical Preliminaries）

> [!IMPORTANT]
> **非官方学习笔记**：这是我在 University of Bristol「Foundations of Artificial Intelligence（SEMTM0051, 2026/27）」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：Conor Houghton、Roberto Colomboni。原始材料：[semtm0051/2026_27](https://github.com/semtm0051/2026_27/tree/322787dec91f174303ccf90399bb6243a060eb7d/)。经两位讲师同意公开；不含练习解答与非公开材料（如课堂录像）。


> [!NOTE]
> **资料来源**
> 一手讲义：[`Week_01/Notes/Week_01 - Mathematical preliminaries.pdf`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_01/Notes/Week_01%20-%20Mathematical%20preliminaries.pdf)。
> 配套 slides：`Week_01/Slides/Week_01 - Slides.pptx`。
> 本笔记是对讲义的提炼，公式编号沿用讲义。

本讲是全课程的数学工具底座：求和/乘积 → 指数对数 → 向量矩阵 → Kronecker delta → 导数与梯度 → 用梯度找最小值。后续每一周都会反复用到这里的记号和结论。

---

## 1. 求和与乘积

$$\sum_{i=1}^{N} x_i := x_1 + \cdots + x_N, \qquad \prod_{i=1}^{N} x_i := x_1 \cdots x_N.$$

上下文清晰时可省略上下限，写作 $\sum_i x_i$。

## 2. 指数与对数（`log` 一律指自然对数）

- $e^s > 0$，$e^0 = 1$，$\log 1 = 0$；$e^{\log x} = x$（$x>0$）。
- **乘积转求和**：$\log(bc) = \log b + \log c$，$\log(u/v) = \log u - \log v$，更一般地
  $$\log\!\left(\prod_{i=1}^{N} x_i\right) = \sum_{i=1}^{N} \log x_i.$$
- 工程意义：① 求和比连乘好求导；② 很多小数相乘会下溢，直接加对数可避免。
- **单调递增**：$u < v \iff \log u < \log v$。因此"最大化一个正数"等价于"最大化它的对数"或"最小化它的负对数"。
- 对概率 $p \in (0,1)$：$-\log p > 0$，且 $p=1$ 时为 0。这就是交叉熵里"负对数似然"恒正的来源。

## 3. 向量、矩阵与下标

| 对象 | 记号 | 例子 |
|---|---|---|
| 标量 | 正体 | $a \in \mathbb{R}$ |
| 向量 | **粗体小写** | $\boldsymbol{x} \in \mathbb{R}^d$，分量 $x_j$ |
| 矩阵 | **粗体大写** | $\boldsymbol{A} \in \mathbb{R}^{m \times n}$，分量 $A_{ij}$ |

课程约定：**输入特征向量是行**，**单个输出的权重是列**：
$$\boldsymbol{x} = (x_1,\dots,x_d) \in \mathbb{R}^{1\times d}, \qquad \boldsymbol{w} = (w_1,\dots,w_d)^\top \in \mathbb{R}^{d\times 1}.$$

### 3.1 转置
$(\boldsymbol{A}^\top)_{ji} = A_{ij}$；$(\boldsymbol{A}^\top)^\top = \boldsymbol{A}$；$(\boldsymbol{A}\boldsymbol{B})^\top = \boldsymbol{B}^\top \boldsymbol{A}^\top$（顺序反转）。

### 3.2 矩阵乘法
$\boldsymbol{A}\in\mathbb{R}^{m\times n}$，$\boldsymbol{B}\in\mathbb{R}^{n\times p}$：
$$C_{ik} = \sum_{j=1}^{n} A_{ij} B_{jk}, \qquad \underbrace{\boldsymbol{A}}_{m\times n}\underbrace{\boldsymbol{B}}_{n\times p} = \underbrace{\boldsymbol{C}}_{m\times p}.$$
- 内维必须相等；$i,k$ 在求和中固定，$j$ 跑遍所有配对后消失。
- 一般**不可交换**：$\boldsymbol{A}\boldsymbol{B} \neq \boldsymbol{B}\boldsymbol{A}$。

### 3.3 内积 vs 外积
- 内积（行×列 = 标量）：$\boldsymbol{x}\boldsymbol{w} = \sum_j x_j w_j$；$\boldsymbol{w}^\top\boldsymbol{w} = \sum_j w_j^2 = \|\boldsymbol{w}\|_2^2$。
- 外积（列×行 = 矩阵）：$\boldsymbol{u}\boldsymbol{v}^\top$，$M_{ik} = u_i v_k$。

### 3.4 单位矩阵与逆矩阵
- $\boldsymbol{I}_n$：对角为 1，其余为 0。
- 二阶逆矩阵：$\boldsymbol{A}=\begin{pmatrix}a&b\\c&d\end{pmatrix}$，$\boldsymbol{A}^{-1} = \dfrac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}$，要求行列式 $ad-bc \neq 0$。
- 解线性系统 $\boldsymbol{A}\boldsymbol{u} = \boldsymbol{c} \Rightarrow \boldsymbol{u} = \boldsymbol{A}^{-1}\boldsymbol{c}$。

## 4. Kronecker delta 与指示函数

$$\delta_{ij} := \begin{cases}1 & i=j\\ 0 & i\neq j\end{cases}; \qquad \mathbf{1}\{A\} := \begin{cases}1 & A\text{ 为真}\\ 0 & \text{否则}\end{cases}.$$

- $\delta_{ij} = \mathbf{1}\{i=j\}$，就是单位矩阵的元素。
- **筛出一项**：$\sum_j \delta_{ij} a_j = a_i$（求导时会反复用）。
- 错误数 = $\sum_i \mathbf{1}\{\hat y_i \neq y_i\}$。

## 5. 导数与梯度

### 5.1 一元导数
$$f'(x_0) = \lim_{h\to 0} \frac{f(x_0+h)-f(x_0)}{h}.$$
**局部线性近似**：$f(x+h) \approx f(x) + f'(x)h$。

### 5.2 常用导数
- 幂法则：$\dfrac{d}{dx}x^p = p x^{p-1}$。
- $\dfrac{d}{dx} e^x = e^x$；$\dfrac{d}{dx}\log x = 1/x$（$x>0$）。
- 求导线性：$(a u + b v)' = a u' + b v'$。

### 5.3 链式法则（反向传播的核心）
$F(x) = v(u(x))$，则
$$F'(x) = v'(u(x))\, u'(x).$$
外层导数在**内层输出处**取值。Backprop 就是反复套链式法则。

### 5.4 乘积法则
$(uv)' = u'v + uv'$。

### 5.5 偏导数
对 $F(a,b)$，固定一个变量对另一个求导：
$$\partial_1 F = \frac{\partial F}{\partial a}, \qquad \partial_2 F = \frac{\partial F}{\partial b}.$$

### 5.6 梯度
对列向量 $\boldsymbol{w}=(w_1,\dots,w_d)^\top$，梯度是把所有偏导收集成同形状列向量：
$$\nabla Q(\boldsymbol{w}) = \begin{pmatrix}\partial_1 Q(\boldsymbol{w}) \\ \vdots \\ \partial_d Q(\boldsymbol{w})\end{pmatrix}.$$
局部线性近似：$Q(\boldsymbol{w}+\Delta) \approx Q(\boldsymbol{w}) + \nabla Q(\boldsymbol{w})^\top \Delta$。

### 5.7 分量求导（为下周回归铺垫）
$\dfrac{\partial r_i}{\partial r_j} = \delta_{ij}$。
若 $s(\boldsymbol{w}) = \sum_j x_j w_j$，则 $\dfrac{\partial s}{\partial w_k} = x_k$。

## 6. 用导数找最小值

### 6.1 单参数
- **全局最小**：$\mathcal{L}(a^*) \le \mathcal{L}(a)$ 对所有 $a$ 成立。
- **Fermat 定理**：若可微函数在 $a^*$ 取到局部最小，则 $\mathcal{L}'(a^*) = 0$。
- 导数为 0 的点叫**驻点**，但驻点不一定是最小（$a^2$ 最小、$-a^2$ 最大、$a^3$ 都不是）。
- **min vs argmin**：
  - $\min_a F(a)$ = 最小**值**（一个数）；
  - $\operatorname{argmin}_a F(a)$ = 取到最小值的**参数集合**。
  - 同理有 max / argmax。平局要指定 tie-breaking 规则。

### 6.2 多参数
$\nabla \mathcal{L}(\boldsymbol{\theta}^*) = \boldsymbol{0}$ 是局部最小的**必要不充分**条件。
反例 $F(\boldsymbol{u}) = u_1^2 - u_2^2$：$\nabla F(\boldsymbol{0}) = \boldsymbol{0}$，但四面八方有升有降——这是**鞍点**。

> [!TIP]
> **全课程会重复出现的范式**
> 选一个函数族 → 选一个损失函数 → 对经验损失求梯度 → 令梯度为 0（解析解）或沿负梯度迭代（梯度下降）。

## 7. 习题要点（讲义 Section 7）

- **Ex 12**：$\boldsymbol{g}(\boldsymbol{w}) = \boldsymbol{X}\boldsymbol{w}$，求 $\partial_k g_i / \partial w_k$。提示：先把 $g_i$ 写成分量求和的形式，再对单个 $w_k$ 求导。结果下周线性回归会直接用到。
- **Ex 15**：$F(s) = \log(1+e^s)$。提示：用链式法则，求出导数后和后面分类里的 logistic sigmoid 比较一下。

## 跨周联系

- 矩阵向量乘法、$\nabla \|\boldsymbol{X}\theta - \boldsymbol{y}\|^2$ → [FOAI-W02-线性回归](FOAI-公开版-W02-线性回归.md)。
- 链式法则 → Week 5 反向传播、Week 6+ 神经网络。
- 负对数概率 → 之后分类任务的交叉熵损失。
