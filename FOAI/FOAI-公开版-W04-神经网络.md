---
title: FOAI W04｜神经网络
course: FOAI
week: 4
type: lecture
tags:
  - AI-MSc
  - TB1
  - FOAI
  - 神经网络
  - ReLU
  - 前向传播
  - 梯度下降
source: 课程材料（Conor Houghton、Roberto Colomboni）的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: published
publication: public
updated: 2026-10-09
---

# FOAI W04｜神经网络（Neural Networks）

> [!IMPORTANT]
> **非官方学习笔记**：这是我在 University of Bristol「Foundations of Artificial Intelligence（SEMTM0051, 2026/27）」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：Conor Houghton、Roberto Colomboni。原始材料：[semtm0051/2026_27](https://github.com/semtm0051/2026_27/tree/322787dec91f174303ccf90399bb6243a060eb7d/)。经两位讲师同意公开；不含练习解答与非公开材料（如课堂录像）。

> [!NOTE]
> **资料范围与可靠度：high**
> 一手讲义：[`Week_04/Notes/Week_04 - Neural Networks.pdf`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_04/Notes/Week_04%20-%20Neural%20Networks.pdf)（24 页，version of 28 September 2026）。
> 配套一手 Slides：`Week_04/Slides/Week_04 - Slides.pdf`（44 页，Lecture 4）。**公式编号 (n) 沿用讲义**，便于对照。
> 本笔记没有逐页标注 Slides：讲义已覆盖 Slides 的全部内容。标注为「补充说明」的方框是我为了方便入门加的解释，**不是讲义原文**。

> [!TIP]
> **基础薄弱怎么读这篇**
> 1. 先读 **§0（全貌）**，只记住「新东西只有一个：特征函数里也带参数」。
> 2. 然后直接看 **§3.4 的数值前向传播**，用手算一遍。这是整篇最重要的一页。
> 3. 再读 §5（tent 与 XOR）理解「为什么要隐藏层」，最后读 §6（拟合）。
> 4. §1、§2 是铺垫，§7、§8 是补充；术语不熟时翻最后的 **名词速查**。

> [!TIP]
> **数值核对（2026-10-09）**
> 讲义和 Slides 里能算的数字，我都用独立代码重算过，全部一致：前向传播例子的输出 $-7/2$、tent 函数各区间的值、XOR 网络四个分数与预测、§6 的分段损失 (44) 与两个极小点（损失 $0$ 和 $8$；$0.627$ 和 $2.627$）、式 (46) 的导数（与数值微分最大误差约 $3\times10^{-8}$）、Slides 第 40–41 页的梯度下降迭代值（$\theta^{(t)}$）与平方损失、参数个数 81。（logistic loss 沿路径的数值是我自己复算的，Slides 文字里没有逐项核对。）没有发现讲义与 Slides 的数值冲突。

> [!WARNING]
> **来源冲突｜向量记号（2026-10-09）**
> 本讲讲义与 Slides 都把输入写成**行向量** $\mathbb R^{1\times d}$。而 W03 的 Slides 在 2026-10-08 更新后改成了 $\mathbb R^{d}$（W03 讲义仍是 $\mathbb R^{1\times d}$）。含义相同，只是写法不同；见 [FOAI-W03-线性分类](FOAI-公开版-W03-线性分类.md) 的来源冲突标注。本篇一律用讲义的行向量写法。

## 核心脉络

**以前**：先挑好特征函数（如 $x,\ x^2$），数据只用来决定「怎么组合这些特征」的系数。
**现在**：让特征函数自己也带参数，数据同时决定特征长什么样、以及怎么组合它们。做法是 **仿射变换 → ReLU → 仿射变换 → ReLU → … → 输出层**。后半段（输出怎么变概率、用什么损失、怎么梯度下降）和 W02、W03 完全一样。

---

## 0. 全貌：这一讲到底新增了什么

### 0.1 一张对照表

| | W02 线性回归 | W03 线性分类 | W04 神经网络 |
| --- | --- | --- | --- |
| 预测函数 | $f_{\boldsymbol\theta}(\boldsymbol x)=b+\boldsymbol x\boldsymbol w$（可加固定的非线性特征） | 同左，输出当作**打分** | 多层：仿射 + ReLU 叠起来 |
| 特征 | 事先固定 | 事先固定 | **带参数，随数据调整** |
| 输出怎么用 | 直接当数值预测 | sigmoid / softmax 变概率，再取标签 | **同左，一点没变** |
| 损失 | 平方损失 | logistic loss / 交叉熵 | **同左，一点没变** |
| 怎么拟合 | 梯度下降 | 梯度下降 | 梯度下降（但可能卡在局部极小，见 §6） |

> [!TIP]
> **补充说明（非讲义原文）：一句话记忆**
> 神经网络 = 「会自己学形状的特征」 + 「W02/W03 学过的那套输出与损失」。如果 W03 的 sigmoid、softmax、logistic loss 还不熟，先回去看 [FOAI-W03-线性分类](FOAI-公开版-W03-线性分类.md)，本讲不再重复推导它们。

### 0.2 一次预测的全流程（本讲的主线）

输入 $\boldsymbol x$ → 第 1 层：仿射 $\boldsymbol x\boldsymbol W_1+\boldsymbol b_1$，再 ReLU → 第 2 层：同理 → … → 输出层：仿射（不加 ReLU）→ 得到**数值**（回归）或**打分**（分类）→ 分类时再 sigmoid / softmax 变概率 → 与真实标签算损失 → 梯度下降更新所有 $\boldsymbol W_r,\boldsymbol b_r$。

### 0.3 需要先会的几个小工具

> [!TIP]
> **补充说明（非讲义原文）：入门工具箱**
> - **行向量**：一行数，如 $\boldsymbol x=(1,2)$，形状 $1\times 2$。本讲所有「一个样本」都写成行向量。
> - **矩阵乘法的形状规则**：$(1\times n)\cdot(n\times m)=(1\times m)$。中间的 $n$ 必须相等，会「消掉」。看不懂公式时先检查形状。
> - **仿射（affine）函数**：线性部分加一个常数项，如 $y=2x+1$。图像是直线（多维时是平面）。
> - **梯度（gradient）**：把损失对每个参数的偏导排成一列。梯度下降就是沿着「下降最快的反方向」小步更新参数：$\boldsymbol\theta\leftarrow\boldsymbol\theta-\eta\nabla\mathcal L$，$\eta$ 是学习率。
> - **sigmoid**：$\sigma(s)=1/(1+e^{-s})$，把任何打分压到 $(0,1)$，当作「属于类 1 的概率」。

---

## 1. 为什么要改预测函数，同时保持学习问题不变（§1）

### 1.1 保持不变的部分

回归与分类里我们用特征和系数组合出预测。只用原始坐标时，一个输出是仿射形式

$$f_{\boldsymbol\theta}(\boldsymbol x)=\boldsymbol x\boldsymbol w+b,\qquad \boldsymbol\theta=\begin{pmatrix}b\\\boldsymbol w\end{pmatrix}.\tag{1}$$

输出怎么用，取决于任务，且**用神经网络后一律不变**：

| 任务 | 网络输出怎么解释 | 损失 | 预测规则 |
| --- | --- | --- | --- |
| 回归 | 直接当数值预测 | 平方损失 | — |
| 二分类 | 当**打分**（logit），sigmoid 变成类 1 的概率 | logistic loss | plug-in 规则（概率 $>1/2$ 判类 1） |
| 多分类 | 每类一个打分，softmax 变概率 | 交叉熵 | 取概率最大的类 |

### 1.2 从固定特征到学出来的特征

W02 里我们用过 $x\mapsto b+w_1x+w_2x^2$。特征函数 $x$ 和 $x^2$ 是**我们事先选的**；拟合只能调 $b,w_1,w_2$，改变不了「用的是 $x$ 和 $x^2$」这件事。

现在我们让特征函数**自带权重和截距**，并和最终的组合系数**一起拟合**。这就是「网络在学特征」的含义：

- 我们仍然选择：用哪些运算、中间算几个值（即**架构**）。
- 数据决定：运算里面的具体参数。

---

## 2. 仿射变换的复合（§2）

### 2.1 一次仿射变换

输入是行向量 $\boldsymbol u\in\mathbb R^{1\times n}$，想算 $m$ 个输出：

$$\boldsymbol a=\boldsymbol u\boldsymbol W+\boldsymbol b,\quad \boldsymbol W\in\mathbb R^{n\times m},\ \boldsymbol b\in\mathbb R^{1\times m}.\tag{3}$$

第 $j$ 个坐标：

$$a_j=\sum_{k=1}^{n}u_kW_{kj}+b_j,\qquad j=1,\dots,m.\tag{4}$$

- $W_{kj}$：第 $k$ 个输入坐标，用在第 $j$ 个输出上的权重。**第一个下标是输入，第二个下标是输出**。
- $\boldsymbol W$ 的**每一列**对应一个输出；每个输出有自己的权重和偏置（bias，就是截距），但都从同一个输入行算出。
- 若只要一个输出，$\boldsymbol W$ 只有一列，就退化成 W02 的权重向量 $\boldsymbol w$。

两个输入、三个输出时：

$$\boldsymbol W=\begin{pmatrix}W_{11}&W_{12}&W_{13}\\W_{21}&W_{22}&W_{23}\end{pmatrix},\ \ \boldsymbol b=(b_1,b_2,b_3),\qquad a_j=u_1W_{1j}+u_2W_{2j}+b_j.$$

### 2.2 叠两层仿射，仍然只是仿射

先看一维的小例子：$u(x)=2x+1$，$v(z)=3z-4$。先 $u$ 再 $v$：

$$(v\circ u)(x)=3(2x+1)-4=6x-1.\tag{7}$$

斜率和截距变了，**但还是一条直线**。Slides 第 6 页也用的这个例子。

一般情形：

> **引理 1（仿射映射的复合）**：两个仿射映射的复合仍是仿射映射。

证明思路：设 $f(\boldsymbol x)=\boldsymbol x\boldsymbol W_1+\boldsymbol b_1$，$g(\boldsymbol z)=\boldsymbol z\boldsymbol W_2+\boldsymbol b_2$，代入并展开：

$$(g\circ f)(\boldsymbol x)=(\boldsymbol x\boldsymbol W_1+\boldsymbol b_1)\boldsymbol W_2+\boldsymbol b_2=\boldsymbol x\underbrace{(\boldsymbol W_1\boldsymbol W_2)}_{=:\boldsymbol W_{\rm eff}}+\underbrace{(\boldsymbol b_1\boldsymbol W_2+\boldsymbol b_2)}_{=:\boldsymbol b_{\rm eff}}.\tag{8}$$

参数固定后，$\boldsymbol W_{\rm eff}$ 和 $\boldsymbol b_{\rm eff}$ 就是一个固定矩阵和一个固定行，所以仍是仿射。重复使用这个结论，**任意有限多层仿射变换都能压缩成一层**。

**后果**：只叠仿射层造不出非仿射的特征；若把输出当二分类打分，决策边界仍是直线。要突破，必须在层与层之间插入**非线性运算**。

---

## 3. 神经网络与前向传播（§3）

### 3.1 ReLU：本讲唯一的非线性

标量版：

$$\mathrm{ReLU}(a)=\max\{0,a\}=\begin{cases}0,&a\le 0,\\a,&a>0.\end{cases}\tag{9}$$

正数原样通过，负数变成 0，所以输出一定非负；$\mathrm{ReLU}(0)=0$。图像在 0 处有一个折角，该点**不可导**（§6 会讲怎么处理）。

向量版：对每个坐标分别做（**坐标逐个作用**，形状不变）：

$$\mathrm{ReLU}(\boldsymbol a)=(\mathrm{ReLU}(a_1),\dots,\mathrm{ReLU}(a_m)),\quad \boldsymbol a\in\mathbb R^{1\times m}.\tag{10}$$

例：$\mathrm{ReLU}(-2,\ 0,\ 3)=(0,\ 0,\ 3)$。

> [!NOTE]
> **记号**
> 讲义用粗体 $\mathbf{ReLU}$ 表示向量版、普通体 $\mathrm{ReLU}$ 表示标量版（与 W03 的 sigmoid 同样处理）。本笔记里按上下文区分，不再区别字体。

### 3.2 怎么读一张神经网络图

- 一个 **单元（unit）**，也叫 **神经元（neuron）**：先对输入做加权求和，加偏置，再套 ReLU：$\mathrm{ReLU}(w_1x_1+w_2x_2+b)$。它给每个输入行一个数，可当作一个新特征。
- 把多个单元（各自有各自的权重和偏置）并排放，叫一个 **层（layer）**。五个单元就得到五个特征值。
- 这五个值可以当作下一层的输入，再做加权求和、加偏置、ReLU，得到更多特征。
- 产生中间特征的层叫 **隐藏层（hidden layer）**；最后一层叫 **输出层（output layer）**。
- 一层里单元的个数叫 **宽度（width）**；层数（**不算输入**）叫 **深度（depth）**。
- 「神经元」「神经网络」的名字来自大脑的启发，但讲义强调：这只是一个很松散的类比，生物神经元要复杂得多。

讲义的 Figure 2：输入 2 个坐标，两层隐藏层（宽度 5 和 7），输出层 3 个神经元，记作架构 $2\to5\to7\to3$。**输出层不加 ReLU**，三个仿射组合直接就是输出。从输入到输出依次计算，叫 **前向传播（forward pass）**。

### 3.3 只有一个隐藏层的网络

输入行有 $d$ 个坐标，隐藏层宽度 $m$，输出层宽度 $q$。参数形状：

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| $\boldsymbol W_1$ | $d\times m$ | 每列是一个隐藏单元对 $d$ 个输入的权重 |
| $\boldsymbol b_1$ | $1\times m$ | 每个隐藏单元的偏置 |
| $\boldsymbol W_2$ | $m\times q$ | 每列是一个输出单元对 $m$ 个隐藏值的权重 |
| $\boldsymbol b_2$ | $1\times q$ | 每个输出单元的偏置 |

所有可调数放进一个参数列 $\boldsymbol\theta$（顺序固定）。前向传播三步：

$$\boldsymbol a_1(\boldsymbol x)=\boldsymbol x\boldsymbol W_1+\boldsymbol b_1\ \ \text{（预激活 preactivation）}\tag{11}$$
$$\boldsymbol\phi_1(\boldsymbol x)=\mathrm{ReLU}(\boldsymbol a_1(\boldsymbol x))\ \ \text{（激活 activation，也就是特征）}\tag{12}$$
$$\boldsymbol f_{\boldsymbol\theta}(\boldsymbol x)=\boldsymbol\phi_1(\boldsymbol x)\boldsymbol W_2+\boldsymbol b_2\ \ \text{（输出）}\tag{13}$$

合起来：

$$\boldsymbol f_{\boldsymbol\theta}(\boldsymbol x)=\mathrm{ReLU}(\boldsymbol x\boldsymbol W_1+\boldsymbol b_1)\boldsymbol W_2+\boldsymbol b_2.\tag{14}$$

- 为什么叫 $\boldsymbol\phi$：它扮演的角色就是 W02 的特征映射 $\phi$（所以讲义不用神经网络文献里常见的 $h$，也避免和 W03 的分类器 $h$ 冲突）。
- 与引理 1 的区别：**中间隔着 ReLU，$\boldsymbol W_1$ 与 $\boldsymbol W_2$ 无法合并成一个矩阵**，所以网络能表示非仿射函数。个别参数选法仍可退化成仿射：例如 $\boldsymbol W_2=\boldsymbol 0$ 时输出就是常数行 $\boldsymbol b_2$。
- 第 $j$ 个隐藏特征写开：

$$[\boldsymbol\phi_1(\boldsymbol x)]_j=\mathrm{ReLU}\Big(\sum_{k=1}^{d}x_k(\boldsymbol W_1)_{kj}+(\boldsymbol b_1)_j\Big).\tag{15}$$

**什么叫「学到特征」**：若固定 $\boldsymbol W_1,\boldsymbol b_1$，只拟合 $\boldsymbol W_2,\boldsymbol b_2$，那就是对**固定特征**做仿射组合（W02 的做法）。若连 $\boldsymbol W_1,\boldsymbol b_1$ 也拟合，特征函数本身就变了。

**特征函数怎么变**：一个两输入神经元 $\phi(\boldsymbol x)=\mathrm{ReLU}(w_1x_1+w_2x_2+b)$，当 $(w_1,w_2)\neq(0,0)$ 时，直线 $w_1x_1+w_2x_2+b=0$ 把平面分成两块：一侧特征值为 0（**不激活**），另一侧为正（**激活**）。固定权重改偏置，这条线**平移**；改权重，线的方向和激活一侧的变化快慢随之变。（一维时，这条线退化成断点 $x=-b/w$。）

例（讲义式 (16)）：把 $(w_1,w_2,b)$ 从 $(1,-1,0)$ 改成 $(1,1,-1)$，特征就从 $\mathrm{ReLU}(x_1-x_2)$ 变成 $\mathrm{ReLU}(x_1+x_2-1)$。同一个输入 $\boldsymbol x=(1,2)$：

| 参数 | 特征 | 在 $(1,2)$ 处的值 |
| --- | --- | --- |
| $(1,-1,0)$ | $\mathrm{ReLU}(x_1-x_2)$ | $\mathrm{ReLU}(1-2)=0$ |
| $(1,1,-1)$ | $\mathrm{ReLU}(x_1+x_2-1)$ | $\mathrm{ReLU}(1+2-1)=2$ |

输入没变，特征值因为参数变了而变了。图中的深色线只是「激活/不激活」的分界，**不是分类决策边界**。

### 3.4 一个完整的数值前向传播（务必手算一遍）

设 $d=2,\ m=3,\ q=1$（单个输出，当标量处理）。参数：

$$\boldsymbol W_1=\begin{pmatrix}1&2&-1\\-1&0&1\end{pmatrix},\quad \boldsymbol b_1=(0,\ 1,\ 0),\quad \boldsymbol W_2=\begin{pmatrix}2\\-1\\1/2\end{pmatrix},\quad b_2=-1.\tag{17}$$

输入 $\boldsymbol x=(1,2)$。

**第 1 步：预激活 $\boldsymbol a_1=\boldsymbol x\boldsymbol W_1+\boldsymbol b_1$**，用 $\boldsymbol W_1$ 的第 $j$ 列：

| 隐藏单元 | 计算 | 预激活 |
| --- | --- | --- |
| 1 | $1\cdot1+2\cdot(-1)+0$ | $-1$ |
| 2 | $1\cdot2+2\cdot0+1$ | $3$ |
| 3 | $1\cdot(-1)+2\cdot1+0$ | $1$ |

即 $\boldsymbol a_1(\boldsymbol x)=(-1,\ 3,\ 1)$。

**第 2 步：ReLU**：负的变 0，其余不变 → $\boldsymbol\phi_1(\boldsymbol x)=(0,\ 3,\ 1)$。

**第 3 步：输出层**：把三个激活按 $\boldsymbol W_2$ 加权求和，再加 $b_2$：

$$s=f_{\boldsymbol\theta}(\boldsymbol x)=0\cdot2+3\cdot(-1)+1\cdot\tfrac12-1=-\tfrac72.\tag{20}$$

隐藏激活都非负，输出却是负的：因为输出层的权重和偏置**没有符号限制**，并且输出层后面没有 ReLU。Slides 第 10–13 页是这个例子的逐步动画。

### 3.5 多个隐藏层

总层数（含输出层）记 $L\ge2$，也就是深度。第 $r$ 层宽度 $d_r$，令 $d_0=d$，$d_L=q$。参数形状：

$$\boldsymbol W_r\in\mathbb R^{d_{r-1}\times d_r},\quad \boldsymbol b_r\in\mathbb R^{1\times d_r},\quad r=1,\dots,L.$$

令 $\boldsymbol\phi_0(\boldsymbol x)=\boldsymbol x$。隐藏层 $r=1,\dots,L-1$ 反复做「仿射 + ReLU」：

$$\boldsymbol a_r(\boldsymbol x)=\boldsymbol\phi_{r-1}(\boldsymbol x)\boldsymbol W_r+\boldsymbol b_r,\qquad \boldsymbol\phi_r(\boldsymbol x)=\mathrm{ReLU}(\boldsymbol a_r(\boldsymbol x)).\tag{21}$$

输出层只做仿射，不做 ReLU：

$$\boldsymbol a_L(\boldsymbol x)=\boldsymbol f_{\boldsymbol\theta}(\boldsymbol x)=\boldsymbol\phi_{L-1}(\boldsymbol x)\boldsymbol W_L+\boldsymbol b_L.\tag{22}$$

**参数个数**：第 $r$ 层有 $d_{r-1}d_r$ 个权重和 $d_r$ 个偏置，所以

$$P=\sum_{r=1}^{L}\big(d_{r-1}d_r+d_r\big).\tag{24}$$

例：架构 $2\to5\to7\to3$（$L=3$）：

| 层 $r$ | 权重矩阵 | 偏置行 | 参数个数 |
| --- | --- | --- | --- |
| 1 | $2\times5$ | $1\times5$ | $10+5=15$ |
| 2 | $5\times7$ | $1\times7$ | $35+7=42$ |
| 3 | $7\times3$ | $1\times3$ | $21+3=24$ |

$P=15+42+24=81$，即 $\boldsymbol\theta$ 有 81 个分量（Slides 第 21 页）。

---

## 4. 网络的输出怎么用（§4）

输出层要有几个单元，取决于任务（输出层宽度 $d_L$）：

| 任务 | 输出层单元数 | 输出的角色 | 损失（对数据集求和） | 预测 |
| --- | --- | --- | --- | --- |
| 回归 | $d_L=1$ | 直接是数值预测 | $\mathcal L_{\rm sq}(\boldsymbol\theta)=\sum_i\big(f_{\boldsymbol\theta}(\boldsymbol x_i)-y_i\big)^2$ (26) | 就是输出 |
| 二分类 | $d_L=1$ | 打分（logit），$p_{\boldsymbol\theta}(\boldsymbol x)=\sigma(f_{\boldsymbol\theta}(\boldsymbol x))$ (27) | $\mathcal L_{\rm log}=\sum_i\ell_{\rm log}(p_{\boldsymbol\theta}(\boldsymbol x_i),y_i)$ (28) | $h(\boldsymbol x)=\mathrm{step}(p-\tfrac12)=\mathrm{step}(f_{\boldsymbol\theta}(\boldsymbol x))$ (29) |
| $K$ 类分类 | $d_L=K$ | $K$ 个打分，$\boldsymbol p=\mathrm{softmax}(\boldsymbol f)$ (30) | $\mathcal L_{\rm multi}=\sum_i\ell_{\rm multi}(\boldsymbol p(\boldsymbol x_i),y_i)$ (31) | 取概率最大的类 (32) |

**平局规则（沿用 W03 更新后的版本）**：二分类里，打分为 0 时判**类 0**（打分为正才判类 1）；多分类里，若有并列最大，取**最小下标**：

$$h_{\boldsymbol p_{\boldsymbol\theta}}(\boldsymbol x)=\min\ \mathop{\arg\max}_{k\in\{1,\dots,K\}}\ p_{\boldsymbol\theta,k}(\boldsymbol x).\tag{32}$$

Slides 第 22、23 页写法相同。

---

## 5. 隐藏层能表示什么（§5）

讲义用两个小构造回答：①造一个三角形函数；②分开 XOR 的四个点（任何仿射打分都做不到）。

### 5.1 三个 ReLU 搭出一个「帐篷」（tent）函数

目标：$x\le0$ 时为 0，在 $x=1$ 升到高度 1，在 $x=2$ 回到 0，之后保持 0。思路是**在断点处改斜率**：

1. $\mathrm{ReLU}(x)$：在 $x=0$ 处斜率从 0 变成 $+1$。
2. 在 $x=1$ 处要让斜率从 $+1$ 变成 $-1$，即改变 $-2$：减去 $2\,\mathrm{ReLU}(x-1)$。
3. 在 $x=2$ 处要让斜率从 $-1$ 回到 0，即改变 $+1$：加上 $\mathrm{ReLU}(x-2)$。

$$T(x)=\mathrm{ReLU}(x)-2\,\mathrm{ReLU}(x-1)+\mathrm{ReLU}(x-2).\tag{33}$$

逐区间核对（哪些 ReLU 是激活的）：

| 区间 | 计算 | 结果 |
| --- | --- | --- |
| $x\le0$ | $0-0+0$ | $0$ |
| $0<x\le1$ | $x-0+0$ | $x$ |
| $1<x<2$ | $x-2(x-1)+0$ | $2-x$ |
| $x\ge2$ | $x-2(x-1)+(x-2)$ | $0$ |

在每个断点相邻两段表达式取值相等，所以 $T$ 是连续的折线。作为网络：一个输入、三个隐藏单元、一个仿射输出，参数为（Slides 第 29 页）

$$\boldsymbol W=(1,1,1),\quad \boldsymbol b=(0,-1,-2),\quad \boldsymbol W_{\rm out}=(1,-2,1)^\top,\quad b_{\rm out}=0.$$

「偏置决定断点位置，输出权重决定斜率变化」。这些参数在实际中**可以被拟合出来，而不是手工指定**。

### 5.2 一个能分开 XOR 的网络

四个输入与标签（XOR：恰好一个坐标是 1 时标签为 1）：

| $\boldsymbol x$ | $(0,0)$ | $(1,0)$ | $(0,1)$ | $(1,1)$ |
| --- | --- | --- | --- | --- |
| $y$ | 0 | 1 | 1 | 0 |

**先证明任何仿射打分都不行**。打分 $f(\boldsymbol x)=w_1x_1+w_2x_2+b$，规则「打分 $\le0$ 判类 0，$>0$ 判类 1」。要全对必须同时满足

$$b\le0,\quad w_1+w_2+b\le0,\quad w_1+b>0,\quad w_2+b>0.\tag{37}$$

前两式相加得 $w_1+w_2+2b\le0$；后两式相加得 $w_1+w_2+2b>0$。矛盾，所以**没有**任何 $b,w_1,w_2$ 能全对。

**再构造网络**。观察：标签为 1 的两个点，两个坐标**不相等**；标签为 0 的点，两个坐标**相等**。用两个隐藏单元检测「差的正负」：

$$f_{\boldsymbol\theta}(\boldsymbol x)=\mathrm{ReLU}(x_1-x_2)+\mathrm{ReLU}(x_2-x_1)-\tfrac12.\tag{38}$$

参数：$\boldsymbol W_1=\begin{pmatrix}1&-1\\-1&1\end{pmatrix},\ \boldsymbol b_1=(0,0),\ \boldsymbol W_2=\begin{pmatrix}1\\1\end{pmatrix},\ b_2=-\tfrac12$。

| 输入 | $\mathrm{ReLU}(x_1-x_2)$ | $\mathrm{ReLU}(x_2-x_1)$ | 打分 | 预测 |
| --- | --- | --- | --- | --- |
| $(0,0)$ | 0 | 0 | $-1/2$ | 0 |
| $(1,0)$ | 1 | 0 | $+1/2$ | 1 |
| $(0,1)$ | 0 | 1 | $+1/2$ | 1 |
| $(1,1)$ | 0 | 0 | $-1/2$ | 0 |

四个全对。由于 $\mathrm{ReLU}(u)+\mathrm{ReLU}(-u)=\lvert u\rvert$，打分也可写成 $f_{\boldsymbol\theta}(\boldsymbol x)=\lvert x_1-x_2\rvert-\tfrac12$。

**为什么行得通**：隐藏层把输入换成特征 $\boldsymbol\phi_1(\boldsymbol x)$。$(0,0)$ 和 $(1,1)$ 都变成 $(0,0)$，另外两点变成 $(1,0)$ 和 $(0,1)$。在**特征空间**里，一条直线（$\phi_1+\phi_2=1/2$）就能把类 0 与类 1 分开；回到原输入空间，这条直线对应的是**非直线**的分类区域（中间一条带状区域 $\lvert x_1-x_2\rvert\le 1/2$ 判类 0，带外判类 1。打分为 0 的点判类 0）。Slides 第 30–33 页是图示。

**注意**：虽然四个标签全对，概率是 $\sigma(\mp1/2)$ 而不是 0 和 1，所以 logistic loss 仍为正（见练习 8）。

### 5.3 这两个例子说明了什么，又没说明什么

- 这两个构造都是**先变换输入，再用仿射输出层**。它们**不能**证明神经网络是唯一办法：tent 可以固定那三个 ReLU 特征，只拟合最后的系数；XOR 也可以加一个固定特征 $x_1x_2$，打分 $x_1+x_2-2x_1x_2-\tfrac12$ 在这四个点上符号恰好正确。
- 网络的关键是：特征本身**也带参数**，所以拟合能同时调特征和组合方式（比如调一个隐藏偏置就能移动 ReLU 的折点）。架构和激活函数仍由我们事先选。
- 上面的参数是**我们给出来的**。这只证明这些任务「存在合适的参数」，**不是**说梯度下降一定找得到。
- 训练损失小不等于对新数据预测好（泛化 generalisation）：欠拟合（没抓住规律）与过拟合（记住了训练数据细节）都可能。本讲与下讲只关注拟合过程。

---

## 6. 拟合：驻点与梯度下降（§6）

### 6.1 导数与参数更新

要拟合网络，就是在 §4 的某个经验损失 $\mathcal L$ 上找参数。可导的地方用和 W02、W03 相同的梯度下降：

$$\boldsymbol\theta^{(t+1)}=\boldsymbol\theta^{(t)}-\eta\,\nabla\mathcal L(\boldsymbol\theta^{(t)}),\quad \eta>0.\tag{40}$$

梯度的每个分量是对一个权重或偏置的偏导；计算要用链式法则穿过整个网络，所以需要 ReLU 的导数：

$$\mathrm{ReLU}'(a)=\begin{cases}0,&a<0,\\1,&a>0\end{cases}=\mathrm{step}(a),\quad a\ne0.\tag{41}$$

在 $a=0$ 处，左导数是 0、右导数是 1，**ReLU 不可导**。**计算上的约定**：更新时把 $\mathrm{step}(a)$ 当作 ReLU 对应的乘子，且 $\mathrm{step}(0)=0$（PyTorch 就是这样做的）。这个约定**并不让 ReLU 在 0 处变得可导**，只是给出一个可用的数。

### 6.2 只有一个可调参数的网络

为了画出整个损失曲线，讲义用一个只有标量参数 $\theta$ 的网络：

$$f_\theta(x)=\mathrm{ReLU}(x-\theta)+\mathrm{ReLU}(\theta-x)-3=\lvert x-\theta\rvert-3.\tag{42}$$

两个隐藏偏置是 $-\theta$ 和 $+\theta$，同时变；$\theta$ 的作用是让 V 形图像的顶点 $(\theta,-3)$ **水平平移**。以下两个例子都用输入 $-1$ 和 $1$。

> [!TIP]
> **补充说明（非讲义原文）：全局极小与局部极小**
> - **全局极小**：所有参数中损失最小的点。
> - **局部极小**：在它附近，任何足够小的移动都不会让损失更小，但它**可能比别处的损失大**。
> - **驻点**：导数为 0 的点。梯度下降本来就是靠「导数为 0」停下来，所以它**可能停在局部极小**。

### 6.3 回归（平方损失）

两个观测 $(x_1,y_1)=(-1,-1)$，$(x_2,y_2)=(1,1)$：

$$\mathcal L_{\rm sq}(\theta)=\big(f_\theta(-1)+1\big)^2+\big(f_\theta(1)-1\big)^2.\tag{43}$$

代入 (42)，绝对值在 $\theta$ 经过两个输入点时换表达式，得分段二次函数：

$$\mathcal L_{\rm sq}(\theta)=\begin{cases}2(\theta+3)^2,&\theta\le-1,\\2(\theta+1)^2+8,&-1<\theta<1,\\2(\theta-3)^2+8,&\theta\ge1.\end{cases}\tag{44}$$

| 驻点 | 两个输入处的预测 | 平方损失 | 性质 |
| --- | --- | --- | --- |
| $\theta=-3$ | $-1,\ 1$（正好等于目标） | $0$ | **全局极小**（平方损失不会为负） |
| $\theta=+3$ | $1,\ -1$（符号相反） | $8$ | **局部极小**（附近任何小改变都使 $2(\theta-3)^2$ 变大） |

两点导数都为 0：$\mathcal L_{\rm sq}'(-3)=\mathcal L_{\rm sq}'(3)=0$，**但损失不同**。解「导数 $=0$」可能得到一个损失比全局极小大的局部极小。

### 6.4 二分类（logistic loss）

同样的输入，标签变成 $(-1,0)$ 和 $(1,1)$，网络输出当打分，$p_\theta(x)=\sigma(f_\theta(x))$：

$$\mathcal L_{\rm log}(\theta)=-\log\big(1-p_\theta(-1)\big)-\log p_\theta(1).\tag{45}$$

求导（与 W03 一样，单个损失对其打分的导数分别是 $p_\theta(-1)$ 和 $p_\theta(1)-1$，再乘打分对 $\theta$ 的导数）：

$$\mathcal L_{\rm log}'(\theta)=\begin{cases}-\sigma(-\theta-4)+\sigma(\theta+2),&\theta<-1,\\\sigma(\theta-2)+\sigma(\theta+2),&-1<\theta<1,\\\sigma(\theta-2)-\sigma(4-\theta),&\theta>1.\end{cases}\tag{46}$$

两侧的外区间里导数由负变正，分别在 $\theta=-3$ 与 $\theta=3$ 处为 0，所以两点都是局部极小；中间区间导数为正，损失在上升。

| $\theta$ | 打分 | 概率 $p(-1),\,p(1)$ | 预测对错 | 损失 |
| --- | --- | --- | --- | --- |
| $-3$ | $-1,\ 1$ | $\sigma(-1)\approx0.269,\ \sigma(1)\approx0.731$ | 两个都对（错 0/2） | $2\log(1+e^{-1})\approx0.627$ |
| $+3$ | $1,\ -1$ | $\sigma(1)\approx0.731,\ \sigma(-1)\approx0.269$ | 两个都错（错 2/2） | $2\log(1+e)\approx2.627$ |

所以 $\theta=-3$ 是全局极小，$\theta=3$ 是损失更大的局部极小。注意：**在 $\theta=3$ 这个导数为 0 的点，分类器对两个观测都分错**（Slides 第 39 页）。

### 6.5 梯度下降会停在哪

一个参数时，梯度下降就是 $\theta^{(t+1)}=\theta^{(t)}-\eta\mathcal L'(\theta^{(t)})$（47）。对上述两个目标，取 $\eta=0.2$：从 $\theta^{(0)}=-4$ 出发收敛到全局极小 $-3$；从 $\theta^{(0)}=4$ 出发收敛到局部极小 $3$。**两次运行的损失每一步都在下降，但收敛到不同的极限值。** 所以「每步损失都下降」**并不保证**收敛到全局极小。

Slides 第 40、41 页用 $\theta^{(0)}=\pm1.5$ 画了迭代过程。我复算的迭代值与 Slides 一致（下表中的 $-1.5$ 一侧为负，$+1.5$ 一侧符号相反）：

| 迭代 $t$ | 0 | 1 | 2 | 5 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- |
| 平方损失，$\eta=0.05$，$\theta^{(t)}$（$+1.5$ 一侧） | 1.5000 | 1.8000 | 2.0400 | 2.5085 | 2.8389 | 2.9827 |
| 对应 $\mathcal L_{\rm sq}$（$+1.5$ 一侧，趋向 8） | 12.5000 | 10.8800 | 9.8432 | 8.4832 | 8.0519 | 8.0006 |
| 对应 $\mathcal L_{\rm sq}$（$-1.5$ 一侧，趋向 0） | 4.5000 | 2.8800 | 1.8432 | 0.4832 | 0.0519 | 0.0006 |
| logistic loss，$\eta=0.5$，$\theta^{(t)}$（$+1.5$ 一侧） | 1.5000 | 1.7733 | 2.0028 | 2.4747 | 2.8233 | 2.9802 |
| 对应 $\mathcal L_{\rm log}$（$+1.5$ 一侧） | 3.0530 | 2.9154 | 2.8190 | 2.6805 | 2.6327 | 2.6266 |
| 对应 $\mathcal L_{\rm log}$（$-1.5$ 一侧） | 1.0530 | 0.9154 | 0.8190 | 0.6805 | 0.6327 | 0.6266 |

（$-1.5$ 一侧的 $\theta^{(t)}$ 是上表数值取负。两种损失在 $-1.5$ 一侧趋向全局极小，在 $+1.5$ 一侧趋向局部极小。）

> [!NOTE]
> **这张表的来源**
> 讲义（Figure 8）和 Slides（第 40、41 页）给出了 $\theta^{(t)}$ 和平方损失的数值；logistic loss 的数值是我按式 (45) 和上面的 $\theta^{(t)}$ 复算的，没有直接取自 Slides 文字。

---

## 7. 实践中怎么找到好解（§7）

局部极小问题在实践中没有万能解，这些都只是**启发式**策略（heuristic）：

1. **多换几个初始值**：从不同随机初始化重复训练，比较训练损失。增加计算成本，也不保证某次运行到达全局极小。
2. **改变更新的算法**：**小批量随机梯度下降**（mini-batch SGD）每次只用一小撮随机抽取的观测算梯度。在整体损失的局部极小点，单个观测的梯度可能不为零但互相抵消；一个随机小批量未必抵消，所以 SGD 可能继续移动，随机波动有时能帮助跳出差的局部极小。**但这不保证**：例如起点不是全局极小，却恰好每个观测各自的梯度都为 0，那么每个小批量梯度也为 0，SGD 完全不动。
3. **换架构**：更多隐藏神经元提供更多可调方式，更宽的网络有时更好优化、表达力也更强。但增加参数不会自动消除差的局部极小。

有些理论条件下这些策略可以被证明有效，但依赖具体假设；总体上还没有通用的理论解释它们何时有效，所以效果要靠实验检验，而不是假定成立。

---

## 8. 实验室里怎么求导（§8）

参数很多时，需要对每个权重和偏置求偏导。实验室里由 PyTorch 的**自动求导**（automatic differentiation）完成：标明哪些参数需要导数，算网络输出和损失，再让 PyTorch 对损失求导，用得到的导数更新参数。Lecture 5 会讲这背后的算法 **反向传播（backpropagation）**：用链式法则并复用中间结果。W04 的 Lab 还没用到自动求导，见 [FOAI-W04-PyTorch线性分类实验](FOAI-公开版-W04-PyTorch线性分类实验.md)。

---

## 名词速查

| 术语 | 含义 |
| --- | --- |
| 仿射（affine） | 线性 + 常数项，如 $uW+b$ |
| 偏置（bias） | 仿射里的常数项（截距），也是要拟合的参数 |
| 单元 / 神经元（unit / neuron） | 加权求和 + 偏置 + 激活函数，输出一个数 |
| 层（layer） | 并排的一组单元 |
| 隐藏层 / 输出层 | 产生中间特征的层 / 最后一层 |
| 宽度 / 深度 | 一层的单元数 / 层数（含输出层、不含输入） |
| 架构（architecture） | 层数、各层宽度、用什么激活函数（如 $2\to5\to7\to3$） |
| 预激活 $\boldsymbol a_r$ | 仿射结果，ReLU 之前 |
| 激活 $\boldsymbol\phi_r$ | ReLU 之后的值，即特征 |
| ReLU | $\max\{0,a\}$ |
| 前向传播（forward pass） | 从输入到输出依次计算 |
| 打分 / logit | 分类网络输出层未变成概率前的数 |
| 局部极小 | 附近最小，但可能比别处大 |
| 全局极小 | 所有参数中损失最小 |
| 小批量 SGD | 每次只用一小撮随机观测算梯度的梯度下降 |
| 泛化（generalisation） | 对没见过的新数据预测得好不好 |

## 学习提示（易错点）

- **形状是最好的检查**：$(1\times d)(d\times m)=(1\times m)$。每写一个公式先对形状。$\boldsymbol W$ 的**列数**是这一层的单元数。
- **ReLU 是坐标逐个作用**，不改变形状；输出层**不加** ReLU。
- **层间没有 ReLU 就等于一层仿射**（引理 1）。「深」本身不带来表达力，是非线性带来的。
- **预激活 $\boldsymbol a$ 与激活 $\boldsymbol\phi$ 别混**：先算 $\boldsymbol a=\boldsymbol\phi_{r-1}\boldsymbol W_r+\boldsymbol b_r$，再 $\boldsymbol\phi_r=\mathrm{ReLU}(\boldsymbol a)$。
- **导数为 0 不等于最优**：§6 的 $\theta=3$ 导数为 0，损失却更大，分类器还全错。
- **「每步损失都下降」≠「收敛到全局最优」**。
- **存在合适参数 ≠ 梯度下降能找到**。
- **标签全对 ≠ 损失为零**：XOR 例子里概率是 $\sigma(\pm1/2)$，logistic loss 仍为正。

## 练习要点（只给思路，不给解答）

官方没有发布解答。以下只提示**用哪段内容、先想什么**，请自己完成后再核对。

1. **维度与参数**：输入 $1\times3$，隐藏层宽 4，输出 2 个 logit，两层都有偏置。回看 §3.3 的形状表和 (24)。(b) 先数网络的参数，再数「从同样输入到同样输出的单个仿射映射」的参数，比较。
2. **两层仿射**：(a) 用引理 1 / (8) 求 $\boldsymbol W_{\rm eff},\boldsymbol b_{\rm eff}$，并分别用「分两步」和「合并公式」在 $\boldsymbol x=(0,1)$ 算，两个结果应一致。(b) 在两步之间插入 ReLU 再算。(c) 取 $\boldsymbol x=(t,0)$，写出分段函数，看各段斜率是否相同，以此判断是否仿射。
3. **新输入的前向传播**：用 (17) 的参数，把输入换成 $(2,0)$，按 §3.4 的三步走。(b) 当回归预测算平方损失；(c) 当二分类 logit 算概率、预测标签（打分为 0 判类 0）和 logistic loss。
4. **固定特征 vs 学出来的特征**：(b) 在 $\boldsymbol W_1,\boldsymbol b_1$ 固定时，看两个输入的隐藏特征是否**相同**，再想「不同的目标输出能否由同一个特征行得到」。(c) 改 $\boldsymbol b_1$ 的第二个分量后重新算特征，再找 $\boldsymbol W_2,\boldsymbol b_2$。
5. **ReLU 在 0 处**：用差商分别算左、右导数，再判断普通导数是否存在。
6. **两个初始值的梯度下降**：回看 (44) 的分段表达式和 §6.5。(a) 用导数符号判断更新方向；(b) 在每个区间里用对应的表达式迭代，用这两次运行检验「每步损失下降就一定收敛到全局极小」是否成立。
7. **用 ReLU 构造函数**：$U(x)=2\,\mathrm{ReLU}(x+1)-4\,\mathrm{ReLU}(x)+2\,\mathrm{ReLU}(x-1)$。(a) 仿照 §5.1 的「逐区间」表格；(b) 对照 Slides 第 29 页 tent 的参数写法，找出 $\boldsymbol W,\boldsymbol b,\boldsymbol W_{\rm out},b_{\rm out}$。
8. **正确标签、概率与损失**：回看 §5.2 末尾。把整个打分乘以 $\alpha>0$ 后，预测标签变不变？概率怎么变？总损失能否在有限的 $\alpha$ 下为零，$\alpha\to\infty$ 时趋向什么？

## 自测

1. 为什么只叠仿射层造不出非线性函数？用一维例子 $u(x)=2x+1,\ v(z)=3z-4$ 说明。
2. 写出一隐藏层网络 (11)–(13) 的三步，并说出每个量的形状。
3. 架构 $2\to5\to7\to3$ 有多少个参数？分层算出来。
4. $\mathrm{ReLU}$ 作用在行 $(-2,0,3)$ 上得到什么？为什么说它是「坐标逐个作用」？
5. 回归、二分类、多分类，网络的输出分别怎么解释？各用什么损失？打分为 0 时判哪一类？
6. 为什么 XOR 不能用仿射打分解决？网络 (38) 是怎样在特征空间里把它变成线性可分的？
7. §6 的例子里 $\theta=3$ 为什么导数为 0 还不是最优？这对「损失在下降」意味着什么？
8. 说出两种缓解局部极小的启发式策略，并说明为什么它们都不是保证。
