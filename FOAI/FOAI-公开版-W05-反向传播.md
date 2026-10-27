---
title: FOAI W05｜反向传播（Backpropagation）
course: FOAI
week: 5
type: lecture
tags:
  - AI-MSc
  - TB1
  - FOAI
  - 反向传播
  - backpropagation
  - 链式法则
  - 计算图
  - autograd
  - 梯度下降
source: 课程材料的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: published
publication: public
created: 2026-10-25
updated: 2026-10-27
---

# FOAI W05｜反向传播（Backpropagation）

> [!IMPORTANT]
> **非官方学习笔记**：这是我在「Foundations of Artificial Intelligence」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：课程讲师。原始材料：[官方课程仓库](https://github.com/semtm0051/2026_27/tree/12af5a9ea0650b35590f2f0f9f9bb6785a7593d6)。经讲师同意公开；不含练习解答与非公开材料（如课堂录像）。

> [!NOTE] 资料范围与可靠度：high
> 一手讲义：[`Week_05/Notes/Week_05 - Backpropagation.pdf`](https://github.com/semtm0051/2026_27/blob/12af5a9ea0650b35590f2f0f9f9bb6785a7593d6/Week_05/Notes/Week_05%20-%20Backpropagation.pdf)（课程讲师）。
> 配套一手 Slides：[`Week_05/Slides/Week_05 - Slides.pdf`](https://github.com/semtm0051/2026_27/blob/12af5a9ea0650b35590f2f0f9f9bb6785a7593d6/Week_05/Slides/Week_05%20-%20Slides.pdf)（Lecture 5）。
> 配套一手 Lab：[`Week_05/Lab/Week_05_nns_in_pytorch.ipynb`](https://github.com/semtm0051/2026_27/blob/12af5a9ea0650b35590f2f0f9f9bb6785a7593d6/Week_05/Lab/Week_05_nns_in_pytorch.ipynb)。
> 公式编号 (n) 沿用讲义，便于对照。标注为「补充说明」的部分是我为入门加的解释，不是讲义原文。

> [!TIP] 基础薄弱怎么读这篇
> 1. 先读 **§0 全貌**，只记住一句话：反向传播 = 把同一个中间导数算一次、到处复用。
> 2. 然后跟着 **§1 标量网络** 手算一遍前向 + 反向（这是整篇最核心的部分）。
> 3. 再读 **§2 向量版本**，重点是两张「通用公式」和维度检查；§2.5 用 W04 的网络把向量版本完整演算一遍。
> 4. **§3 三种损失的初始行**、**§4 数据集汇总** 是记忆型内容；§5 效率是理解型内容；Lab 部分另见配套实验笔记。

> [!TIP] 数值核对（2026-10-25）
> 标量例子的前向（$a_1=3,\phi_1=3,a_2=5,\phi_2=5,a_3=-3,L=1$）、三个中间导数（$-2,2,4$）、六个参数梯度（$\partial w_3=-10,\partial b_3=-2,\partial w_2=6,\partial b_2=2,\partial w_1=8,\partial b_1=4$）我都独立重算，与讲义一致。
> W04 网络例子的 $\nabla J_2=(-1)$、$\nabla J_1=(0,1,-1/2)$、$\nabla W_2=(0,-3,-1)$、$\nabla W_1=\begin{pmatrix}0&1&-1/2\\0&2&-1\end{pmatrix}$ 同样复算一致。未发现讲义与 Slides 数值冲突。

## 核心脉络

W04 我们已经能把输入一路算成网络输出，并选好损失。要做梯度下降，还需要知道**损失对每个权重、每个偏置的导数**。

- 用 W01 的链式法则（chain rule）逐个硬算当然可以，但**大量中间步骤会被重复计算**。
- **反向传播（backpropagation）只是一种组织计算的方式**：先保存前向传播的中间值，从损失出发、由后往前，把每个中间导数算一次，既用于本层的参数导数，又传给前一层复用。
- 它不是新的数学，就是链式法则；它也不负责更新参数（那是梯度下降的事）。`L.backward()` 只算导数，不更新参数。

---

## 0. 全貌：反向传播解决什么问题

### 0.1 我们有损失，需要梯度

前向传播在当前参数下算出损失：

$$\underbrace{\boldsymbol x}_{\text{输入}}\longrightarrow\underbrace{f_{\boldsymbol\theta}(\boldsymbol x)}_{\text{预测/打分}}\longrightarrow\underbrace{\mathcal L(\boldsymbol\theta)}_{\text{损失}}.$$

梯度下降需要损失对网络参数的梯度：

$$\boldsymbol\theta^{(t+1)}=\boldsymbol\theta^{(t)}-\eta\,\nabla\mathcal L\bigl(\boldsymbol\theta^{(t)}\bigr).$$

问题：怎样为**每一个**权重和偏置高效地求出导数？

### 0.2 反向传播的思路

> 先算**最后一层**参数的损失导数，然后**逐层往回**，复用已经算出的导数。

具体到讲义的做法，是为每一层定义一个「剩余函数」$J_r$（见 §1.2），它表示「从该层预激活出发、到最终损失为止的所有运算」。把 $J_r$ 对其输入的导数算出来后：

- 它**本身就是损失对该层偏置的导数**；
- 乘上该层输入，得到**对权重的导数**；
- 再乘本层权重和 ReLU 的导数，得到**前一层的**那个中间导数。

一份中间导数，三处复用，这就是高效的来源。

---

## 1. 标量网络：手算完整例子

先看一个每一层都只有一个神经元的小网络（结构 $1\to1\to1\to1$，两个隐藏层 + 一个输出层），用 W01 的一元链式法则就够了。

### 1.1 网络结构与前向传播

约定 $\phi_0(x):=x$（输入本身），于是 $\phi_{r-1}(x)$ 就是第 $r$ 层的输入。前向计算为讲义式 (1)：

$$\begin{aligned}
a_1(x)&=\phi_0(x)w_1+b_1, &\phi_1(x)&=\operatorname{ReLU}(a_1(x)),\\
a_2(x)&=\phi_1(x)w_2+b_2, &\phi_2(x)&=\operatorname{ReLU}(a_2(x)),\\
a_3(x)&=\phi_2(x)w_3+b_3, &f_{\boldsymbol\theta}(x)&=a_3(x).
\end{aligned}$$

注意：**输出层不接 ReLU**，它的仿射输出 $a_3$ 就是预测值。参数向量按 $\boldsymbol\theta=(b_1,w_1,b_2,w_2,b_3,w_3)^{\top}$ 排列。

取讲义式 (3) 的数值（单个固定样本，平方损失 $\mathcal L_{\text{sq}}=(a_3-y)^2$）：

$$x=2,\quad y=-2,\qquad w_1=1,\ b_1=1,\quad w_2=2,\ b_2=-1,\quad w_3=-1,\ b_3=2.$$

**前向结果**：

$$\begin{aligned}
a_1&=2(1)+1=3, &\phi_1&=3,\\
a_2&=3(2)-1=5, &\phi_2&=5,\\
a_3&=5(-1)+2=-3, &\mathcal L_{\text{sq}}&=(-3+2)^2=1.
\end{aligned}$$

> [!NOTE] 补充说明：前向传播要「保存」什么
> 反向计算会用到前向的中间值，所以前向时要把这些数留下来：各层输入 $\phi_0,\phi_1,\phi_2$、各层预激活 $a_1,a_2,a_3$（用于判断 ReLU 开关），以及参数和目标。这就是 Slides 反复强调的「First, save the forward values」。

### 1.2 「剩余函数」$J_r$

从每个预激活出发，把「得到最终损失前还要做的全部运算」定义成一个函数 $J_r$。定义时目标 $y$ 和更后层的参数都视为固定，只让它的自变量 $u$ 变化。

**输出层（$J_3$）**：给定候选输出 $u$，剩下的运算就是减 $y$ 再平方：

$$J_3(u)=(u-y)^2.$$

**隐藏层 2（$J_2$）**：还要经过 ReLU、最后一层仿射、再算损失，讲义式 (4)：

$$J_2(u)=\bigl(\operatorname{ReLU}(u)\,w_3+b_3-y\bigr)^2=J_3\bigl(\operatorname{ReLU}(u)\,w_3+b_3\bigr).$$

它只依赖 $w_3,b_3,y$，**不依赖** $x,w_1,b_1,w_2,b_2$（那些只用于算出 $a_2$）。

**隐藏层 1（$J_1$）**：讲义式 (5)：

$$J_1(u)=J_2\bigl(\operatorname{ReLU}(u)\,w_2+b_2\bigr).$$

在各自的前向取值上，三个函数都给出同一个损失（讲义式 6）：

$$J_r\bigl(a_r(x)\bigr)=\mathcal L_{\text{sq}},\qquad r=1,2,3.$$

> [!NOTE] 补充说明：为什么要引入 $J_r$
> 因为对 $J_r$ 求导、并在 $a_r(x)$ 处取值，正好得到损失对 $b_r$ 的导数；再乘层输入 $\phi_{r-1}$ 就得到对 $w_r$ 的导数；还能继续往前推。一个函数把「本层两个参数 + 前一层」三件事串起来。

### 1.3 反向传播：从输出层开始

先回忆 ReLU 的导数（$u\neq0$ 时）：

$$\operatorname{ReLU}'(u)=\operatorname{step}(u),\qquad
\operatorname{step}(u)=\begin{cases}0,&u\le0,\\1,&u>0.\end{cases}$$

预激活为正，梯度「放行」（乘 1）；为负，梯度「截断」（乘 0）。本例 $a_1=3,a_2=5$ 都为正，故 $\operatorname{step}(3)=\operatorname{step}(5)=1$。

#### 第一步：输出层（层 3）

$J_3(u)=(u-y)^2$ 的导数是

$$J_3'(u)=2(u-y),\qquad J_3'\bigl(a_3(x)\bigr)=2(a_3-y).$$

在 $a_3=-3$ 处：

$$\boxed{J_3'(-3)=2(-3+2)=-2.}$$

由 $a_3=\phi_2 w_3+b_3$（$\phi_2$ 不依赖 $w_3,b_3$）和链式法则，讲义式 (7)：

$$\frac{\partial\mathcal L_{\text{sq}}}{\partial b_3}=J_3'(a_3)\cdot1=-2,\qquad
\frac{\partial\mathcal L_{\text{sq}}}{\partial w_3}=J_3'(a_3)\,\phi_2=(-2)(5)=-10.$$

同一个 $J_3'(a_3)=-2$：偏置直接用，权重乘 $\phi_2$。

#### 第二步：隐藏层 2

对 $J_2(u)=J_3(\operatorname{ReLU}(u)w_3+b_3)$ 求导（$u\neq0$），讲义式 (8)：

$$J_2'\bigl(a_2(x)\bigr)=J_3'(a_3)\,\underbrace{w_3}_{\text{仿射导数}}\,\underbrace{\operatorname{step}(a_2)}_{\text{ReLU 导数}}.$$

代入数值：

$$\boxed{J_2'(5)=(-2)(-1)(1)=2.}$$

于是本层两个参数导数（讲义式 9）：

$$\frac{\partial\mathcal L_{\text{sq}}}{\partial b_2}=J_2'(a_2)=2,\qquad
\frac{\partial\mathcal L_{\text{sq}}}{\partial w_2}=J_2'(a_2)\,\phi_1=2(3)=6.$$

#### 第三步：隐藏层 1

同理，讲义式 (10)：

$$J_1'\bigl(a_1(x)\bigr)=J_2'(a_2)\,w_2\,\operatorname{step}(a_1).$$

$$\boxed{J_1'(3)=(2)(2)(1)=4.}$$

本层参数导数（讲义式 11）：

$$\frac{\partial\mathcal L_{\text{sq}}}{\partial b_1}=J_1'(a_1)=4,\qquad
\frac{\partial\mathcal L_{\text{sq}}}{\partial w_1}=J_1'(a_1)\,\phi_0=4(2)=8.$$

> [!NOTE] 标量例子的通用规则（讲义式 12）
> 对 $r=1,2,3$：
> $$\frac{\partial\mathcal L_{\text{sq}}}{\partial b_r}=J_r'(a_r),\qquad
> \frac{\partial\mathcal L_{\text{sq}}}{\partial w_r}=J_r'(a_r)\,\phi_{r-1}.$$
> 顺序是先算 $J_3'(a_3)$，复用得 $J_2'(a_2)$，再复用得 $J_1'(a_1)$。

### 1.4 完整梯度

按 $\boldsymbol\theta=(b_1,w_1,b_2,w_2,b_3,w_3)$ 的顺序：

$$\boxed{\nabla\mathcal L_{\text{sq}}(\boldsymbol\theta)=\begin{pmatrix}4&8&2&6&-2&-10\end{pmatrix}^{\top}.}$$

随后梯度下降才更新：$\boldsymbol\theta^{(t+1)}=\boldsymbol\theta^{(t)}-\eta\,\nabla\mathcal L$。**务必在全部梯度算完之后再更新参数。**

### 1.5 零预激活的约定（step(0)）

ReLU 在 $0$ 处不可导（左导 0、右导 1）。PyTorch 的 ReLU 在反向时把 $0$ 处的乘子取为 $0$，即采用 $\operatorname{step}(0)=0$。讲义强调这是**计算上的约定（computational convention）**，并不是 ReLU 在 0 处真正的导数。

---

## 2. 向量 / 矩阵版本

标量例子已经包含了反向传播的全部组织方式：保存前向值、从损失开始、逐层复用。宽网络只是把数字换成行向量与矩阵。

### 2.1 从标量到行和矩阵

设网络共 $L$ 层（含输出层），$d_r$ 是第 $r$ 层宽度，$d_0$ 是输入宽度。对应关系：

| 量 | 标量网络 | 宽网络 |
| --- | --- | --- |
| 第 $r$ 层输入 | $\phi_{r-1}(x)$ | $\boldsymbol\phi_{r-1}(\boldsymbol x)\in\mathbb R^{1\times d_{r-1}}$ |
| 权重 | $w_r$ | $\boldsymbol W_r\in\mathbb R^{d_{r-1}\times d_r}$ |
| 偏置 | $b_r$ | $\boldsymbol b_r\in\mathbb R^{1\times d_r}$ |
| 预激活 | $a_r(x)$ | $\boldsymbol a_r(\boldsymbol x)\in\mathbb R^{1\times d_r}$ |

前向传播（讲义式 13，$\cdot$ 表示普通矩阵乘法）：

$$\begin{aligned}
\boldsymbol\phi_0(\boldsymbol x)&=\boldsymbol x,\\
\boldsymbol a_r(\boldsymbol x)&=\boldsymbol\phi_{r-1}(\boldsymbol x)\cdot\boldsymbol W_r+\boldsymbol b_r,&&r=1,\ldots,L,\\
\boldsymbol\phi_r(\boldsymbol x)&=\operatorname{ReLU}(\boldsymbol a_r(\boldsymbol x)),&&r=1,\ldots,L-1.
\end{aligned}$$

最后一层不接 ReLU：$\boldsymbol a_L(\boldsymbol x)=\boldsymbol f_{\boldsymbol\theta}(\boldsymbol x)$。

### 2.2 剩余函数的梯度：一个行向量

剩余函数现在以「行」为输入、返回一个标量：

$$J_r:\mathbb R^{1\times d_r}\longrightarrow\mathbb R.$$

它的梯度（讲义式 15）是一个形状 $1\times d_r$ 的行，每个坐标对应一个输入分量的损失导数：

$$\nabla J_r\bigl(\boldsymbol a_r(\boldsymbol x)\bigr)=\begin{pmatrix}\dfrac{\partial J_r}{\partial u_1}&\cdots&\dfrac{\partial J_r}{\partial u_{d_r}}\end{pmatrix}.$$

这一行取代了标量例子里的单个数字 $J_r'(a_r)$。

### 2.3 两种乘法：矩阵乘 $\cdot$ 与逐元素乘 $\odot$

反向传播要用到两种不同的乘法，讲义专门做了区分：

- **普通矩阵乘法 $\cdot$**：$A\in\mathbb R^{m\times n}$、$B\in\mathbb R^{n\times p}$，$[A\cdot B]_{ij}=\sum_k A_{ik}B_{kj}$，结果形状 $m\times p$（第一个的列数 = 第二个的行数）。
- **逐元素乘法（componentwise）$\odot$**：$A,B$ 形状完全相同，$[A\odot B]_{ij}=A_{ij}B_{ij}$，不做求和，结果形状不变。

讲义给的例子，$A=\begin{pmatrix}1&2\\3&4\end{pmatrix}$、$B=\begin{pmatrix}5&6\\7&8\end{pmatrix}$：

$$A\cdot B=\begin{pmatrix}19&22\\43&50\end{pmatrix},\qquad
A\odot B=\begin{pmatrix}5&12\\21&32\end{pmatrix}.$$

左上角：矩阵乘是 $1\cdot5+2\cdot7=19$，逐元素乘是 $1\cdot5=5$。

### 2.4 通用反向传播公式（讲义式 17、19）

**参数梯度**：已知 $\nabla J_r(\boldsymbol a_r)$，则

$$\boxed{\nabla_{\boldsymbol W_r}\mathcal L=\boldsymbol\phi_{r-1}^{\top}\cdot\nabla J_r(\boldsymbol a_r),\qquad
\nabla_{\boldsymbol b_r}\mathcal L=\nabla J_r(\boldsymbol a_r).}$$

形状与参数一致：权重梯度 $d_{r-1}\times d_r$，偏置梯度 $1\times d_r$。

**继续往前一层**（$r>1$，讲义式 19）：

$$\boxed{\nabla J_{r-1}(\boldsymbol a_{r-1})=\Bigl(\nabla J_r(\boldsymbol a_r)\cdot\boldsymbol W_r^{\top}\Bigr)\odot\operatorname{step}(\boldsymbol a_{r-1}).}$$

其中 $\operatorname{step}(\boldsymbol u)$ 表示把标量 step 逐分量作用（讲义式 18）。

**维度检查**（这是自查最有用的一步）：

- 参数梯度：$\boldsymbol\phi_{r-1}^{\top}\in\mathbb R^{d_{r-1}\times1}$，$\nabla J_r\in\mathbb R^{1\times d_r}$，相乘得 $d_{r-1}\times d_r$，与 $W_r$ 同形。
- 往前一层：$\nabla J_r\in\mathbb R^{1\times d_r}$、$W_r^{\top}\in\mathbb R^{d_r\times d_{r-1}}$，相乘得 $1\times d_{r-1}$；$\operatorname{step}(\boldsymbol a_{r-1})$ 也是 $1\times d_{r-1}$，两者可逐元素相乘。

### 2.5 完成 Lecture 4 网络的例子

这是 W04 那个网络：输入 2 维、隐藏层宽 3、输出 1 个。参数（讲义式 21）：

$$\boldsymbol W_1=\begin{pmatrix}1&2&-1\\-1&0&1\end{pmatrix},\quad\boldsymbol b_1=\begin{pmatrix}0&1&0\end{pmatrix},\quad
\boldsymbol W_2=\begin{pmatrix}2\\-1\\1/2\end{pmatrix},\quad b_2=-1.$$

输入 $\boldsymbol x=(1\ \ 2)$、目标 $y=-3$。

**前向**：

$$\boldsymbol a_1=(-1\ \ 3\ \ 1),\quad\boldsymbol\phi_1=(0\ \ 3\ \ 1),\quad
\boldsymbol a_2=(-7/2),\quad\mathcal L_{\text{sq}}=(-7/2+3)^2=1/4.$$

**输出层反向**：

$$\nabla J_2(\boldsymbol a_2)=\bigl(2(-7/2+3)\bigr)=(-1).$$

$$\nabla_{\boldsymbol W_2}\mathcal L=\boldsymbol\phi_1^{\top}\cdot(-1)=\begin{pmatrix}0\\3\\1\end{pmatrix}(-1)=\begin{pmatrix}0\\-3\\-1\end{pmatrix},\qquad
\frac{\partial\mathcal L}{\partial b_2}=-1.$$

**隐藏层反向**：

$$\nabla J_2\cdot\boldsymbol W_2^{\top}=(-1)(2,-1,1/2)=(-2,1,-1/2),$$

$$\operatorname{step}(\boldsymbol a_1)=\operatorname{step}(-1,3,1)=(0,1,1),$$

$$\nabla J_1(\boldsymbol a_1)=(-2,1,-1/2)\odot(0,1,1)=\boxed{(0,1,-1/2)}.$$

注意最后一个分量是 $-1/2$（负的也保留），因为它的预激活为正。于是

$$\nabla_{\boldsymbol W_1}\mathcal L=\begin{pmatrix}1\\2\end{pmatrix}(0,1,-1/2)=\boxed{\begin{pmatrix}0&1&-1/2\\0&2&-1\end{pmatrix}},\qquad
\nabla_{\boldsymbol b_1}\mathcal L=(0,1,-1/2).$$

> [!NOTE] 补充说明：为什么第一列是 0
> 第一个隐藏单元的预激活 $a_{1,1}=-1<0$，ReLU 关闭，$\nabla J_1$ 第一个分量为 0；小幅改动连到它的权重仍保持该预激活为负、激活为 0，所以 $W_1$ 第一列、$b_1$ 第一个分量的梯度都是 0。这也是「哪一列不更新、为什么」这类问题的答案。

---

## 3. 三种任务的初始反向行

反向传播的第一步，取决于任务对应的输出损失 $J_L$，由此得到「初始反向行」。之后的参数梯度公式和逐层递推对三种任务完全相同。以下都针对单个固定样本。

### 3.1 回归（Regression）

标量目标 $y\in\mathbb R$，输出 1 维（$d_L=1$）：$J_L(\boldsymbol u)=(u_1-y)^2$。

$$\boxed{[\nabla J_L(\boldsymbol a_L)]_1=2(f_{\boldsymbol\theta}(\boldsymbol x)-y).}\tag{23}$$

### 3.2 二分类（Binary classification）

$y\in\{0,1\}$，输出 1 维但它是**打分/ logit**，概率 $p_{\boldsymbol\theta}=\sigma(f_{\boldsymbol\theta})$。剩余函数同时包含 sigmoid 和 logistic 损失：$J_L(\boldsymbol u)=\ell_{\log}(\sigma(u_1),y)$。

$$\boxed{[\nabla J_L(\boldsymbol a_L)]_1=p_{\boldsymbol\theta}(\boldsymbol x)-y.}\tag{24}$$

> [!WARNING] 注意：不要再乘一次 sigmoid 导数
> 式 (24) 已经把 sigmoid 的导数包含进去了（推导见讲义 A.4.2：$\ell_{\log}$ 对概率的导数 $\frac{p-y}{p(1-p)}$，乘上 $\sigma'(u)=\sigma(u)(1-\sigma(u))$，恰好约成 $\sigma(u)-y=p-y$）。反向循环里直接用，不要重复乘。

### 3.3 多分类（Multiclass classification）

$K$ 类，输出宽 $d_L=K$，softmax 把打分变成概率 $\boldsymbol p_{\boldsymbol\theta}=\operatorname{softmax}(\boldsymbol a_L)$。

$$\boxed{[\nabla J_L(\boldsymbol a_L)]_k=p_{\boldsymbol\theta,k}-\delta_{ky},\qquad k=1,\ldots,K.}\tag{25}$$

即：**概率行里，把真实类别 $y$ 对应那一项减 1，其余不变**。例如 $K=3,y=2$：$(p_1,p_2-1,p_3)$。softmax 的导数同样已包含在内。

> [!NOTE] 三种初始行速记
> | 任务 | 初始反向行 |
> | --- | --- |
> | 回归 | $2(f-y)$ |
> | 二分类 | $p-y$ |
> | 多分类 | $p_k-\delta_{ky}$（真实类减 1） |

---

## 4. 从单个样本到整个数据集

前面算的都是单个固定样本的损失导数。对含 $N$ 个样本 $(\boldsymbol x_i,y_i)$ 的数据集，所有样本共用同一套参数 $\boldsymbol\theta$。

经验损失是各样本损失之和（讲义式 26）：

$$\mathcal L(\boldsymbol\theta)=\sum_{i=1}^{N}\mathcal L_i(\boldsymbol\theta).$$

求和的导数等于导数求和（讲义式 27）：

$$\boxed{\nabla\mathcal L(\boldsymbol\theta)=\sum_{i=1}^{N}\nabla\mathcal L_i(\boldsymbol\theta).}$$

**完整一次梯度下降更新的步骤**：

1. 准备一个全 0 的梯度累加器；
2. 对每个样本：前向 → 反向 → 把它的参数梯度加到累加器；
3. 全部样本处理完后，用累加梯度做一次参数更新。

前两步中参数保持固定（每个 $\nabla\mathcal L_i$ 都要在同一个 $\boldsymbol\theta$ 下计算），求和完成后才更新。

> [!NOTE] 补充说明：minibatch 与内存
> 可以一次处理一个样本、加完梯度就丢弃它的前向值，这样工作内存始终是 $O(P)$（$P$ 为参数个数），不含数据集本身。把「整个数据集」换成「一个 minibatch」就是 minibatch 梯度下降；全数据集一次更新即批量梯度下降。

---

## 5. 效率分析：为什么反向传播高效

### 5.1 单样本：$O(P)$

对一个含 $P$ 个标量权重/偏置的网络，一次前向 + 一次反向就能得到损失和全部 $P$ 个参数导数：

$$\text{时间 }O(P),\qquad \text{内存 }O(P).$$

内存包括参数、梯度和保存的前向值。在需要显式写出并存储完整梯度的前提下，这已是**最优（只差常数因子）**：光是写出梯度的 $P$ 个分量就至少要 $P$ 次操作、存 $P$ 个数。

### 5.2 对比：一次只算一个参数会是 $O(P^2)$

如果对每个参数分别走一遍链式法则：在标量网络（$L$ 层、每层一个权重一个偏置，$P=2L$）中，第 1 层参数要穿过全部 $L$ 层、第 2 层穿过 $L-1$ 层……总「层级访问」次数为

$$2\bigl(L+(L-1)+\cdots+1\bigr)=L(L+1)=O(P^2).$$

重复劳动发生在靠后的层：同样的「剩余计算」对每个前面的参数都要重算一遍。反向传播把这些中间导数算一次、反复复用，把总时间降到 $O(P)$。

### 5.3 整个数据集

$N$ 个样本：逐个加梯度共 $O(NP)$ 时间（反向传播）；而「逐参数分别算」的笨办法会是 $O(NP^2)$。

> 把公式落到代码上：Lab 5 在 PyTorch 里用 autograd 做反向、手动做梯度下降。[W05 Lab｜用 PyTorch 搭建并拟合神经网络](FOAI-公开版-W05-PyTorch神经网络实验.md)

---

## 学习提示

- **必须能手算**：给定一个 2–3 层小网络（含数值权重和一个样本），完整写出前向各值、从损失开始逐层写出参数梯度。这是必须掌握的典型手算题，§1 和 §2.5 请各独立算一遍。
- **维度检查是防错利器**：写不出矩阵形状时，先把每个量的行列维度标出来，再决定用 $\cdot$ 还是 $\odot$。
- **三件容易错的事**：① 输出层不接 ReLU；② 二分类/多分类的初始行已含 sigmoid/softmax 导数，不要重复乘；③ 梯度算完才更新，更新在 `no_grad` 下做、更新后清 `.grad`。
- sigmoid/softmax、交叉熵、$\sigma'(x)$、$\operatorname{ReLU}'$ 等常见公式建议记熟；反向传播的通用递推要能自己推出来。

## 自测 / 课后思考

1. 不看答案，把 §1 标量例子从 $x=2$ 开始重算到六个梯度，再与 $(4,8,2,6,-2,-10)$ 核对。
2. 给定网络 $\boldsymbol a_1=\boldsymbol x\boldsymbol W_1,\phi_1=\operatorname{ReLU}(\boldsymbol a_1),f=\phi_1\boldsymbol W_2,L=(f-y)^2$，按 §2 的方法完成 (a)–(d)，并回答「哪一列不更新、为什么」。
3. 写出多分类初始反向行为什么是「真实类减 1」：从 $J_L=-u_y+\log\sum_j e^{u_j}$ 对每个 $u_k$ 求导验证。
4. 为什么两个线性层之间没有非线性，就等价于一个线性层？（联系 W04 与本讲 §5。）
5. 讲义 §6 有三道练习题（回归反向、二分类、三分类）；本文不提供解答，建议先独立完成。

## 待核对 / Open Questions

- [ ] 讲义 §6 三道练习题的官方解答发布后，对照自己的结果。
- [ ] 跟做 Lab 5：跑通 XOR 数据、autograd、重启、宽/深网络四个实验，对照 notebook 里的参考说明自查。
- [ ] 本讲未覆盖：W07 PCA/ICA、W08 各类自编码器、W09 CNN/RNN/Transformer 的反向传播细节，待对应周次材料整理。

## 返回链接

- [FOAI 公开版目录](README.md)｜[W05 Lab｜用 PyTorch 搭建并拟合神经网络](FOAI-公开版-W05-PyTorch神经网络实验.md)
- 前置：[W04 神经网络](FOAI-公开版-W04-神经网络.md)、[W01 数学预备](FOAI-公开版-W01-数学预备.md)
- 后续：W07 PCA/ICA、W08 Autoencoder、W09 CNN/RNN/Transformer（待整理）
