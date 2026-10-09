---
title: FOAI W02｜线性回归
course: FOAI
week: 2
type: lecture
tags:
  - AI-MSc
  - TB1
  - FOAI
  - 线性回归
  - 优化
source: 课程材料（Conor Houghton、Roberto Colomboni）的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: draft
publication: pending
updated: 2026-10-09
---

# FOAI W02｜线性回归（Linear Regression）

> [!IMPORTANT]
> **非官方学习笔记**：这是我在 University of Bristol「Foundations of Artificial Intelligence（SEMTM0051, 2026/27）」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：Conor Houghton、Roberto Colomboni。原始材料：[semtm0051/2026_27](https://github.com/semtm0051/2026_27/tree/322787dec91f174303ccf90399bb6243a060eb7d/)。经两位讲师同意公开；不含练习解答与非公开材料（如课堂录像）。

> [!NOTE]
> **资料来源**
> 一手讲义：[`Week_02/Notes/Week_02 - Linear Regression.pdf`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_02/Notes/Week_02%20-%20Linear%20Regression.pdf)。
> 本讲是整个课程的方法论母本：**选函数族 → 选损失 → 最小化经验损失**，后面所有模型都是这个套路的变体。

---

## 1. 建模流程（一图流）

观测数据 $(\boldsymbol{x}_i, y_i)_{i=1}^N$，假设
$$y_i = f_{\text{true}}(x_i) + \varepsilon_i,$$
其中 $f_{\text{true}$ 未知、$\varepsilon_i$ 是噪声。我们做三件事：

1. **选预测函数族** $\{f_\theta\}$（例如仿射函数 $b + ax$）；
2. **选损失** $\ell(\hat y, y)$，衡量单条预测错多少；
3. **最小化经验损失** $\mathcal{L}(\theta) = \sum_i \ell(f_\theta(x_i), y_i)$，得到 $\theta^* \in \operatorname{argmin}_\theta \mathcal{L}(\theta)$。

> [!NOTE]
> **三个必须分清的对象**
> - $\ell(\hat y, y)$：**一条**样本上的损失（输入：预测、目标）；
> - $\mathcal{L}(\theta)$：**整个数据集**上的经验损失（输入：参数）；
> - $f_\theta(x)$：预测函数（输入：特征；输出：预测值）。
> 拟合时 $x_i, y_i$ 固定，只有 $\theta$ 在变。

## 2. 损失函数

$$\ell_{\text{abs}}(\hat y, y) = |\hat y - y|, \qquad \ell_{\text{sq}}(\hat y, y) = (\hat y - y)^2.$$

- 都非负，预测=目标时为 0。
- 误差翻倍：绝对损失翻倍，平方损失翻**四倍**——平方损失对大误差更敏感。
- $\ell_{\text{sq}}$ 处处可微（对 $r = \hat y-y$ 导数是 $2r$）；$\ell_{\text{abs}}$ 在 0 处有尖角，不可导（习题 4–5 会用）。
- 残差定义为 $r_i(\theta) := f_\theta(x_i) - y_i$（**预测减目标**）：正=高估，负=低估。

**归一化无关性**：$\frac1N \mathcal{L}$、$\frac12 \mathcal{L}$、$\frac{1}{2N}\mathcal{L}$ 的最优点完全相同，只是梯度差一个常数倍。

**过拟合提示**：$\mathcal{L}$ 小 ≠ 在没见过的数据上准。后者叫**泛化（generalisation）**，本讲只讨论拟合。

## 3. 三个越来越灵活的函数族（标量输入）

| 族 | 预测 | 参数 |
|---|---|---|
| 常数 | $b$ | $b \in \mathbb{R}$ |
| 过原点直线 | $ax$ | $a \in \mathbb{R}$ |
| 一般直线（仿射） | $b + ax$ | $(b,a) \in \mathbb{R}^2$ |

### 3.1 最佳常数 = 样本均值
$$\mathcal{L}(b) = \sum_i (b - y_i)^2, \quad \mathcal{L}'(b) = 2(Nb - \sum_i y_i).$$
令其为 0：
$$b^* = \bar y = \frac{1}{N}\sum_i y_i.$$
> 平方损失 → 均值；换绝对损失 → **中位数**（讲义习题 5）。**"最佳"取决于损失。**

### 3.2 最佳过原点直线
$$a^* = \frac{\sum_i x_i y_i}{\sum_i x_i^2} \quad (\text{要求 } \sum x_i^2 > 0).$$

### 3.3 最佳一般直线（最小二乘）
令 $\mathcal{L}(b,a) = \sum_i (b + a x_i - y_i)^2$ 的两个偏导为 0：
$$\begin{cases}
Nb + a\sum x_i = \sum y_i \\
b\sum x_i + a\sum x_i^2 = \sum x_i y_i
\end{cases}$$
解出：
$$a^* = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2}, \qquad b^* = \bar y - a^* \bar x.$$
- 拟合线必过 $(\bar x, \bar y)$。
- 讲义用"展开 $L(b^*+u, a^*+v)$"证明这是**全局最小**（交叉项因 $\sum r_i^* = 0$ 和 $\sum x_i r_i^* = 0$ 而消失，剩下 $\sum (u+vx_i)^2 \ge 0$）。

## 4. 多元线性回归（矩阵形式）

### 4.1 设计矩阵
输入 $\boldsymbol{x}_i \in \mathbb{R}^{1\times d}$，权重列 $\boldsymbol{w} \in \mathbb{R}^{d\times 1}$，预测 $f_\theta(\boldsymbol{x}_i) = b + \boldsymbol{x}_i \boldsymbol{w}$。

把截距 $b$ 也并入参数列 $\theta = (b, \boldsymbol{w}^\top)^\top \in \mathbb{R}^{(d+1)\times 1}$，并在每个观测前加一列 1：
$$\boldsymbol{X} = \begin{pmatrix}1 & \boldsymbol{x}_1 \\ \vdots & \vdots \\ 1 & \boldsymbol{x}_N\end{pmatrix} \in \mathbb{R}^{N\times(d+1)}, \qquad \boldsymbol{y} \in \mathbb{R}^{N\times 1}.$$
则所有预测 = $\boldsymbol{X}\theta$，残差 = $\boldsymbol{r}(\theta) = \boldsymbol{X}\theta - \boldsymbol{y}$，损失
$$\mathcal{L}_{\text{sq}}(\theta) = \|\boldsymbol{X}\theta - \boldsymbol{y}\|_2^2.$$

### 4.2 梯度与正规方程
对 $\theta_j$ 求导（用 [W01](FOAI-公开版-W01-数学预备.md) 的 $\partial r_i/\partial\theta_j = X_{ij}$）：
$$\frac{\partial \mathcal{L}}{\partial \theta_j} = 2\sum_i X_{ij} r_i = 2(\boldsymbol{X}^\top \boldsymbol{r})_j.$$
写成向量：
$$\boxed{\nabla \mathcal{L}_{\text{sq}}(\theta) = 2\boldsymbol{X}^\top(\boldsymbol{X}\theta - \boldsymbol{y}).}$$

令梯度为 0 → **正规方程（normal equations）**：
$$\boxed{\boldsymbol{X}^\top \boldsymbol{X}\,\theta = \boldsymbol{X}^\top \boldsymbol{y}.}$$

若 $\boldsymbol{X}$ 列满秩（要求 $N \ge d+1$ 且列线性无关），$\boldsymbol{X}^\top\boldsymbol{X}$ 可逆（讲义附录 A 用 $\|\boldsymbol{X}\boldsymbol{v}\|^2 = 0 \Rightarrow \boldsymbol{v}=0$ 证明）：
$$\boxed{\theta^* = (\boldsymbol{X}^\top\boldsymbol{X})^{-1}\boldsymbol{X}^\top \boldsymbol{y}.}$$

> [!WARNING]
> **形状别搞反**
> - $\boldsymbol{X}^\top\boldsymbol{X}$ 是 $(d+1)\times(d+1)$，是正规方程里的矩阵；
> - $\boldsymbol{X}\boldsymbol{X}^\top$ 是 $N\times N$，**不是**正规方程里的东西；
> - 两者不能互换。

### 4.3 非线性特征（"线性"指对参数，不是对 x）
选固定的特征函数 $\phi_1,\dots,\phi_m$，把输入映射成 $\phi(x) = (\phi_1(x),\dots,\phi_m(x))$，预测
$$f_w(x) = \phi(x)\boldsymbol{w} = \sum_j w_j \phi_j(x).$$
即使 $\phi$ 是非线性的（例如 $\phi(x) = (1, x, x^2)$），模型对 $\boldsymbol{w}$ 仍然是线性的，**正规方程原样适用**：
$$\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\,\boldsymbol{w} = \boldsymbol{\Phi}^\top\boldsymbol{y}, \qquad \boldsymbol{w}^* = (\boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top\boldsymbol{y}.$$
讲义例子：$\phi(x)=(1,x,x^2)$ 对 5 个点拟合得 $w^* = (1, 2, 1/2)$，损失从直线的 6 降到 2.5。

> [!NOTE]
> **什么叫"线性模型"**
> $w_1 + w_2 x + w_3 x^2$ 对 $x$ 是非线性的，但对 $(w_1,w_2,w_3)$ 是线性的——所以仍叫线性回归。反例：$\sin(wx)$ 对 $w$ 非线性，不在此列。

## 5. 梯度下降（Gradient Descent）

正规方程直接求解代价 $O(Np^2 + p^3)$（$p=d+1$），$p$ 大时贵。梯度下降每步 $O(Np)$，适合大模型，也是后面神经网络的唯一选择。

### 5.1 更新规则
从 $\theta^{(0)}$ 出发，反复：
$$\boxed{\theta^{(t+1)} := \theta^{(t)} - \eta\, \nabla \mathcal{L}(\theta^{(t)}), \qquad \eta > 0 \text{ 是学习率}.}$$

**为什么朝负梯度走**：由局部线性近似 $\mathcal{L}(\theta+\Delta) \approx \mathcal{L}(\theta) + \nabla\mathcal{L}^\top \Delta$，取 $\Delta = -\eta\nabla\mathcal{L}$ 时增量为 $-\eta\|\nabla\mathcal{L}\|^2 < 0$（梯度非零时）。
> [!WARNING]
> **步长太大不收敛**
> 例子 $F(a)=100a^2$，$\eta=0.1$ 时 $a^{(t+1)} = -19 a^{(t)}$，发散。学习率要小到"充分小"。

### 5.2 线性回归的具体更新
残差 $\boldsymbol{r}^{(t)} = \boldsymbol{X}\theta^{(t)} - \boldsymbol{y}$，梯度 $2\boldsymbol{X}^\top \boldsymbol{r}^{(t)}$，所以
$$\boldsymbol{r}^{(t)} := \boldsymbol{X}\theta^{(t)} - \boldsymbol{y}, \qquad \theta^{(t+1)} := \theta^{(t)} - 2\eta\, \boldsymbol{X}^\top \boldsymbol{r}^{(t)}.$$
- 先用旧参数算一次残差，**两个系数同时更新**；
- 然后才用新参数算下一轮残差。

对标量直线：
$$b^{(t+1)} = b^{(t)} - 2\eta \sum_i r_i^{(t)}, \qquad a^{(t+1)} = a^{(t)} - 2\eta \sum_i x_i r_i^{(t)}.$$

讲义 worked example：从 $(b,a)=(0,0)$、$\eta=0.04$ 出发，一步到 $(0.8, 1.6)$，损失从 66 降到 14.8；解析解是 $(2,2)$，损失 6。

## 6. 习题要点（讲义 Section 8）

- **Ex 1**：比较 $f_1=x, f_2=2x, f_3=3x$ 在新数据集上的 SSE；最小的那个只是"这三条里最好"，**不能**推出是所有直线里最好。
- **Ex 2**：用矩阵法手算直线拟合（构造 $X$、算 $X^\top X$、$X^\top y$、解正规方程）。
- **Ex 3**：拟合 $w_1 x + w_2 x^2$（**不加**常数列），检查 $\Phi^\top\Phi$ 是否可逆。
- **Ex 4**：讨论 $q(s)=|s|$ 在 $s\neq 0$ 和 $s=0$ 处的可导性。提示：分 $s>0$、$s<0$ 两段看，再看 0 处左右导数。
- **Ex 5**：把常数预测从平方损失换成绝对损失 → 最优 $b$ 变成**中位数**。这题是全课"损失决定最优解"的最佳示例。

## 跨周联系

- 数学工具：[FOAI-W01-数学预备](FOAI-公开版-W01-数学预备.md)（梯度、链式法则、$\partial r_i/\partial\theta_j$）。
- 代码实现：[FOAI-W03-PyTorch线性回归实验](FOAI-公开版-W03-PyTorch线性回归实验.md)（用 PyTorch 写设计矩阵、闭式解、梯度下降）。
- 下一步：Week 3 讲义 Linear Classification（PDF 尚未上传）——把平方损失换成分类损失，把 $y\in\mathbb{R}$ 换成 $y\in\{0,1\}$，会用到 W01 Ex 15 的 sigmoid。
- 概率视角：平方损失对应高斯噪声假设，交叉熵对应伯努利——之后会联系 SCEM 的概率统计。
