---
title: FOAI W02｜课堂 Slides 逐页标注（缺课自学版）
course: FOAI
week: 2
type: slides-annotation
tags:
  - AI-MSc
  - TB1
  - FOAI
  - 线性回归
  - 梯度下降
  - 课堂标注
source: 课程材料（Conor Houghton、Roberto Colomboni）的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: draft
publication: pending
updated: 2026-10-09
---

# FOAI W02｜课堂 Slides 逐页标注（缺课自学版）

> [!IMPORTANT]
> **非官方学习笔记**：这是我在 University of Bristol「Foundations of Artificial Intelligence（SEMTM0051, 2026/27）」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：Conor Houghton、Roberto Colomboni。原始材料：[semtm0051/2026_27](https://github.com/semtm0051/2026_27/tree/322787dec91f174303ccf90399bb6243a060eb7d/)。经两位讲师同意公开；不含练习解答与非公开材料（如课堂录像）。

> [!NOTE]
> **这份笔记是什么**
> 逐页对应 [`Week_02/Slides/Week_02 - Slides.pdf`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_02/Slides/Week_02%20-%20Slides.pdf)（共 49 页），
> 把 PPT 上的英文术语翻成中文、补上直觉和简单证明。
> 它和 [FOAI-W02-线性回归](FOAI-公开版-W02-线性回归.md)（基于 Notes 讲义的教材式整理）互补：那份按逻辑脉络写，这份按你回看 PPT 的顺序写。
> 上课老师：**Dr Roberto Colomboni**，2026/27 学年 Lecture 2。

> [!TIP]
> **一个贯穿全课的套路**
> Colomboni 在这门课反复强调：**所有模型都是同一个三步流程**——
> ① 选函数族 $\{f_\theta\}$ → ② 选损失 $\ell$ → ③ 最小化经验损失 $\mathcal{L}(\theta)=\sum_i \ell(f_\theta(x_i),y_i)$。
> 今天线性回归是这个母版的第一个实例，后面分类、神经网络全是换损失、换函数族。

---

## P.0 标题页

- **Lecture 2: Linear regression**。本讲只讲回归（预测连续数值），不讲分类。
- 讲师：Dr Roberto Colomboni（三位老师之一，负责前几周的线性模型部分）。

## P.1–2 本讲目录

四节：
1. **Regression and loss functions**（回归与损失函数）
2. **Least-squares fitting**（最小二乘拟合）
3. **Gradient descent**（梯度下降）
4. **Extensions**（扩展：非线性特征、多元输入）

> [!NOTE]
> **自学路线建议**
> 第 1–2 节是"怎么算最优解"（闭式解），第 3 节是"算不出来时怎么办"（迭代法），第 4 节是"怎么让模型更强"。顺序不能跳。

---

## 第一节：Regression and loss functions（P.3–10）

### P.3 Regression: predicting a numerical outcome

我们观测到 $N$ 个输入-目标对 $(x_1,y_1),\dots,(x_N,y_N)\in\mathbb{R}^2$。

**模型设定**：
$$y_i = f_{\text{true}}(x_i) + \varepsilon_i, \qquad i=1,\dots,N$$

> [!NOTE]
> **名词解释**
> - **regression（回归）**：预测**连续数值**输出（如房价、温度），不是预测类别。
> - $f_{\text{true}}$：真实函数，**未知**，我们的任务是猜它。
> - $\varepsilon_i$（noise / 噪声项）：测量误差或未观测因素造成的扰动，**未知**。
> - 图中橙色竖线 = 噪声 $\varepsilon_i$，即真实函数值和观测点之间的差。

**关键**：$f_{\text{true}}$ 和 $\varepsilon_i$ 都 unknown（PPT 用红字标出）。我们只有 $(x_i,y_i)$。

### P.4 An affine relationship

先假设真实关系是**仿射的（affine）**：
$$f_{\text{true}}(x)=ax+b, \qquad y_i = ax_i+b+\varepsilon_i$$

> [!NOTE]
> **名词解释**
> - **affine（仿射）**：$ax+b$，即直线。严格说"linear（线性）"在数学里要求过原点（$b=0$），但 ML 里"linear model"习惯把 $b$ 也算进去。
> - **a: slope（斜率）**：$x$ 每增加 1，$y$ 变化多少。
> - **b: intercept / bias（截距/偏置）**：$x=0$ 时的预测值。ML 里常叫 **bias**。
> - 观测点**不需要**精确落在直线上，因为有 $\varepsilon_i$。

### P.5 Several possible prediction lines

我们不知道真实的 $a,b$。一般也没有一条直线能穿过所有点。所以问题变成：**从无数条候选直线里选哪条？**

PPT 举了三个候选（用 P.10 的同一组 5 个数据点）：
- $f(x)=1+2x$（斜率 2，截距 1）
- $g(x)=1-x$（斜率 $-1$，截距 1）
- $h(x)=2$（常数，斜率 0）

### P.6 Comparing prediction errors

怎么比较哪条线好？看**残差（residual）**：

> [!WARNING]
> **残差定义（全课统一，别搞反）**
> $$\text{residual} = \text{prediction} - \text{target} = f(x_i) - y_i$$
> **预测减目标**，不是目标减预测。正残差 = 高估，负残差 = 低估。

三个子图分别画了 $f, g, h$ 的残差竖线。肉眼看 $f$ 的竖线最短，但需要一个**数值标准**来客观比较。

### P.7 Loss functions

**loss function（损失函数）**：对单条预测 $\hat y$ 和目标 $y$ 打一个惩罚分。两个选择：

$$\ell_{\text{abs}}(\hat y,y)=|\hat y-y|, \qquad \ell_{\text{sq}}(\hat y,y)=(\hat y-y)^2$$

> [!NOTE]
> **名词解释**
> - $\hat y$（读作 y-hat）：模型的预测值。
> - **absolute loss / L1 loss**：绝对误差，图中橙色折线。
> - **squared loss / L2 loss**：平方误差，图中青色抛物线。
> - 右边图横轴是 residual $r=\hat y-y$，纵轴是 penalty（惩罚大小）。

**两个关键性质**（PPT 底部）：
1. 误差翻倍 → 绝对损失翻倍，平方损失翻**四倍**（因为 $r^2$）。所以平方损失对**大误差更敏感**。
2. 平方损失**处处可微（differentiable everywhere）**；绝对损失在 $r=0$ 处有个**尖角（kink）**，不可微。

> [!TIP]
> **为什么可微很重要**
> 后面要"求导找最小值"，不可微的点不能直接用微积分。这就是为什么本讲主要用平方损失。

### P.8 Empirical loss

单条损失不够，要把所有 $N$ 条加起来：

$$\mathcal{L}_{\text{abs}}(\theta)=\sum_{i=1}^N |f_\theta(x_i)-y_i|, \qquad \mathcal{L}_{\text{sq}}(\theta)=\sum_{i=1}^N (f_\theta(x_i)-y_i)^2$$

> [!NOTE]
> **名词解释**
> - **empirical loss（经验损失）**：在**已有数据集**上算出来的总损失。"empirical"=基于观测数据。
> - $\theta$（theta）：模型的参数集合（比如直线的 $b,a$）。
> - PPT 底部红字：**"The data stay fixed. The parameters vary."**——拟合时 $x_i,y_i$ 不动，我们只动 $\theta$。

### P.9 Choosing the loss

> [!IMPORTANT]
> **核心思想**
> **损失函数应该反映应用的目标（The loss should reflect the goal of the application.）**。
> - 平方损失：光滑、方便求导；对大误差惩罚重（适合 outliers 不多、想让大误差尽量小的场景）。
> - 绝对损失：在 0 处有折角，优化时需要特殊处理；对 outliers 更鲁棒。
> - **本讲聚焦平方损失（squared loss）**。

### P.10 Comparing the candidates numerically

用 P.5 的三个候选在同一组数据上算 SSE（sum of squared errors）：

| Observation | 1 | 2 | 3 | 4 | 5 | $\mathcal{L}_{\text{sq}}$ |
|---|---|---|---|---|---|---|
| Input $x_i$ | $-2$ | $-1$ | $0$ | $1$ | $2$ | |
| Target $y_i$ | $-1.5$ | $0.5$ | $1$ | $2.5$ | $7.5$ | |
| Residual of $f(x)=1+2x$ | $-1.5$ | $-1.5$ | $0$ | $0.5$ | $-2.5$ | **11** |
| Residual of $g(x)=1-x$ | $4.5$ | $1.5$ | $0$ | $-2.5$ | $-8.5$ | **101** |
| Residual of $h(x)=2$ | $3.5$ | $1.5$ | $1$ | $-0.5$ | $-5.5$ | **46** |

> [!WARNING]
> **重要提醒（PPT 底部红字）**
> 平方误差标准**更偏好 $f$**，但这只是在这三条候选里选最好的——**不代表 $f$ 是所有直线里最好的**。
> 下一步要做的是：不猜候选，直接从整个直线族里**算出**最优的那条。

---

## 第二节：Least-squares fitting（P.11–31）

### P.11 Finding a minimum

> [!NOTE]
> **名词解释**
> - **Fermat's theorem（费马引理）**：可微函数在局部最小值处，梯度为零。
> - **stationary point（驻点）**：梯度 $\nabla \mathcal{L}=0$ 的点。
> - 策略：先找驻点，再验证它是不是最小值。

PPT 下方三个小图展示三个越来越灵活的函数族：
- 常数：$f(x)=b$（只能上下平移）
- 过原点直线：$f(x)=ax$（只能绕原点转）
- 一般仿射：$f(x)=b+ax$（平移+旋转都自由）

我们按这个顺序逐个解。

### P.12 A constant predictor（最简单的族）

令 $a=0$，预测值恒为 $b$：
$$f_b^{\text{const}}(x)=b, \qquad \mathcal{L}_{\text{sq}}^{\text{const}}(b)=\sum_{i=1}^N (b-y_i)^2$$

左图：$b=1$ 时的预测线和残差；右图：损失作为 $b$ 的函数是一条 U 形抛物线，最低点在 $b^*$。

**问题**：怎么找 $b^*$？答案：$b^*$ 是损失函数的驻点。

### P.13 Differentiating the constant-predictor loss

定义残差 $r_i(b)=b-y_i$，则 $r_i'(b)=1$（因为 $y_i$ 是常数）。

用**链式法则（chain rule）**：外层 $u(r)=r^2$，内层 $r_i(b)$：

$$\frac{d}{db}\mathcal{L}(b)=\sum_{i=1}^N \frac{d}{db}r_i(b)^2=\sum_{i=1}^N \underbrace{2r_i(b)}_{\text{外层导数}}\cdot\underbrace{r_i'(b)}_{\text{内层导数}}=2\sum_{i=1}^N(b-y_i)=2\left(Nb-\sum_i y_i\right)$$

> [!TIP]
> **简单证明要点**
> 链式法则：$(r^2)' = 2r \cdot r'$。这里 $r_i(b)=b-y_i$，所以 $r_i'(b)=1$。目标 $y_i$ 不依赖 $b$，求导时当常数。

### P.14 The best constant is the sample mean

令导数为 0：
$$2\left(Nb-\sum_i y_i\right)=0 \implies b^*=\frac{1}{N}\sum_i y_i=\bar y$$

> [!IMPORTANT]
> **结论**
> **平方损失下，最佳常数预测 = 样本均值（sample mean）。**

**为什么这是全局最小？**
- $\mathcal{L}(b)=Nb^2-2b\sum y_i+\sum y_i^2$ 是关于 $b$ 的二次函数，首项系数 $N>0$，开口向上。
- 开口向上的抛物线只有一个驻点，那个驻点就是全局最小。

### P.15 The constant fit on our data

用 5 个数据点算：
$$b^*=\frac{-1.5+0.5+1+2.5+7.5}{5}=\frac{10}{5}=2$$

各点残差的平方和 $=3.5^2+1.5^2+1^2+(-0.5)^2+(-5.5)^2=12.25+2.25+1+0.25+30.25=46$。

> [!NOTE]
> **课堂练习（PPT 底部）**
> 保持常数族不变，把平方损失换成**绝对损失** $\sum|b-y_i|$，最佳 $b$ 变成什么？
> **答案：中位数（median）**。这就是讲义 Exercise 5。
> 这道题的意义：**"最佳"取决于你选什么损失**——平方损失→均值，绝对损失→中位数。

### P.16 A line through the origin

令 $b=0$，只有斜率 $a$ 可调：
$$f_a^{\text{slope}}(x)=ax, \qquad \mathcal{L}(a)=\sum_{i=1}^N(ax_i-y_i)^2$$

所有候选直线都**过原点**。

### P.17 Differentiating with respect to the slope

残差 $r_i(a)=ax_i-y_i$，$r_i'(a)=x_i$。链式法则：
$$\frac{d}{da}\mathcal{L}(a)=\sum_i 2r_i(a)r_i'(a)=2\sum_i(ax_i-y_i)x_i=2\left(a\sum_i x_i^2-\sum_i x_i y_i\right)$$

### P.18 The best slope

令导数为 0：
$$a\sum_i x_i^2=\sum_i x_i y_i \implies a^*=\frac{\sum_i x_i y_i}{\sum_i x_i^2}, \qquad \text{要求 }\sum_i x_i^2>0$$

> [!NOTE]
> **边界情况**
> 如果所有 $x_i=0$，那所有斜率给出相同预测（都是 0），损失都一样，没有唯一最优解。

同样是开口向上的抛物线，唯一驻点 = 全局最小。

### P.19 The fitted line through the origin

代入数据：
$$\sum x_i^2=(-2)^2+(-1)^2+0+1^2+2^2=10, \qquad \sum x_i y_i=(-2)(-1.5)+(-1)(0.5)+0\cdot1+1\cdot2.5+2\cdot7.5=3-0.5+0+2.5+15=20$$
$$a^*=\frac{20}{10}=2, \qquad \mathcal{L}(a^*)=26$$

对比：常数族最好损失 46，过原点直线最好损失 26。**函数族越灵活，拟合越好**。

### P.20 Fitting both slope and intercept

现在两个参数都自由。收集成列向量：
$$\theta=\begin{pmatrix}b\\a\end{pmatrix}, \qquad f_\theta(x)=b+ax, \qquad \mathcal{L}(b,a)=\sum_{i=1}^N(b+ax_i-y_i)^2$$

损失现在是**二元函数**，需要对两个变量分别求偏导。

### P.21 Affine loss: the intercept derivative（对 $b$ 求偏导）

残差 $r_i(b,a)=b+ax_i-y_i$，$\frac{\partial r_i}{\partial b}=1$。

对 $b$ 求偏导时**固定 $a$**：
$$\frac{\partial\mathcal{L}}{\partial b}=2\sum_i r_i\frac{\partial r_i}{\partial b}=2\sum_i(b+ax_i-y_i)=2\left(Nb+a\sum_i x_i-\sum_i y_i\right)$$

### P.22 Affine loss: the slope derivative（对 $a$ 求偏导）

$\frac{\partial r_i}{\partial a}=x_i$。固定 $b$：
$$\frac{\partial\mathcal{L}}{\partial a}=2\sum_i r_i\frac{\partial r_i}{\partial a}=2\sum_i x_i(b+ax_i-y_i)=2\left(b\sum_i x_i+a\sum_i x_i^2-\sum_i x_i y_i\right)$$

> [!TIP]
> **为什么偏导里 $\partial r_i/\partial b=1$ 而 $\partial r_i/\partial a=x_i$**
> 因为 $r_i=b+ax_i-y_i$。对 $b$ 求导：$b$ 的系数是 1；对 $a$ 求导：$a$ 的系数是 $x_i$。这就是 [W01](FOAI-公开版-W01-数学预备.md) 讲的偏导：求哪个变量的导，另一个当常数。

### P.23 The stationarity equations

令两个偏导都为 0，除以 2：

$$\begin{cases}
Nb+a\sum_i x_i=\sum_i y_i \\
b\sum_i x_i+a\sum_i x_i^2=\sum_i x_i y_i
\end{cases}$$

这是关于 $b,a$ 的**线性方程组（linear system）**。

### P.24–26 Solving the system

定义经验均值（empirical means）：
$$\bar x=\frac{1}{N}\sum_i x_i, \qquad \bar y=\frac{1}{N}\sum_i y_i$$

第一个方程两边除以 $N$：$b+a\bar x=\bar y$，即 $b=\bar y-a\bar x$。

代入第二个方程，整理得：
$$a\left(\sum_i x_i^2-N\bar x^2\right)=\sum_i x_i y_i-N\bar x\bar y$$

### P.27 The best slope and intercept

假设 $x_i$ 不全相等（分母为正）：

$$\boxed{a^*=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2}, \qquad b^*=\bar y-a^*\bar x}$$

> [!NOTE]
> **两个等价形式**
> $a^*$ 有两种写法，完全等价：
> $$a^*=\frac{\sum x_i y_i-N\bar x\bar y}{\sum x_i^2-N\bar x^2}=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}$$
> 右边的写法更直观：分子是**协方差**，分母是 $x$ 的**方差**（除以 $N$ 后就是相关系数 $\times$ 标准差比）。

**拟合线必过 $(\bar x,\bar y)$**——因为 $f(\bar x)=b^*+a^*\bar x=(\bar y-a^*\bar x)+a^*\bar x=\bar y$。

PPT 问：这是全局最小吗？下一页证明。

### P.28–29 Residuals at the stationary point → Global minimum

驻点处的残差 $r_i^*=b^*+a^*x_i-y_i$。驻点条件正好是：
$$\sum_i r_i^*=0, \qquad \sum_i x_i r_i^*=0$$

**证明是全局最小**：任取另一组参数 $(b^*+u,\ a^*+v)$，其残差为 $r_i^*+u+vx_i$。展开平方：

$$\mathcal{L}(b^*+u,a^*+v)=\sum_i(r_i^*+u+vx_i)^2=\underbrace{\mathcal{L}(b^*,a^*)}_{\text{原损失}}+2u\underbrace{\sum_i r_i^*}_{=0}+2v\underbrace{\sum_i x_i r_i^*}_{=0}+\underbrace{\sum_i(u+vx_i)^2}_{\ge 0}$$

交叉项因驻点条件消失，剩下 $\ge 0$。所以**任何其他参数的损失都不小于驻点损失**——$(b^*,a^*)$ 是全局最小。

> [!TIP]
> **简单证明的直觉**
> 这就是讲义里说的"展开 $L(b^*+u,a^*+v)$"。交叉项消失是因为驻点条件 $\sum r_i^*=0$ 和 $\sum x_i r_i^*=0$。最后一项 $\sum(u+vx_i)^2$ 是平方和，不可能为负。

### P.30 The affine fit on our data

数据 $\bar x=0,\ \bar y=2$：
$$a^*=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}=\frac{20}{10}=2, \qquad b^*=2-2\cdot0=2$$
$$f_{\theta^*}(x)=2+2x, \qquad \mathcal{L}=6$$

**三个族的损失对比**：常数 46 > 过原点直线 26 > 仿射直线 6。越灵活越好（在训练数据上）。

---

## 第三节：Gradient descent（P.31–40）

### P.31 An iterative approach

仿射模型能直接解出最优解，但**更复杂的模型没有闭式解（explicit solution）**。怎么办？

> [!NOTE]
> **名词解释**
> - **closed-form / explicit solution（闭式解）**：能用公式直接写出来的解，比如 $a^*=...$。
> - **iterative approach（迭代法）**：从一个初始猜测开始，一步一步改进参数。

我们从 $\theta^{(0)}$ 出发，得到 $\theta^{(1)},\theta^{(2)},\dots$，每步用**梯度（gradient）**决定往哪走。

### P.32 A local linear approximation

对可微函数，在 $\theta$ 附近做一阶泰勒展开：
$$\mathcal{L}(\theta+\Delta)\approx\mathcal{L}(\theta)+\nabla\mathcal{L}(\theta)^\top\Delta$$

如果选 $\Delta=-\eta\nabla\mathcal{L}(\theta)$（$\eta>0$ 是步长），则：
$$\mathcal{L}(\theta+\Delta)\approx\mathcal{L}(\theta)-\eta\|\nabla\mathcal{L}(\theta)\|_2^2<\mathcal{L}(\theta)$$

> [!IMPORTANT]
> **为什么朝负梯度走**
> 因为 $\|\nabla\mathcal{L}\|^2\ge0$，减去一个非负数 → 损失下降。
> 前提：梯度非零、步长**足够小**（一阶近似只在小步时成立）。

### P.33–34 The gradient descent update

$$\boxed{\theta^{(t+1)}=\theta^{(t)}-\eta\,\nabla\mathcal{L}(\theta^{(t)})}$$

> [!NOTE]
> **名词解释**
> - **gradient descent（梯度下降）**：沿负梯度方向走一小步，重复。
> - **learning rate / step size（学习率/步长）**：$\eta$，控制每步走多大。
> - PPT 把两个偏导合写成向量梯度，两个参数同时更新。

### P.35–40 一步步算（用同一组 5 个数据点，$\eta=0.04$）

**初始点** $\theta^{(0)}=(b^{(0)},a^{(0)})=(0,0)$：
- 预测全是 0，残差 $r_i^{(0)}=0-y_i=-y_i$
- $\sum r_i^{(0)}=-(−1.5+0.5+1+2.5+7.5)=-10$
- $\sum x_i r_i^{(0)}=\sum x_i(-y_i)=-20$
- 梯度 $=2\begin{pmatrix}\sum r_i\\\sum x_i r_i\end{pmatrix}=2\begin{pmatrix}-10\\-20\end{pmatrix}=\begin{pmatrix}-20\\-40\end{pmatrix}$
- 损失 $\mathcal{L}(\theta^{(0)})=\sum y_i^2=(-1.5)^2+0.5^2+1+2.5^2+7.5^2=2.25+0.25+1+6.25+56.25=66$

**第 1 步**：
$$\theta^{(1)}=\begin{pmatrix}0\\0\end{pmatrix}-0.04\begin{pmatrix}-20\\-40\end{pmatrix}=\begin{pmatrix}0.8\\1.6\end{pmatrix}$$
新直线 $f(x)=0.8+1.6x$，损失降到 **14.8**。

**第 2 步**：$\theta^{(2)}=(1.28,1.92)$，损失 8.656。
**第 5 步**：$\theta^{(5)}=(1.84448,1.99936)$，损失 6.12094，已经很接近解析解 $(2,2)$，损失 6。

> [!TIP]
> **右侧等高线图怎么看**
> PPT 右侧是参数平面 $(b,a)$ 上的损失等高线，最优解在椭圆中心 $(2,2)$。橙色折线就是梯度下降走的路径——从 $(0,0)$ 出发，一步步螺旋逼近中心。注意路径不是直线走向中心，因为等高线是椭圆（不是正圆），梯度方向垂直于等高线，所以走出来是锯齿状。

> [!WARNING]
> **步长太大不收敛**
> 讲义里举的例子：$F(a)=100a^2$，$\eta=0.1$ 时更新为 $a^{(t+1)}=a^{(t)}-0.1\cdot200a^{(t)}=-19a^{(t)}$，**越跳越远、发散**。所以 $\eta$ 要小到"充分小"。

---

## 第四节：Extensions（P.41–48）

### P.41 Extension 1: nonlinear predictors

之前预测函数都是 $x$ 的仿射函数。能不能用 $x$ 的非线性函数？

例子：二次函数
$$f_\theta(x)=b+w_1x+w_2x^2, \qquad \theta=(b,w_1,w_2)^\top\in\mathbb{R}^{3\times1}$$

引入**特征映射（feature map）** $\phi(x)=(1,x,x^2)$，则 $f_\theta(x)=\phi(x)\theta$。

> [!IMPORTANT]
> **核心概念：什么叫"线性模型"**
> 预测函数对 $x$ 可以是非线性的（$x^2$），但**对参数 $\theta$ 是线性的**。
> 所以正规方程、梯度下降**原样适用**，只是把设计矩阵 $X$ 换成 $\Phi$。
> 反例：$f(x)=\sin(wx)$ 对 $w$ 非线性，不能写成 $\phi(x)\theta$，不属于线性模型。

### P.42 A quadratic fit to the same observations

对同一组 5 个点，用 $\phi(x)=(1,x,x^2)$：
$$\theta^*=(1,2,0.5)^\top, \qquad f(x)=1+2x+0.5x^2$$
损失降到 **2.5**（仿射直线是 6）。

### P.43 Fixed feature expansion

一般化：选固定的特征函数 $\phi_1,\dots,\phi_m:\mathbb{R}\to\mathbb{R}$，构造特征矩阵
$$\Phi\in\mathbb{R}^{N\times m}, \qquad \Phi_{ij}=\phi_j(x_i)$$
预测 $f_\theta(x)=\phi(x)\theta=\sum_j\theta_j\phi_j(x)$。

### P.44 Feature expansion: fitting the coefficients

损失 $\mathcal{L}(\theta)=\|\Phi\theta-y\|_2^2$，梯度 $\nabla\mathcal{L}=2\Phi^\top(\Phi\theta-y)$。

若 $\Phi^\top\Phi$ 可逆：
$$\theta^*=(\Phi^\top\Phi)^{-1}\Phi^\top y$$

梯度下降：
$$\theta^{(t+1)}=\theta^{(t)}-2\eta\Phi^\top(\Phi\theta^{(t)}-y)$$

### P.45 Extension 2: several input variables

之前 $x$ 是标量。实际问题里一个观测可能有多个输入特征（比如用年龄、教育年限、工作经验预测薪水）：
$$\mathbf{x}=(x_1,\dots,x_d)\in\mathbb{R}^{1\times d}$$

每个特征配一个权重 $w_j$，再加一个偏置 $b$：
$$f_\theta(\mathbf{x})=b+\sum_{j=1}^d x_j w_j$$

### P.46 The design matrix

把所有 $N$ 个观测堆成设计矩阵（design matrix）：
$$\mathbf{X}=\begin{pmatrix}1&\mathbf{x}_1\\\vdots&\vdots\\1&\mathbf{x}_N\end{pmatrix}\in\mathbb{R}^{N\times(d+1)}$$

> [!NOTE]
> **名词解释**
> - **design matrix（设计矩阵）**：每行一个观测，每列一个特征；第一列全是 1（用来吸收截距 $b$）。
> - 预测值向量 $=X\theta$，残差 $r=X\theta-y$。

### P.47–48 Solution and gradient descent

和单输入完全一样，只是 $X$ 变宽了：

$$\nabla\mathcal{L}=2X^\top(X\theta-y), \qquad \boxed{\theta^*=(X^\top X)^{-1}X^\top y}$$

梯度下降：$\theta^{(t+1)}=\theta^{(t)}-2\eta X^\top(X\theta^{(t)}-y)$。

> [!WARNING]
> **形状别搞反（考试常考）**
> - $X^\top X$ 是 $(d+1)\times(d+1)$——正规方程里的矩阵。
> - $XX^\top$ 是 $N\times N$——**不是**正规方程里的东西。
> - $X^\top X$ 可逆的条件：$X$ 的列线性无关（即 $N\ge d+1$ 且特征之间没有完全共线）。证明见讲义附录 A。

### P.49 结尾页

- 讲义和习题在课程 GitHub：`github.com/semtm0051/2026_27`
- 习题解答会在下一次课前公布。

---

## 本讲自测（缺课补完后检查）

- [ ] 能默写：残差定义（预测减目标）、平方损失 vs 绝对损失的区别。
- [ ] 能手算：5 个数据点的常数最优值（均值=2）、过原点斜率（=2）、仿射参数（$b^*=2,a^*=2$）。
- [ ] 能解释：为什么 $b=\bar y$ 是平方损失的最优常数？换绝对损失为什么变成中位数？
- [ ] 能推导：$\frac{\partial\mathcal{L}}{\partial b}=2\sum r_i$、$\frac{\partial\mathcal{L}}{\partial a}=2\sum x_i r_i$。
- [ ] 能说：梯度下降为什么朝负梯度走？步长太大会怎样？
- [ ] 能说：为什么 $f(x)=1+2x+0.5x^2$ 还叫"线性模型"？
- [ ] 能写：多元设计矩阵 $X$ 的形状、正规方程 $X^\top X\theta=X^\top y$。

## 跨周联系

- 数学工具：[FOAI-W01-数学预备](FOAI-公开版-W01-数学预备.md)（梯度、链式法则、偏导）。
- 代码实现：[FOAI-W03-PyTorch线性回归实验](FOAI-公开版-W03-PyTorch线性回归实验.md)（用 PyTorch 写设计矩阵、闭式解、梯度下降）。
- 下一步：Week 3 Linear Classification——把平方损失换成逻辑损失，把 $y\in\mathbb{R}$ 换成 $y\in\{0,1\}$。
- 概率视角：平方损失 ↔ 高斯噪声，交叉熵 ↔ 伯努利，后续联系 SCEM。
