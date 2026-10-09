---
title: FOAI W03｜课堂 Slides 逐页标注（零基础版）
course: FOAI
week: 3
type: slides-annotation
tags:
  - AI-MSc
  - TB1
  - FOAI
  - 线性分类
  - logistic回归
  - softmax
  - 课堂标注
source: 课程材料（Conor Houghton、Roberto Colomboni）的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: published
publication: public
updated: 2026-10-09
---

# FOAI W03｜课堂 Slides 逐页标注（零基础版）

> [!IMPORTANT]
> **非官方学习笔记**：这是我在 University of Bristol「Foundations of Artificial Intelligence（SEMTM0051, 2026/27）」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：Conor Houghton、Roberto Colomboni。原始材料：[semtm0051/2026_27](https://github.com/semtm0051/2026_27/tree/322787dec91f174303ccf90399bb6243a060eb7d/)。经两位讲师同意公开；不含练习解答与非公开材料（如课堂录像）。

> [!NOTE]
> **这份笔记是什么**
> 逐页对应 [`Week_03/Slides/Week_03 - Slides.pdf`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_03/Slides/Week_03%20-%20Slides.pdf)（共 52 页，**2026-10-08 晚更新后的最新版**）。把页面上的英文翻成中文，补上直觉、公式拆解和易错点。
> 和 [FOAI-W03-线性分类](FOAI-公开版-W03-线性分类.md) 互补：那份按讲义的逻辑写，这份按你**上课看 Slides 的顺序**写。
> 这里写的是对 Slides 文字和讲义的解释，不是课堂口述的记录。标注为「补充说明」「自编例子」的方框不是讲义或 Slides 的原文。
> 数值我都用代码核对过（见各页和文末）。

> [!WARNING]
> **来源冲突｜记号**
> 这套 Slides 更新后把输入写成 $\mathbb R^{d}$（第 45–51 页），而讲义用行向量 $\mathbb R^{1\times d}$。含义相同，只是写法不同。详见 [FOAI-W03-线性分类](FOAI-公开版-W03-线性分类.md) 的来源冲突标注。

---

## 一、明早 10 分钟速览

**这一讲的主线**（记住这条链就够了）：

> 给每个输入算一个**打分** $s=b+\boldsymbol x\boldsymbol w$ → 用 **sigmoid** 把打分变成「属于类 1 的概率」$p$ → 概率 $>1/2$ 判类 1，否则判类 0 → 想找好参数，但「数错误个数」没法做梯度下降 → 改用 **logistic loss**（光滑）代替 → 用**梯度下降**最小化它 → 推广到多类：**softmax + 交叉熵**。

**必须认识的 6 个式子**

| 名字 | 式子 | 一句话 |
| --- | --- | --- |
| 打分 | $s=b+\boldsymbol x\boldsymbol w$ | 仿射函数，可以是任何实数 |
| sigmoid | $\sigma(s)=\dfrac{1}{1+e^{-s}}$ | 把任何实数压到 $(0,1)$，$\sigma(0)=\tfrac12$ |
| 预测规则 | $p>\tfrac12\Rightarrow$ 类 1；否则类 0 | **平局（$p=\tfrac12$）归类 0**，这是最近改过的 |
| logistic loss | $\ell_{\log}(p,y)=-y\log p-(1-y)\log(1-p)$ | 给「真实类」分配的概率越低，罚得越重 |
| 梯度 | $\nabla\mathcal L_{\log}=\boldsymbol X^\top(\boldsymbol p-\boldsymbol y)$ | 「预测概率 − 真实标签」乘设计矩阵的转置 |
| softmax | $\mathrm{softmax}(\boldsymbol s)_k=\dfrac{e^{s_k}}{\sum_r e^{s_r}}$ | 多类时把 $K$ 个打分变成和为 1 的概率 |

**3 个数字**（都核对过）：打分为 0 时概率正好是 $0.5$；初始参数 $(-5.75,0,1)$ 在 100 个点上错 **31** 个；用 $\eta=0.002$ 梯度下降时**第一步损失就从 51.048 涨到 64.556**（步长太大）。

**3 个警告**（老师在 Slides 上明确写出）：
1. 只数错误个数，梯度是 0，梯度下降**一步都不会动**（第 14 页）。
2. logistic loss 变小，**错误个数不一定变少**（第 29、36 页）。
3. 学习率太大，损失可能变大（第 29、43 页）。

---

## 二、先修知识速查（基础薄弱先看这一节）

> [!TIP]
> **补充说明（非讲义原文）：下面几样东西，后面每一页都会用到。**

### 2.1 对数与指数

- $e\approx2.718$，$e^{0}=1$；$e^{-s}$ 在 $s$ 很大时接近 0，在 $s$ 很负时变得很大。
- $\log$ 在本课里**一律是自然对数**（以 $e$ 为底）。$\log 1=0$；$\log p$ 对 $0<p<1$ 是负数；$p$ 越小，$-\log p$ 越大。
- 几个会用到的值：$-\log 0.49\approx0.713$，$-\log0.1\approx2.303$，$-\log0.01\approx4.605$，$\log2\approx0.693$。

### 2.2 sigmoid 的取值

$\sigma(s)=\dfrac{1}{1+e^{-s}}$：

| $s$ | $-4$ | $-2$ | $0$ | $2$ | $4$ |
| --- | --- | --- | --- | --- | --- |
| $\sigma(s)$ | 0.018 | 0.119 | 0.500 | 0.881 | 0.982 |

打分越正，概率越接近 1；越负，越接近 0；0 对应 $0.5$。另一个常用事实：$1-\sigma(s)=\sigma(-s)$。

### 2.3 导数、偏导数、梯度

- **导数**：函数在某点变化有多快。$\dfrac{d}{dp}(-\log p)=-\dfrac1p$。
- **偏导数**：函数有多个参数时，**只让一个参数变，其他固定**，看函数怎么变。
- **梯度**：把对每个参数的偏导排成一列，指向函数**上升最快**的方向。梯度下降就是沿**反方向**小步走：$\boldsymbol\theta\leftarrow\boldsymbol\theta-\eta\nabla\mathcal L(\boldsymbol\theta)$，$\eta>0$ 是学习率。
- 如果一个函数在某点附近**是平的**（局部常数），那里导数是 0，梯度下降不会动。

### 2.4 链式法则（第 23–27 页就是在用它）

若 $y=g(u)$，而 $u=h(x)$，则 $\dfrac{dy}{dx}=g'(u)\cdot h'(x)$：**外层导数 × 内层导数**。
三层嵌套 $G(\sigma(F(\theta)))$ 就是连乘三个导数：

$$\frac{\partial}{\partial\theta}G(\sigma(F(\theta)))=G'(\sigma(F))\cdot\sigma'(F)\cdot F'(\theta).$$

> [!NOTE]
> **自编例子（非讲义）：用一个观测亲手验证链式法则**
> 取 $d=1$，一个观测 $x=2,\ y=1$，参数 $\boldsymbol\theta=(b,w)=(0,\ 0.5)$。
> - 打分 $s=b+wx=1$；概率 $p=\sigma(1)\approx0.7311$；损失 $\ell=-\log p\approx0.3133$。
> - 三项导数：$\dfrac{\partial\ell}{\partial p}=\dfrac{p-y}{p(1-p)}\approx-1.3679$；$\sigma'(s)=p(1-p)\approx0.1966$；$\dfrac{\partial s}{\partial(b,w)}=(1,\ x)=(1,\ 2)$。
> - 连乘：$-1.3679\times0.1966\times(1,2)\approx(-0.2689,\ -0.5379)$。
> - 公式 $(p-y)\boldsymbol X_i=(0.7311-1)\times(1,2)=(-0.2689,\ -0.5379)$。**一致**。我也用数值微分验证过：$(-0.2689,\ -0.5379)$。
> 这说明：三项连乘之后，$p(1-p)$ 恰好消掉了，只剩下漂亮的「$(p-y)\times$ 输入」。

### 2.5 矩阵与向量写法

- 一个观测的输入写成**行** $\boldsymbol x_i=(x_{i1},\dots,x_{id})$；参数写成**列** $\boldsymbol w$。$\boldsymbol x\boldsymbol w$ 是一个数。
- **设计矩阵** $\boldsymbol X$：每行是 $(1,\boldsymbol x_i)$，第一列全 1，让截距 $b$ 也成为参数的一部分。这样 $\boldsymbol X\boldsymbol\theta$ 一次算出全部 $N$ 个打分。

---

## 三、逐页标注

### 开场（第 1–2 页）

**P.1 标题页**：Lecture 3: Linear classification，讲师 Dr Roberto Colomboni。

**P.2 目录**：四块：① Binary classification（二分类）② Classification error and logistic loss（分类误差与 logistic 损失）③ Gradient descent（梯度下降）④ More than two classes（多于两类）。第 11、20、44 页是同样目录页，用来分隔四块。

### 第一块：二分类（第 3–10 页）

**P.3 Binary classification（二分类）**
- 数据：一个合成数据集（改编自 Iris），**100 个观测**，每个有两个输入特征 $x_{i1},x_{i2}$，标签 $y_i\in\{0,1\}$。类 0 与类 1 各 **50 个**。
- 任务：学一条规则，**预测新输入属于哪一类**。数据与说明在 `Week_03/Examples`。
- 图里横轴是 $x_1$，纵轴是 $x_2$，颜色/形状区分类别。**类别用颜色表示，不是高度。**

**P.4 Measuring classification error（衡量分类误差）**
对比回归和分类：

| | 回归 | 分类 |
| --- | --- | --- |
| 预测器 | $f:\mathbb R^d\to\mathbb R$（返回一个数） | $h:\mathbb R^d\to\{0,1\}$（返回类别标签） |
| 单个损失 | $\ell_{\rm sq}(\hat y,y)=(\hat y-y)^2$ | $\ell_{0,1}(\hat y,y)=\begin{cases}0,&\hat y=y\\1,&\hat y\ne y\end{cases}$ |
| 求和后 | 经验平方损失：误差大罚得重 | **错误个数**（经验 0-1 损失）：每个错误罚得一样 |

要点：0-1 损失**只关心对错**，不关心错得多离谱。

**P.5 Predicting a probability first（先预测概率）**
- 标签只给出一个决定；**概率还记录模型有多倾向于这个标签**。
- 概率预测器 $p:\mathbb R^d\to[0,1]$：$p(\boldsymbol x)$ 是类 1 的概率，$1-p(\boldsymbol x)$ 是类 0 的概率。
- 例：$p=0.51$ 表示 51% 认为是类 1、49% 认为是类 0。问：如果必须选一个标签，选哪个？

**P.6 The plug-in classification rule（plug-in 分类规则）**
- 选概率更大的标签。**平局归类 0**：

$$h_p(\boldsymbol x)=\begin{cases}0,&p(\boldsymbol x)\le1/2,\\1,&p(\boldsymbol x)>1/2.\end{cases}$$

- **Heaviside 阶跃函数** $\mathrm{step}(s)=\begin{cases}0,&s\le0\\1,&s>0\end{cases}$。于是 $h_p(\boldsymbol x)=\mathrm{step}\big(p(\boldsymbol x)-\tfrac12\big)$。
- 注意阶跃函数的 $\mathrm{step}(0)=0$：**恰好等于 0 的打分归类 0**。这是最近讲义改的地方（旧版平局归类 1）。
- 留下问题：概率函数 $p$ 怎么选？

**P.7 How do we build probability predictions?（怎么构造概率预测）**
- 沿用回归：截距加每个特征一个权重，$f_{\boldsymbol\theta}(\boldsymbol x)=b+\boldsymbol x\boldsymbol w=b+\sum_{j=1}^{d}x_jw_j$，参数列 $\boldsymbol\theta=(b,w_1,\dots,w_d)^\top\in\mathbb R^{d+1}$。
- 问题：$f_{\boldsymbol\theta}(\boldsymbol x)$ **可以是任意实数**，不能直接当概率。所以把它叫做 **打分（score，也叫 logit）**，下一步再把它变成概率。

**P.8 The sigmoid function（sigmoid 函数）**
- $\sigma(s)=\dfrac{1}{1+e^{-s}}=\dfrac{e^s}{1+e^s}$，光滑、单调增，$0<\sigma(s)<1$，$\sigma(0)=\tfrac12$。
- 图：横轴打分 $s\in[-6,6]$，纵轴概率；S 形曲线。

**P.9 Probabilities, labels and the boundary（概率、标签与决策边界）**
- 概率模型 $p_{\boldsymbol\theta}(\boldsymbol x)=\sigma(f_{\boldsymbol\theta}(\boldsymbol x))=\sigma(b+\boldsymbol x\boldsymbol w)$。
- 关键等价链：

$$h_{p_{\boldsymbol\theta}}(\boldsymbol x)=1\iff p_{\boldsymbol\theta}(\boldsymbol x)>\tfrac12\iff f_{\boldsymbol\theta}(\boldsymbol x)>0\iff b+\boldsymbol x\boldsymbol w>0.$$

  **含义**：打分的正负就决定类别，sigmoid 只管「置信度」。
- **决策边界**：所有满足 $b+\boldsymbol x\boldsymbol w=0$ 的 $\boldsymbol x$。只要 $\boldsymbol w\ne\boldsymbol 0$，它就是一个**超平面**：
  - $d=1$：一个**点**；
  - $d=2$：一条**直线**；
  - $d=3$：一个**平面**。

> [!NOTE]
> **自编例子（非讲义）：$d=1$**
> 取 $b=-3,\ w=1.5$：打分 $s=-3+1.5x$。边界是 $-3+1.5x=0$，即 $x=2$。$x>2$ 时打分为正，判类 1；$x\le2$ 判类 0（$x=2$ 恰好打分为 0，按新规则归类 0）。

**P.10 Comparing two candidate boundaries（比较两条候选边界）**
- 问「哪个分类器更好？」，答「**数错误个数**」。
- 左：$\boldsymbol\theta=(-5.75,\ 0,\ 1)^\top$，边界 $x_2=5.75$；右：$\boldsymbol\theta=(-1,\ -0.6,\ 1)^\top$，边界 $x_2=0.6x_1+1$。
- 页面结论：右边犯的错更少。我用 100 个观测复算：左边错 **31** 个，右边错 **9** 个（页面上没有写这两个数）。

> [!TIP]
> **补充说明：边界方程怎么来的**
> $b+w_1x_1+w_2x_2=0$ 解出 $x_2=-\dfrac{b+w_1x_1}{w_2}$。左：$-(-5.75+0\cdot x_1)/1=5.75$；右：$-(-1-0.6x_1)/1=1+0.6x_1$。

### 第二块：分类误差与 logistic loss（第 11–19 页）

**P.11**：目录页，进入第二块。

**P.12 Finding the parameters（找参数）**
- 目标：找 $\boldsymbol\theta^*$ 使错误个数最小：$\mathcal E(\boldsymbol\theta)=\sum_{i=1}^{N}\ell_{0,1}\big(h_{p_{\boldsymbol\theta}}(\boldsymbol x_i),y_i\big)$。
- 这是**组合问题**（要在离散的分类方案里挑）。
- 问：能不能像回归那样用梯度下降 $\boldsymbol\theta^{(t+1)}=\boldsymbol\theta^{(t)}-\eta\nabla\mathcal E(\boldsymbol\theta^{(t)})$？先看看边界稍微移动时错误数会怎么变。

**P.13 Moving the boundary and counting mistakes（移动边界并数错误）**
- 固定 $w_1=0,w_2=1$，只变截距 $b$：直线**平行移动**。右图画错误个数随 $b$ 的变化，$b$ 从 $-7$ 到 $-3$。
- 结论（原话大意）：**错误个数是局部常数，只有当直线扫过一个观测点时才变**。

> [!NOTE]
> **我复算的几个点（页面上没有逐点标出）**
> $b=-7$：错 50；$b=-6$：37；$b=-5.75$：31；$b=-5.55$：25；$b=-5$：8；$b=-4.5$：17；$b=-3$：49。可以看到：它是**阶梯状**的，两个阶梯之间是水平的一段。

**P.14 Flat regions give no direction（平坦区域没有方向）**
- 若每个观测的打分都不等于 0，则参数小幅变动**不会改变任何打分的符号**，所以所有预测标签不变：$\mathcal E(\boldsymbol\theta+\boldsymbol\Delta)=\mathcal E(\boldsymbol\theta)$，$\boldsymbol\Delta$ 足够小。
- 局部常数函数的梯度是 0：$\nabla\mathcal E(\boldsymbol\theta)=\boldsymbol0$。
- 于是 $\boldsymbol\theta^{(t+1)}=\boldsymbol\theta^{(t)}-\eta\cdot\boldsymbol0=\boldsymbol\theta^{(t)}$：**梯度下降不会动！！！**

> [!NOTE]
> **这一页是整讲最重要的动机**
> 想用梯度下降，就需要一个**光滑**的目标函数，让每个小变化都被「感觉到」。这就是下面要换成 logistic loss 的原因。

**P.15 What does the mistake count miss?（错误个数漏掉了什么）**
假设真实类是 0，两个预测都选了类 1：

| | 预测 A | 预测 B |
| --- | --- | --- |
| 类 1 的概率 | 0.51 | 0.90 |
| 预测类 | 1 | 1 |
| 0-1 损失 | 1 | 1 |
| **真实类**（类 0）获得的概率 | **0.49** | **0.10** |

- 0.51：模型几乎在瞎猜，希望罚得**轻**。
- 0.90：模型「自信地错了」，希望罚得**重**。
- 所以别只看预测的标签，而要**惩罚「给真实类分了低概率」**。

**P.16 Logistic loss: a surrogate for fitting probabilities（logistic 损失：拟合概率用的替代损失）**
- 对 $0<p<1$，$y\in\{0,1\}$，取「分给真实类的概率」的**负对数**：

$$\ell_{\log}(p,y)=-y\log p-(1-y)\log(1-p)=\begin{cases}-\log p,&y=1,\\-\log(1-p),&y=0.\end{cases}$$

- 标签 $y=1$ 时只剩第一项，$y=0$ 时只剩第二项。
- 例（真实类 0）：$\ell_{\log}(0.51,0)=-\log0.49\approx0.713$（错但不自信）；$\ell_{\log}(0.90,0)=-\log0.10\approx2.303$（自信地错）。我核对过这两个数。
- Logistic loss 也叫**二元交叉熵**。**用自然对数**。
- **surrogate（替代损失）**：我们真正关心的是错误个数，但它不光滑；logistic loss 是一个光滑的「替身」。

**P.17 The empirical logistic loss（经验 logistic 损失）**
- 观测固定，**概率预测随参数变**：$p_{\boldsymbol\theta}(\boldsymbol x_i)=\sigma(b+\boldsymbol x_i\boldsymbol w)$。
- $\mathcal L_{\log}(\boldsymbol\theta)=\sum_{i=1}^{N}\ell_{\log}\big(p_{\boldsymbol\theta}(\boldsymbol x_i),y_i\big)$（**求和，不是平均**）。
- 用梯度下降拟合这个光滑目标，而**错误个数 $\mathcal E(\boldsymbol\theta)$ 仍然是我们衡量分类好坏的指标**。

**P.18 Comparing the two losses（两个损失的比较）**
- **一个错误**意味着「真实类的概率至多 $1/2$」，所以它的 logistic loss **至少是** $-\log\tfrac12=\log2$。
- 图：左 $y=0$，右 $y=1$，横轴打分 $s$。画的是 **logistic loss ÷ $\log2$** 与 **0-1 损失**。logistic loss ÷ $\log2$ 的曲线**永远在 0-1 损失的上方**。
- 对每个观测：

$$\ell_{0,1}\Big(\mathrm{step}\big(\sigma(s)-\tfrac12\big),\,y\Big)\le\frac{\ell_{\log}(\sigma(s),y)}{\log2}.$$

**P.19 Bounding the number of mistakes（给错误个数设上界）**
把上式对全体观测求和：

$$\mathcal E(\boldsymbol\theta)=\sum_{i=1}^N\ell_{0,1}(\cdots)\le\frac{1}{\log2}\sum_{i=1}^N\ell_{\log}\big(p_{\boldsymbol\theta}(\boldsymbol x_i),y_i\big)=\frac{1}{\log2}\mathcal L_{\log}(\boldsymbol\theta).$$

**结论**：控制住 logistic loss，就控制住了错误个数。

> [!NOTE]
> **用上面的数验证（我的复算）**
> 初始参数 $(-5.75,0,1)$：$\mathcal L_{\log}=51.048$，上界 $51.048/0.693\approx73.6$，实际错误 31，满足 $31\le73.6$。$\eta=0.001$ 迭代 20 步后：损失 39.992，上界 $\approx57.7$，实际错误 7。

> [!WARNING]
> **易错点**
> 这是**单向**不等式：损失小 ⇒ 错误数有上界，**但不是反过来**，也不是等式。错误数可以比上界小很多。

### 第三块：梯度下降（第 20–43 页）

**P.20**：目录页，进入第三块。

**P.21 The design matrix and the parameters（设计矩阵与参数）**
- 和 W02 一样，加一列全 1 来包含截距：

$$\boldsymbol X=\begin{pmatrix}1&x_{11}&\cdots&x_{1d}\\\vdots&\vdots&&\vdots\\1&x_{N1}&\cdots&x_{Nd}\end{pmatrix}\in\mathbb R^{N\times(d+1)},\qquad\boldsymbol\theta=\begin{pmatrix}b\\w_1\\\vdots\\w_d\end{pmatrix}\in\mathbb R^{(d+1)\times1}.$$

- 对应关系：$\theta_1=b$，$\theta_{k+1}=w_k$。**第一个分量是截距，其余是权重。**

**P.22 The three functions（三个函数）**
固定第 $i$ 个观测 $(\boldsymbol x_i,y_i)$，变的是参数 $\boldsymbol\theta$。把单个观测的损失拆成三层：

| 函数 | 定义 | 作用 |
| --- | --- | --- |
| $F_i(\boldsymbol\theta)$ | $f_{\boldsymbol\theta}(\boldsymbol x_i)=b+\boldsymbol x_i\boldsymbol w=(1,\boldsymbol x_i)\boldsymbol\theta=\sum_{r=1}^{d+1}X_{ir}\theta_r$ | 参数 → 打分 |
| $\sigma(s)$ | $1/(1+e^{-s})$ | 打分 → 概率 |
| $G_i(p)$ | $\ell_{\log}(p,y_i)$ | 概率 → 损失 |

**P.23 The chain rule for one observation（单个观测的链式法则）**
- 第 $i$ 个观测的损失是三个函数的复合：$\ell_{\log}(p_{\boldsymbol\theta}(\boldsymbol x_i),y_i)=G_i\circ\sigma\circ F_i(\boldsymbol\theta)$。
- 对参数 $\theta_j$ 求偏导，用链式法则连乘三项：

$$\frac{\partial}{\partial\theta_j}\ell_{\log}=\underbrace{G_i'(\sigma(F_i))}_{\partial\ell_{\log}/\partial p}\cdot\underbrace{\sigma'(F_i)}_{\sigma'(f_{\boldsymbol\theta}(\boldsymbol x_i))}\cdot\underbrace{\frac{\partial F_i}{\partial\theta_j}}_{\partial f_{\boldsymbol\theta}(\boldsymbol x_i)/\partial\theta_j}.$$

**接下来三页分别算这三项。**（对照第 2.4 节的自编例子，数一遍就懂。）

**P.24 First term：对 logistic loss 求导（对 $p$）**
固定 $y$，$\ell_{\log}(p,y)=-y\log p-(1-y)\log(1-p)$：

$$\frac{\partial\ell_{\log}}{\partial p}=-\frac{y}{p}+\frac{1-y}{1-p}=\frac{-y(1-p)+(1-y)p}{p(1-p)}=\frac{p-y}{p(1-p)}.$$

页面上的化简步骤：通分、展开、$-y+yp+p-yp=p-y$。

> [!TIP]
> **补充说明：验证**
> $y=1$ 时得 $\dfrac{p-1}{p(1-p)}=-\dfrac1p$；$y=0$ 时得 $\dfrac{p}{p(1-p)}=\dfrac1{1-p}$。这和 $-\log p$、$-\log(1-p)$ 的导数一致。

**P.25 Second term：对 sigmoid 求导**
把 $\sigma$ 看作两个函数复合，**再用一次链式法则**：内层 $u(s)=1+e^{-s}$，$u'(s)=-e^{-s}$；外层 $v(z)=1/z$，$v'(z)=-1/z^2$。

$$\sigma'(s)=v'(u(s))\,u'(s)=-\frac{1}{(1+e^{-s})^2}\cdot(-e^{-s})=\frac{e^{-s}}{(1+e^{-s})^2}=\frac{1}{1+e^{-s}}\cdot\Big(1-\frac{1}{1+e^{-s}}\Big)=\sigma(s)\big(1-\sigma(s)\big).$$

所以 $\sigma'(f_{\boldsymbol\theta}(\boldsymbol x_i))=p_{\boldsymbol\theta}(\boldsymbol x_i)\big(1-p_{\boldsymbol\theta}(\boldsymbol x_i)\big)$。

**P.26 Third term：对仿射打分求导**
$f_{\boldsymbol\theta}(\boldsymbol x_i)=\sum_{r=1}^{d+1}X_{ir}\theta_r$，**输入固定**，对 $\theta_j$ 求偏导时只有 $r=j$ 那一项存活。**Kronecker delta** $\delta_{rj}=\begin{cases}1,&r=j\\0,&r\ne j\end{cases}$：

$$\frac{\partial f_{\boldsymbol\theta}(\boldsymbol x_i)}{\partial\theta_j}=\sum_{r=1}^{d+1}X_{ir}\frac{\partial\theta_r}{\partial\theta_j}=\sum_rX_{ir}\delta_{rj}=X_{ij},\quad j=1,\dots,d+1.$$

**直觉**：打分对第 $j$ 个参数的敏感度，就是对应的输入值 $X_{ij}$。对截距 $j=1$，$X_{i1}=1$。

**P.27 Putting the three terms together（三项合起来）**

$$\frac{\partial}{\partial\theta_j}\ell_{\log}=\underbrace{\frac{p-y_i}{p(1-p)}}_{\text{第一项}}\cdot\underbrace{p(1-p)}_{\text{第二项}}\cdot\underbrace{X_{ij}}_{\text{第三项}}=\big(p_{\boldsymbol\theta}(\boldsymbol x_i)-y_i\big)X_{ij}.$$

**$p(1-p)$ 前后抵消**，只剩 $(p-y)\cdot X_{ij}$：**预测概率减真实标签，再乘输入**。和回归梯度的形状一模一样。

**P.28 Empirical gradient and gradient descent（经验梯度与梯度下降）**
- 经验损失是求和，所以把每个观测的导数加起来：$\dfrac{\partial\mathcal L_{\log}}{\partial\theta_j}=\sum_{i=1}^{N}X_{ij}\big(p_{\boldsymbol\theta}(\boldsymbol x_i)-y_i\big)$。
- 向量写法：$\boldsymbol\sigma(s_1,\dots,s_N)^\top=(\sigma(s_1),\dots,\sigma(s_N))^\top$（**对每个分量分别做 sigmoid**）；$\boldsymbol p(\boldsymbol\theta)=\boldsymbol\sigma(\boldsymbol X\boldsymbol\theta)$；$\boldsymbol y=(y_1,\dots,y_N)^\top$。

$$\nabla\mathcal L_{\log}(\boldsymbol\theta)=\boldsymbol X^\top\big(\boldsymbol p(\boldsymbol\theta)-\boldsymbol y\big)=\boldsymbol X^\top\big(\boldsymbol\sigma(\boldsymbol X\boldsymbol\theta)-\boldsymbol y\big).$$

- 拟合 logistic 回归：$\boldsymbol\theta^{(t+1)}=\boldsymbol\theta^{(t)}-\eta\boldsymbol X^\top\big(\boldsymbol\sigma(\boldsymbol X\boldsymbol\theta^{(t)})-\boldsymbol y\big)$。
- **形状检查**：$\boldsymbol X^\top$ 是 $(d+1)\times N$，括号里是 $N\times1$，结果 $(d+1)\times1$，和 $\boldsymbol\theta$ 同形。

**P.29 Stationary points and minimisers（驻点与极小点）**
- **事实**：经验 logistic loss 是参数的**凸函数**。所以**任何驻点（导数为 0 的点）都是全局极小**。
- **但是**：**有限的极小点不一定存在**！（讲义里有反例）。即使存在，驻点方程也是**非线性方程组**，**一般没有闭式解**。（不像 W02 的线性回归有正规方程。）
- **警告 1**：经验 logistic loss 变小，**不保证错误个数变少**。
- **警告 2**：梯度下降仍需要合适的学习率：**步长太大，损失可能反而增大**。

> [!TIP]
> **补充说明：「凸」是什么意思**
> 凸函数的图像像一只碗，没有「假的低谷」，所以导数为 0 就是最低点。（第 4 周的神经网络就**不再**是凸的，会出现局部极小。）

**P.30–35 Gradient descent, $\eta=0.001$，$t=0,1,2,5,10,20$**
同一组数据，初始参数 $\boldsymbol\theta^{(0)}=(-5.75,\ 0,\ 1)^\top$，学习率 $\eta=0.001$。每页左边是数据和当前决策边界，右边是「损失 ÷ $\log2$」和「错误个数」随迭代的曲线，底部标出当前损失和错误个数。

**P.36 The loss can fall while mistakes increase（损失下降而错误增加）**
同一次运行（$\eta=0.001$）：第 3 步损失 41.4948、错 7 个；**第 4 步损失降到 41.3932，错误却涨到 8 个**。页面强调：当然，上界在这两个参数选择下仍然成立。

**P.37–42 Gradient descent, $\eta=0.002$，$t=0,1,2,5,10,20$**
同一组数据、同样初始参数，只把学习率改成 $\eta=0.002$。

**P.43 Step size and logistic loss（步长与 logistic loss）**
对比两条损失曲线：$\eta=0.001$ 时损失稳步下降；$\eta=0.002$ 时**连第一次更新都让损失上升**（51.048 → 64.556）。页面总结：**步长太大会让损失增大**。

#### 第 30–43 页的数字汇总（我独立复算，与页面标注一致）

100 个观测，初始 $\boldsymbol\theta^{(0)}=(-5.75,0,1)$，**求和损失**，更新 $\boldsymbol\theta\leftarrow\boldsymbol\theta-\eta\boldsymbol X^\top(\boldsymbol p-\boldsymbol y)$。

| $t$ | $\eta=0.001$：$(b,w_1,w_2)$ | 损失 | 错误数 | $\eta=0.002$：$(b,w_1,w_2)$ | 损失 | 错误数 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | $(-5.7500,\ 0.0000,\ 1.0000)$ | 51.048 | 31 | $(-5.7500,\ 0.0000,\ 1.0000)$ | 51.048 | 31 |
| 1 | $(-5.7326,\ 0.1166,\ 1.1052)$ | 42.733 | 15 | $(-5.7152,\ 0.2332,\ 1.2104)$ | 64.556 | 42 |
| 2 | $(-5.7409,\ 0.0703,\ 1.0811)$ | 41.668 | 7 | $(-5.7734,\ -0.1150,\ 0.9670)$ | 75.928 | 49 |
| 3 | $(-5.7408,\ 0.0766,\ 1.0982)$ | 41.495 | 7 | $(-5.7081,\ 0.3161,\ 1.3384)$ | 101.364 | 49 |
| 4 | $(-5.7432,\ 0.0670,\ 1.1027)$ | 41.393 | **8** | $(-5.7887,\ -0.1650,\ 0.9967)$ | 81.900 | 50 |
| 5 | $(-5.7448,\ 0.0626,\ 1.1113)$ | 41.300 | 8 | $(-5.7188,\ 0.2960,\ 1.3923)$ | 105.343 | 49 |
| 10 | $(-5.7537,\ 0.0343,\ 1.1485)$ | 40.848 | 8 | $(-5.8256,\ -0.2568,\ 1.1321)$ | 77.632 | 49 |
| 20 | $(-5.7712,\ -0.0202,\ 1.2213)$ | 39.992 | 7 | $(-5.8848,\ -0.4053,\ 1.3445)$ | 72.282 | 48 |

**怎么读这张表**
1. **$\eta=0.001$**：损失几乎一路下降（51 → 40），错误数 31 → 7，但**中间有起伏**（$t=4$ 从 7 涨到 8）。
2. **$\eta=0.002$**：**学习率只翻倍，结果完全不同**：损失先**涨**到 105，再回落，20 步后仍有 48 个错误，**还不如初始的 31 个**。
3. 两次运行里，**损失和错误数并不总是同方向变化**（对应警告 1）。
4. 官方 `Examples/` 里的参考 CSV 与这张表逐行一致（我核对了 42 行），平局规则对这些数没有影响：42 个迭代点里**没有一个**打分恰好为 0。

> [!TIP]
> **补充说明：为什么 $\eta=0.002$ 会「冲过头」**
> 梯度下降每步沿负梯度走 $\eta\times$ 梯度。损失曲面在某些方向很陡，步子迈大一点就**越过谷底、跳到对面更高的地方**。这就是「步长太大」的含义。$\eta=0.002$ 对这个数据已经偏大了。

### 第四块：多于两类（第 44–52 页）

**P.44**：目录页，进入第四块。

**P.45 More than two classes（多类）**
- 例子：手写数字识别，要在 10 个类（0 到 9）中选。
- 一张灰度图像由**像素**组成，强度构成一个实数矩阵；**按行读出，拼成一个行向量** $\boldsymbol x=(x_1,\dots,x_d)\in\mathbb R^d$，$d$ 是像素个数。**每张图用同样的像素顺序**。
- 目标：**每类预测一个概率**。

**P.46 Predicting class probabilities（预测类概率）**
- 类编号 $1,\dots,K$。数字 0 到 9 对应类索引 $k=$ 数字 $+1$。
- 概率向量 $\boldsymbol p:\mathbb R^d\to[0,1]^K$，$\boldsymbol p(\boldsymbol x)=(p_1(\boldsymbol x),\dots,p_K(\boldsymbol x))$。要求是**合法的概率分布**：分量之和为 1，$\sum_{k=1}^{K}p_k(\boldsymbol x)=1$。

**P.47 The multiclass plug-in rule（多类 plug-in 规则）**
- 选**概率最大**的类：$h_{\boldsymbol p}:\mathbb R^d\to\{1,\dots,K\}$。
- **平局时，取并列最大者中下标最小的**（与二分类「平局归类 0」呼应）。

**P.48 One affine score per class（每类一个仿射打分）**
- 每个类 $k$ 有自己的截距和权重：$f_{\boldsymbol\theta_k}(\boldsymbol x)=b_k+\boldsymbol x\boldsymbol w_k$，$\boldsymbol\theta_k=(b_k,\boldsymbol w_k)\in\mathbb R^{d+1}$。这些打分也叫 logits。
- 把所有参数列并排放进**参数矩阵** $\boldsymbol\Theta=(\boldsymbol\theta_1,\dots,\boldsymbol\theta_K)\in\mathbb R^{(d+1)\times K}$。
- 全部打分排成一行：$\boldsymbol f_{\boldsymbol\Theta}(\boldsymbol x)=(f_{\boldsymbol\theta_1}(\boldsymbol x),\dots,f_{\boldsymbol\theta_K}(\boldsymbol x))\in\mathbb R^K$。

**P.49 Softmax turns scores into probabilities（softmax 把打分变成概率）**
- **警告**：**对每个类单独做 sigmoid，概率之和不保证是 1！** 例：三个打分都是 0，会得到三个 $0.5$（和为 $1.5$）。
- 解决：**先对每个打分取指数，再除以总和**：

$$\mathrm{softmax}(\boldsymbol s)=\Big(\frac{e^{s_1}}{\sum_{r=1}^Ke^{s_r}},\ \dots,\ \frac{e^{s_K}}{\sum_{r=1}^Ke^{s_r}}\Big),\quad\mathrm{softmax}:\mathbb R^K\to(0,1)^K.$$

- 每个分量**为正**（指数恒正）；所有分量**加起来恰好为 1**（分子之和等于分母）。

> [!NOTE]
> **自编例子（非讲义）**
> - 三个打分都是 0：$e^0=1$，总和 3，softmax $=(\tfrac13,\tfrac13,\tfrac13)$，和为 1。（三个 sigmoid 给的是 $0.5,0.5,0.5$，和为 $1.5$，不合法。）
> - 打分 $(2,1,0)$：$e^2\approx7.389,\ e^1\approx2.718,\ e^0=1$，总和 $\approx11.107$，softmax $\approx(0.6652,\ 0.2447,\ 0.0900)$，和为 1。**打分最大的类概率最大。**

**P.50 Our multiclass probability model（多类概率模型）**
- $\boldsymbol p_{\boldsymbol\Theta}(\boldsymbol x)=\mathrm{softmax}(\boldsymbol f_{\boldsymbol\Theta}(\boldsymbol x))$，第 $k$ 个分量 $p_{\boldsymbol\Theta,k}(\boldsymbol x)=\dfrac{\exp f_{\boldsymbol\theta_k}(\boldsymbol x)}{\sum_{r=1}^{K}\exp f_{\boldsymbol\theta_r}(\boldsymbol x)}$。
- 有了概率预测器 $\boldsymbol p_{\boldsymbol\Theta}$，就可以用上一页的 plug-in 规则 $h_{\boldsymbol p_{\boldsymbol\Theta}}$。

**P.51 Cross-entropy loss for the observed class（观测类的交叉熵损失）**
- 损失 = **预测给真实类的概率的负对数**：$\ell_{\rm multi}(\boldsymbol p,y)=-\log p_y$（$p_y>0$）。
- 数据集 $(\boldsymbol x_1,y_1),\dots,(\boldsymbol x_N,y_N)$，$y_i\in\{1,\dots,K\}$。总损失 $\mathcal L_{\rm multi}(\boldsymbol\Theta)=\sum_i\ell_{\rm multi}(\boldsymbol p_{\boldsymbol\Theta}(\boldsymbol x_i),y_i)$。拟合：最小化它，例如用梯度下降。
- 页面右下角：**错误个数 $\le\dfrac{1}{\log2}\mathcal L_{\rm multi}(\boldsymbol\Theta)$**。

> [!NOTE]
> **自编例子（非讲义）：交叉熵**
> 打分 $(2,1,0)$，softmax $\approx(0.6652,0.2447,0.0900)$。真实类是 1：$\ell=-\log0.6652\approx0.408$（小，因为预测对且较自信）；真实类是 3：$\ell=-\log0.0900\approx2.408$（大，因为给真实类的概率很低）。

> [!TIP]
> **补充说明（我自己的推导，非讲义原文）：为什么多类也有「错误数 $\le\mathcal L/\log2$」**
> 若分类器选错，选中的类 $c$ 满足 $p_c\ge p_y$，而 $p_c+p_y\le1$，所以 $p_y\le\tfrac12$，即 $-\log p_y\ge\log2$。每个错误至少贡献 $\log2$ 的损失，求和后就得到上界。和二分类的论证相同。

**P.52 Lecture notes and exercises（讲义与习题）**
指向课程 GitHub（二维码）。讲义解释得更详细并含**习题**；**习题解答会在下一讲前公布**。（旧版这一页有一张「模型与拟合损失」总结表，新版已删除，见 [FOAI-W03-线性分类](FOAI-公开版-W03-线性分类.md) 的来源冲突标注。）

---

## 四、听课前带着的 6 个问题

1. 为什么**不能**直接对「错误个数」做梯度下降？（第 14 页）
2. sigmoid 把什么变成了什么？为什么打分的正负就决定类别？（第 8–9 页）
3. logistic loss 为什么对「自信地错」罚得比「几乎瞎猜时错」重？（第 15–16 页）
4. 为什么「错误个数 $\le\mathcal L_{\log}/\log2$」？这个不等式能反过来用吗？（第 18–19 页）
5. 梯度里的 $p(1-p)$ 是怎么抵消的？最后为什么只剩 $(p-y)X_{ij}$？（第 24–27 页）
6. 为什么多类**不能**对每个类单独做 sigmoid？softmax 做了什么？（第 49 页）

## 自测

1. 写出 plug-in 规则，并说明打分恰好为 0 时判哪一类。
2. 算 $\sigma(0)$、$\sigma(2)$ 的大致值，并解释它们的含义。
3. 对 $y=0$：$p=0.51$ 与 $p=0.9$ 的 logistic loss 各是多少？为什么不同？
4. 一个观测 $x=2,\ y=1,\ \boldsymbol\theta=(0,0.5)$，手算 $s,\ p$，以及 $\partial\ell/\partial b$ 与 $\partial\ell/\partial w$（答案见第 2.4 节）。
5. 用文字说明 $\nabla\mathcal L_{\log}=\boldsymbol X^\top(\boldsymbol p-\boldsymbol y)$ 每一块的形状。
6. 为什么 $\eta=0.002$ 时第一步损失反而上升？
7. 打分 $(1,0,-1)$ 的 softmax 大致是多少？和是否为 1？

## 术语速查

| 术语 | 含义 |
| --- | --- |
| 分类器 $h$ | 输入到类别标签的函数 |
| 打分 / logit | 仿射函数的输出，可为任何实数 |
| sigmoid $\sigma$ | 把打分压成 $(0,1)$ 的概率 |
| plug-in 规则 | 把概率变成标签的规则（选大者，平局取小） |
| 决策边界 | 打分为 0 的点的集合（超平面） |
| 0-1 损失 / 错误个数 | 只数对错的损失 |
| logistic loss / 二元交叉熵 | $-y\log p-(1-y)\log(1-p)$ |
| surrogate loss（替代损失） | 光滑的、能做梯度下降的「替身」 |
| 设计矩阵 $\boldsymbol X$ | 每行是 $(1,\boldsymbol x_i)$ |
| 链式法则 | 复合函数求导：外层导数乘内层导数 |
| 凸函数 | 图像像碗；驻点即全局最小 |
| 学习率 $\eta$ | 梯度下降每步走多远 |
| softmax | $K$ 个打分 → 和为 1 的 $K$ 个概率 |
| 交叉熵损失 | $-\log p_y$ |
