---
title: FOAI W03｜线性分类
course: FOAI
week: 3
type: lecture
tags:
  - AI-MSc
  - TB1
  - FOAI
  - 线性分类
  - logistic回归
  - softmax
source: 课程材料（Conor Houghton、Roberto Colomboni）的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: draft
publication: pending
updated: 2026-10-09
---

# FOAI W03｜线性分类（Linear Classification）

> [!IMPORTANT]
> **非官方学习笔记**：这是我在 University of Bristol「Foundations of Artificial Intelligence（SEMTM0051, 2026/27）」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：Conor Houghton、Roberto Colomboni。原始材料：[semtm0051/2026_27](https://github.com/semtm0051/2026_27/tree/322787dec91f174303ccf90399bb6243a060eb7d/)。经两位讲师同意公开；不含练习解答与非公开材料（如课堂录像）。

> [!NOTE]
> **资料范围与可靠度：high**
> 一手讲义：[`Week_03/Notes/Week_03 - Linear Classification.pdf`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_03/Notes/Week_03%20-%20Linear%20Classification.pdf)（26 页，version of 27 September 2026）。配套 Slides、`Examples/`（CSV + 脚本）已入库，Slides 逐页标注见 [FOAI-W03-Slides课堂逐页标注](FOAI-公开版-W03-Slides逐页标注.md)。
> 公式编号 (n) 沿用讲义，便于对照。

> [!WARNING]
> **平局规则已改（2026-10-08 pull）**
> 旧版讲义：$p \ge 1/2$ 预测类 1，平局归类 1。**新版：$p > 1/2$ 才预测类 1，平局归类 0**（多分类取最小标号）。
> 因此所有 $\ge$ 都变成 $>$，决策边界本身属于类 0，原来的“初始 4 个错误”变成 **3 个**。若你之前抄过旧版数字或课堂笔记，以本篇为准；Slides 对应改动已随 pull 入库但未逐页核对。

> [!WARNING]
> **来源冲突｜讲义与 Slides 的向量记号不一致（2026-10-09）**
> 本笔记依据讲义 PDF，输入 $\boldsymbol x$ 写成**行向量** $\mathbb R^{1\times d}$，多分类的得分和概率写成 $\mathbb R^{1\times K}$、$(0,1)^{1\times K}$。
> 官方 Slides 在 2026-10-08 晚的更新里（`322787d`）把对应的第 45–49、51 页改成 $\mathbb R^d$、$\mathbb R^K$、$(0,1)^K$，第 49 页的 softmax 也改成直接写整个向量。数学含义相同，只是记号不同。
> 讲义 PDF 本身在这次更新里**没有改**，仍是 $\mathbb R^{1\times d}$。两边对照复习时，以讲义为准，遇到 Slides 的写法按等价记号理解。
> 另：Slides 第 19 页的误差上界推导已改写成 $\mathrm{step}\big(\sigma(f_{\boldsymbol\theta}(\boldsymbol x_i))-\tfrac12\big)$，与讲义的 $h_p(x)=\mathrm{step}(p(x)-\tfrac12)$ 一致；最后一页旧版的「模型与拟合损失」汇总表在新版里已删除。

## 核心脉络

**打分（score）→ sigmoid 变概率 → 阈值取标签 → 用 logistic loss 代替 0-1 损失 → 梯度下降 → 推广到 softmax**。

与 W02 套路相同（选函数族 → 选损失 → 最小化），区别是：数 0-1 错误的目标函数处处平坦，梯度没用，所以要换一个光滑的**代理损失（surrogate loss）**。

---

## 1. 二分类：预测、概率、决策边界

### 1.1 目标
数据 $(\boldsymbol x_i, y_i)_{i=1}^N$，$\boldsymbol x_i\in\mathbb R^{1\times d}$（行向量），$y_i\in\{0,1\}$（只是类名，不是数值目标）。分类器 $h:\mathbb R^d\to\{0,1\}$。零一损失：
$$\ell_{0,1}(\hat y,y)=\mathbf 1\{\hat y\ne y\}\tag{1}$$
求和得错误数；除以 $N$ 是错误率，$1-$ 错误率是准确率。**拟合与泛化是两回事**（欠拟合/过拟合同样适用），本讲只讲拟合。

### 1.2 先预测概率，再选标签
模型给 $p(\boldsymbol x)\in[0,1]$ 作为类 1 的概率，类 0 为 $1-p$。**plug-in 分类器**（新版规则）：
$$h_p(\boldsymbol x)=\begin{cases}0,& p(\boldsymbol x)\le 1/2\\ 1,& p(\boldsymbol x)>1/2\end{cases}\tag{2}$$
用 Heaviside 阶跃函数 $\operatorname{step}(s)=\begin{cases}0,&s\le0\\1,&s>0\end{cases}$（(3)，**$\operatorname{step}(0)=0$**），写成 $h_p=\operatorname{step}(p-\tfrac12)$。

> 概率比标签信息多：$p=0.51$ 与 $p=0.99$ 标签相同，但自信程度不同——拟合时我们要惩罚“自信的错误”更重。

### 1.3 从仿射打分到概率
参数 $\boldsymbol\theta=(b,\boldsymbol w)^\top\in\mathbb R^{(d+1)\times1}$，打分（也叫 **logit**）
$$f_{\boldsymbol\theta}(\boldsymbol x)=b+\boldsymbol x\boldsymbol w\tag{5}$$
仿射函数取值不在 $[0,1]$，所以过 **logistic sigmoid**：
$$\sigma(s)=\frac1{1+e^{-s}}=\frac{e^s}{1+e^s},\qquad p_{\boldsymbol\theta}(\boldsymbol x)=\sigma(b+\boldsymbol x\boldsymbol w)\tag{6,7}$$
$\sigma$ 严格递增、$\sigma(0)=1/2$、取值在 $(0,1)$。选 sigmoid 是**建模选择**，原因是配 logistic loss 后梯度会化简成“概率 − 标签”。

### 1.4 决策边界是超平面
$\sigma(s)>1/2\iff s>0$，所以
$$h_{p_{\boldsymbol\theta}}(\boldsymbol x)=\operatorname{step}(b+\boldsymbol x\boldsymbol w)\tag{8}$$
标签只取决于打分的符号；sigmoid 只为拟合服务。$\boldsymbol w\ne\boldsymbol 0$ 时 $b+\boldsymbol x\boldsymbol w=0$ 是**超平面**（$d=1$ 点，$d=2$ 直线，$d=3$ 平面），这就是**决策边界**。$\boldsymbol w=\boldsymbol 0$ 时处处同一标签。“线性”指**打分是仿射的**，不是说 $p_{\boldsymbol\theta}$ 对输入线性。

| 例子 | 打分 | 边界 | 预测类 1 的区域（新规则） |
|---|---|---|---|
| $d=1$，$a>0$ | $b+ax$ | 点 $x=-b/a$ | $x>-b/a$ |
| $d=1$，$a<0$ | $b+ax$ | 同上 | 阈值**左侧**（严格），阈值本身为类 0 |
| $d=2$，$\boldsymbol\theta=(-1,-0.6,1)^\top$ | $x_2-0.6x_1-1$ | 直线 $x_2=0.6x_1+1$ | 直线**严格上方**；线上及下方为类 0 |

图 1：$a=1,b=-1$，阈值 $x=1$，$x\le1$ 预测 0，$x>1$ 预测 1。

---

## 2. 从分类错误到 logistic loss

### 2.1 经验零一损失
$$\mathcal E(\boldsymbol\theta)=\sum_{i=1}^N\mathbf 1\{h_{p_{\boldsymbol\theta}}(\boldsymbol x_i)\ne y_i\}\tag{13}$$
即错分数。图 4 对比两个候选边界：$\boldsymbol\theta_A=(-5.75,0,1)^\top$（水平线 $x_2=5.75$）与 $\boldsymbol\theta_B=(-1,-0.6,1)^\top$，B 错得更少，$\mathcal E(\boldsymbol\theta_B)<\mathcal E(\boldsymbol\theta_A)$。

和最小二乘不同，**最小化 $\mathcal E$ 没有闭式解**，直接搜索分类模式是组合问题。

### 2.2 为什么数错误数不能指导梯度下降
若所有观测的打分都**非零**，参数小幅变动不改变任何标签（打分对参数连续），所以 $\mathcal E$ 在该点邻域内是常数：
$$\nabla\mathcal E(\boldsymbol\theta)=\boldsymbol 0\quad\text{（所有打分非零时）}$$
梯度不给任何改进方向。

### 2.3 例子：一维阈值（7 个观测）
| $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| $x_i$ | −2 | −1 | 0 | 1 | 2 | 3 | 4 |
| $y_i$ | 0 | 0 | 0 | 1 | 1 | 0 | 1 |

- 标签从左到右变化 3 次，而阈值分类器至多变 1 次 ⇒ 任何阈值都无法全对；$x-\tfrac12$ 只错 $x=3$ 一个，**1 个错误是此族的最优**。
- 取 $\boldsymbol\theta=(3,1)^\top$：7 个打分 $1,\dots,7$ 全正，错 4 个，且 $\nabla\mathcal E=\boldsymbol0$ ⇒ **任何 $\eta>0$ 更新都原地不动**，尽管 $(-1/2,1)^\top$ 只错 1 个。
- 固定 $a=1$ 改变 $b$（图 6，新规则，右端点闭）：

$$\mathcal E((b,1)^\top)=\begin{cases}3,&b\le-4\\2,&-4<b\le-3\\3,&-3<b\le-2\\2,&-2<b\le-1\\1,&-1<b\le0\\2,&0<b\le1\\3,&1<b\le2\\4,&2<b\end{cases}\tag{16}$$

阶梯状：只在某个观测的打分变号时才跳。结论：想找一个**随参数光滑变化、又与错误数有联系**的目标。

### 2.4 光滑代理：logistic loss（二元交叉熵）
取“观测标签所得概率”的负对数：
$$\ell_{\log}(p,y)=-y\log p-(1-y)\log(1-p)=\begin{cases}-\log p,&y=1\\-\log(1-p),&y=0\end{cases}\tag{17}$$
（$\log$ 是自然对数。）自信的错误罚得重：$y=0$ 时 $p=0.51$ 损失 $-\log0.49\approx0.713$，$p=0.99$ 损失 $-\log0.01\approx4.605$。
$$\mathcal L_{\log}(\boldsymbol\theta)=\sum_{i=1}^N\ell_{\log}(p_{\boldsymbol\theta}(\boldsymbol x_i),y_i)\tag{18}$$
关于 $\boldsymbol\theta$ 光滑（仿射 → sigmoid → log，$p\in(0,1)$）。小写 $\ell$ 是单样本损失，大写 $\mathcal L$ 是求和。

### 2.5 小 logistic loss ⇒ 少错误
每次错分，观测标签的概率 $\le1/2$（新规则：$y=1$ 错时 $p\le\tfrac12$，$y=0$ 错时 $p>\tfrac12$），故该样本 logistic loss $\ge\log2$：
$$\ell_{0,1}(h_p(\boldsymbol x),y)\le\frac{\ell_{\log}(p(\boldsymbol x),y)}{\log2}\tag{20}$$
求和得
$$\boxed{\ \mathcal E(\boldsymbol\theta)\le\frac{\mathcal L_{\log}(\boldsymbol\theta)}{\log2}\ }\tag{23}$$
推论：$\mathcal L_{\log}<n\log2\Rightarrow\mathcal E<n$（至多 $n-1$ 个错误）。

> [!NOTE]
> **新版措辞变化**
> 旧版写“$\mathcal L_{\log}<\log2\Rightarrow$ 无错误”。新版边界情形 $y=0$ 的错分下界变严（$>\log2$），$y=1$ 的仍是 $\ge\log2$，并把该结论改成习题 2(c)：**仅知 $\mathcal L_{\log}\le\log2$ 能否保证零错误？要证明或给反例**。我没有在此给出答案，留作自己的练习。

用打分写单样本损失（利用 $1-\sigma(s)=\sigma(-s)$）：
$$\ell_{\log}(\sigma(s),1)=\log(1+e^{-s}),\qquad\ell_{\log}(\sigma(s),0)=\log(1+e^{s})\tag{22}$$

### 2.6 设计矩阵
$$\boldsymbol X=\begin{pmatrix}1&\boldsymbol x_1\\\vdots&\vdots\\1&\boldsymbol x_N\end{pmatrix}\in\mathbb R^{N\times(d+1)},\qquad \boldsymbol X\boldsymbol\theta=\text{所有打分}\tag{24,27}$$
$\partial f_{\boldsymbol\theta}(\boldsymbol x_i)/\partial\theta_j=X_{ij}$（含截距：$X_{i1}=1$）。

### 2.7–2.8 梯度推导（链式法则三因子）
$$\frac{\partial\ell_{\log}}{\partial p}=\frac{p-y}{p(1-p)},\qquad\sigma'(s)=\sigma(s)(1-\sigma(s)),\qquad\frac{\partial f}{\partial\theta_j}=X_{ij}$$
相乘，分母消掉：
$$\frac{\partial}{\partial\theta_j}\ell_{\log}(p_{\boldsymbol\theta}(\boldsymbol x_i),y_i)=(p_{\boldsymbol\theta}(\boldsymbol x_i)-y_i)X_{ij}\tag{32}$$
$$\boxed{\ \nabla\mathcal L_{\log}(\boldsymbol\theta)=\boldsymbol X^\top\big(\boldsymbol\sigma(\boldsymbol X\boldsymbol\theta)-\boldsymbol y\big)\ }\tag{35}$$
形状自检：$\boldsymbol X^\top$ 是 $(d+1)\times N$，$\boldsymbol p-\boldsymbol y$ 是 $N\times1$，梯度是 $(d+1)\times1$，截距分量在最前。

**与线性回归对照**：

| | 梯度 | 括号里的“预测” |
|---|---|---|
| 平方损失（W02） | $2\boldsymbol X^\top(\boldsymbol X\boldsymbol\theta-\boldsymbol y)$ | $\boldsymbol X\boldsymbol\theta$ |
| logistic loss（本讲） | $\boldsymbol X^\top(\boldsymbol\sigma(\boldsymbol X\boldsymbol\theta)-\boldsymbol y)$ | $\boldsymbol\sigma(\boldsymbol X\boldsymbol\theta)$，且**没有因子 2** |

**驻点**：$\boldsymbol X^\top(\boldsymbol\sigma(\boldsymbol X\boldsymbol\theta)-\boldsymbol y)=\boldsymbol0$ 是非线性方程组，一般无显式解，所以用迭代。若驻点存在，则它是**全局最小点**（附录 A 证明：沿直线 $g(\alpha)$ 二阶导 $\ge0$，故 $g'$ 非降）。但**有限最小点不一定存在**（见 §5）。

### 2.9 梯度下降（= logistic regression 的拟合）
$$\boldsymbol\theta^{(t+1)}=\boldsymbol\theta^{(t)}-\eta\boldsymbol X^\top\big(\boldsymbol\sigma(\boldsymbol X\boldsymbol\theta^{(t)})-\boldsymbol y\big)\tag{37}$$
按步拆开（38）：$\boldsymbol s^{(t)}=\boldsymbol X\boldsymbol\theta^{(t)}$ → $\boldsymbol p^{(t)}=\boldsymbol\sigma(\boldsymbol s^{(t)})$ → $\boldsymbol r^{(t)}=\boldsymbol p^{(t)}-\boldsymbol y$ → $\boldsymbol\theta^{(t+1)}=\boldsymbol\theta^{(t)}-\eta\boldsymbol X^\top\boldsymbol r^{(t)}$。所有分量**同时**更新，输入和标签始终固定。结构与 W02 一致，只多了一步 sigmoid。

### 2.10–2.11 手算例子（§2.3 的 7 个观测）
$\boldsymbol\theta^{(0)}=(0,0)^\top$，$\eta=0.2$。初始所有 $p=1/2$，新规则下全预测类 0，**错 3 个**（$y=1$ 的三个点），$\mathcal L_{\log}=7\log2\approx4.85203$。

- $\boldsymbol r^{(0)}$：$y=0$ 的点是 $+\tfrac12$，$y=1$ 的点是 $-\tfrac12$。
- 梯度 $\nabla\mathcal L_{\log}(\boldsymbol\theta^{(0)})=\boldsymbol X^\top\boldsymbol r^{(0)}=(\tfrac12,-\tfrac72)^\top$（(40)）。
- 更新 $\boldsymbol\theta^{(1)}=(0,0)^\top-0.2(\tfrac12,-\tfrac72)^\top=(-0.1,0.7)^\top$，打分 $-0.1+0.7x$。
- $p^{(1)}\approx0.1824,0.3100,0.4750,0.6457,0.7858,0.8808,0.9370$（$x=-2\dots4$），只错 $x=3$ 一个，$\mathcal L_{\log}(\boldsymbol\theta^{(1)})\approx4.08738$。

| 迭代 $t$ | 截距 $b^{(t)}$ | 斜率 $a^{(t)}$ | $\mathcal L_{\log}$ | 错误数 $\mathcal E$ |
|---|---|---|---|---|
| 0 | 0.00000 | 0.00000 | 4.85203 | 3 |
| 1 | −0.10000 | 0.70000 | 4.08738 | 1 |
| 2 | −0.34336 | 0.51341 | 3.76388 | 1 |
| 5 | −0.69675 | 0.60092 | 3.55828 | 2 |
| 20 | −1.13979 | 0.74078 | 3.46285 | 2 |

（数字来自讲义，我未重算；中间迭代未取整。）

> [!IMPORTANT]
> **本讲最重要的一点**
> logistic loss 一路下降，但**错误数在 $t=4$ 从 1 增到 2**：阈值 $-b/a$ 从约 0.143（$t=1$）移到约 1.539（$t=20$），越过 $x=1$，使 $y=1$ 的点被判错；同时 $x=3$ 的点概率下降，对损失有利。**代理损失小 ≠ 每步错误更少**，只保证 (23) 这个上界。

---

## 3. 多分类

### 3.0 图像输入（新版新增）
灰度图 = 像素强度矩阵（8 位为 0–255，除以 255 得 $[0,1]$）。**按行展平**成输入行，如 $2\times3$ 图 $\begin{pmatrix}0&0.4&1\\0.8&0.2&0\end{pmatrix}\mapsto(0,0.4,1,0.8,0.2,0)$，$d=2\cdot3=6$。所有图像须同尺寸、同像素顺序。标签 $y\in\{1,\dots,K\}$，数字识别 $K=10$。

### 3.1 每类一个仿射打分
$f_{\boldsymbol\theta_k}(\boldsymbol x)=b_k+\boldsymbol x\boldsymbol w_k$，参数矩阵 $\boldsymbol\Theta=(\boldsymbol\theta_1\ \cdots\ \boldsymbol\theta_K)\in\mathbb R^{(d+1)\times K}$（首行是 $K$ 个截距）。$\boldsymbol X\boldsymbol\Theta\in\mathbb R^{N\times K}$：第 $i$ 行是第 $i$ 个观测的 $K$ 个打分。

### 3.2 softmax
对每类分别用 sigmoid 不行（和不为 1）。用
$$\operatorname{softmax}:\mathbb R^{1\times K}\to\mathbb R^{1\times K},\qquad[\operatorname{softmax}(\boldsymbol s)]_k=\frac{e^{s_k}}{\sum_{r=1}^Ke^{s_r}}\tag{45}$$
三步：取指数 → 求和 → 逐项除以和。全正、和为 1。例：$\boldsymbol s=(0,\log2,\log3)\Rightarrow(1,2,3)/6=(\tfrac16,\tfrac13,\tfrac12)$。

预测取最大概率的类，**平局取最小标号**：
$$h_{\boldsymbol p_{\boldsymbol\Theta}}(\boldsymbol x)=\min\Big(\operatorname*{argmax}_{k}p_{\boldsymbol\Theta,k}(\boldsymbol x)\Big)$$
（argmax 返回集合，见 W01。）与二分类“平局归类 0”一致。多分类最大概率**不必超过 1/2**，如 $(0.40,0.35,0.25)$ 仍预测类 1。

### 3.3 交叉熵
$$\ell_{\text{multi}}(\boldsymbol p,y)=-\log p_y\tag{47}$$
用打分写：
$$\ell_{\text{multi}}(\operatorname{softmax}(\boldsymbol s),y)=-s_y+\log\Big(\sum_{r=1}^Ke^{s_r}\Big)\tag{48}$$
$$\mathcal L_{\text{multi}}(\boldsymbol\Theta)=-\sum_{i=1}^N\log\big[\boldsymbol p_{\boldsymbol\Theta}(\boldsymbol x_i)\big]_{y_i}\tag{49}$$
例：$\boldsymbol p=(\tfrac16,\tfrac13,\tfrac12)$，$y=2$，损失 $=\log3$。讲义**不推导**多分类梯度，原理同二分类。

---

## 4. 习题（讲义 Section 4，新版）

只记要点，答案自己做：

1. 证明 $\sigma'(s)=\sigma(s)\sigma(-s)=\sigma(s)(1-\sigma(s))$（先找内外函数，再链式法则，用 (21)）。
2. 两个概率预测器：类 1 概率 $(0.51,0.51)$ vs $(0.49,0.99)$，标签都是 1。(a) 各自标签、错误数、$\sum\ell_{\log}$；(b) 两种目标分别偏好谁，为何不矛盾 (20)；(c) 平均 logistic loss $\le0.03$ 给出怎样的错误率上界；**以及 $\sum\ell_{\log}\le\log2$ 是否保证零错误，证明或反例**（新版改动）。
3. 数据 $x=(-1,0,2)$，$y=(0,1,1)$，$\boldsymbol\theta^{(0)}=(0,\log3)^\top$，$\eta=1/5$。(a) 构造 $\boldsymbol X$、$\boldsymbol s,\boldsymbol p,\boldsymbol r$（保持精确值）；(b) 算梯度并更新一次；(c) 更新前后错误数，**找出阈值前后位置，以及哪个观测在阈值移动时改变预测类**（新版改动）。
4. 两个三分类观测，打分 $\boldsymbol s_1=(0,\log3,\log3)$，$\boldsymbol s_2=(\log4,0,\log2)$，标签 $y_1=3,\ y_2=1$（新版改了 $\boldsymbol s_1$ 与 $y_1$，专门考平局）。(a) softmax 精确值；(b) **按平局规则**判断预测与对错；(c) 交叉熵与总和；(d) 用 (48) 再验证。

> 全程用自然对数、**求和不除 $N$**；用 (20) 时别忘了 $1/\log2$。

## 5. 附录要点

- **附录 A**：驻点必为全局最小点；但数据 $x=(-2,-1,1,2)$，$y=(0,0,1,1)$ 可被单阈值完全分开，$\mathcal L_{\log}((0,a)^\top)=2\log(1+e^{-2a})+2\log(1+e^{-a})>0$ 却随 $a\to+\infty$ 趋于 0 ⇒ **下确界 0 取不到，没有有限最小点**。即**可分数据上 logistic loss 会无限增大参数以提高置信度**，标签早已全对。
- **附录 B**：$\log\dfrac{p_{\boldsymbol\theta}(\boldsymbol x)}{1-p_{\boldsymbol\theta}(\boldsymbol x)}=b+\boldsymbol x\boldsymbol w$：打分就是 **log-odds**；$p=1/2\Leftrightarrow$ 赔率 1 $\Leftrightarrow$ 打分 0。

## 学习提示

- 四个对象别混：$\ell_{0,1}$（单样本错误）、$\mathcal E$（总错误数）、$\ell_{\log}$（单样本代理）、$\mathcal L_{\log}$（总代理）。
- 打分决定标签，sigmoid 决定置信度；改平局规则只影响边界上的点，但会改变具体数字（如 $t=0$ 的错误数）。
- 本周 Lab 仍是回归，**线性分类 Lab 在 Week 4**。下一讲是神经网络，见 [W04 神经网络](FOAI-公开版-W04-神经网络.md)。
