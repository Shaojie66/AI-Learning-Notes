---
title: FOAI W04 Lab｜用 PyTorch 实现线性分类
course: FOAI
week: 4
type: lab
tags:
  - AI-MSc
  - TB1
  - FOAI
  - 线性分类
  - PyTorch
  - 数值稳定性
source: 课程材料（Conor Houghton、Roberto Colomboni）的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: published
publication: public
updated: 2026-10-09
---

# FOAI W04 Lab｜PyTorch 线性分类实验

> [!IMPORTANT]
> **非官方学习笔记**：这是我在 University of Bristol「Foundations of Artificial Intelligence（SEMTM0051, 2026/27）」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：Conor Houghton、Roberto Colomboni。原始材料：[semtm0051/2026_27](https://github.com/semtm0051/2026_27/tree/322787dec91f174303ccf90399bb6243a060eb7d/)。经两位讲师同意公开；不含练习解答与非公开材料（如课堂录像）。

> [!NOTE]
> **实验文件与可靠度：high**
> [`Week_04/Lab/Week_04_linear_classification.ipynb`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_04/Lab/Week_04_linear_classification.ipynb)（22 个 cell：14 个 markdown、8 个 code）。
> **本周 Lab 对应 W03 的线性分类讲座，不是本周的神经网络讲座**（神经网络笔记见 [FOAI-W04-神经网络](FOAI-公开版-W04-神经网络.md)）。Lab 在 Google Colab 里跑，不需要 GPU，数据由 notebook 自己生成。
> 标注为「补充说明」的方框是我加的解释，不是 notebook 原文。

> [!TIP]
> **数值核对（2026-10-09）**
> 我用 PyTorch 2.10.0 按 notebook 的设定（随机种子 1、$N=100$）独立复现了它给出的所有数字，全部一致：朴素损失得到 `nan`、`logsigmoid(-100)` 与 `logsigmoid(-1000)` 为有限值、训练 100 步后的参数、错误数 3、损失 6.88。随机数在不同 PyTorch 版本间可能略有不同，如果你在 Colab 里看到的数字有细微差异，不一定是你写错了。我没有运行 notebook 本身，而是自己重写了等价的代码。

## 核心脉络

这个 Lab 做两件事：

1. **数值稳定性**：同一个数学公式，写成「先 sigmoid 再 log」会在得分很大或很小时算出 `nan`；换成 PyTorch 专门写好的 `logsigmoid` 就没事。**公式没变，只是算法换了**。
2. **手写训练**：用 W03 讲过的梯度 $\mathbf X^\top(\mathbf p-\mathbf y)$ 做梯度下降，训练一个线性分类器，再检查它的错误数，并验证「错误数不超过 $\mathcal L_{\log}/\log2$」这条上界。

## 0. 数据与记号

**数据**：100 个观测，每个有两个特征 $x_1,x_2$。前 50 个是类 0（围绕 $(-1,-1)$，噪声小），后 50 个是类 1（围绕 $(1,1)$，噪声大）。

| 代码里的名字 | 形状 | 含义 |
| --- | --- | --- |
| `inputs` | $(100,2)$ | 原始两个特征 |
| `y` | $(100,1)$ | 标签，前 50 个 0，后 50 个 1 |
| `X` | $(100,3)$ | **设计矩阵**：`inputs` 左边加一列全 1，它的每一行是 $(1,\boldsymbol x_i)$ |
| `theta` | $(3,1)$ | 参数列 $(b,w_1,w_2)^\top$ |

> [!TIP]
> **补充说明（非 notebook 原文）：为什么要加一列 1**
> 打分是 $s=b+w_1x_1+w_2x_2$。如果 $\mathbf X$ 的第一列全是 1，那么 $\mathbf X\boldsymbol\theta$ 的每一行就恰好是 $1\cdot b+x_1w_1+x_2w_2$。所以**一次矩阵乘法就算出全部 100 个打分**，截距 $b$ 也被当成普通参数一起拟合。

> [!TIP]
> **补充说明：画图那行代码容易看错**
> `plt.scatter(x=inputs[:, 0], y=inputs[:, 1], c=y[:, 0])` 里的 `y=` 是**纵坐标**，和名叫 `y` 的标签张量不是一回事。标签只用 `c=` 变成**颜色**。`y[:, 0]` 是因为 `y` 的形状是 $(N,1)$，取第 0 列得到一维的标签列表。

## 1. 数值稳定性

### 1.1 先把损失写出来

打分 $\boldsymbol s=\mathbf X\boldsymbol\theta$，概率 $p_i=\sigma(s_i)$，单个观测的 log-loss 是 $-y\log p-(1-y)\log(1-p)$。因为 $1-\sigma(s)=\sigma(-s)$，求和后的经验 logistic loss 可写成

$$\mathcal L_{\log}(\boldsymbol\theta)=-\sum_{i=1}^{N}\Big[y_i\log\sigma(s_i)+(1-y_i)\log\sigma(-s_i)\Big].$$

**用求和，不是平均**（和讲义一致）。标签 $y_i=1$ 时只剩第一项，$y_i=0$ 时只剩第二项。

朴素实现的思路：先算打分 `X @ theta`，再分别对 `sigmoid(scores)` 和 `sigmoid(-scores)` 取 `log`，乘上标签系数后求和取负。

> 按课程讲师要求，本文不收录练习解答代码；Question 1.1 的参考实现见[官方 notebook](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_04/Lab/Week_04_linear_classification.ipynb) 中对应题目的 Show answer。

### 1.2 为什么会出现 `nan`

用 $\boldsymbol\theta=(20,20,20)^\top$ 测试，应得到 **`nan`**（not a number，不是 Python 报错）。我复现得到的就是 `nan`。

### 1.3 找出原因

逐步看（我复现的数值，默认精度）：

| 表达式 | 结果 | 说明 |
| --- | --- | --- |
| `sigmoid(-50)` | $\approx1.9\times10^{-22}$ | 很小但还不是 0 |
| `sigmoid(-100)` | **`0.0`** | 小到计算机表示不了，变成了 0（下溢） |
| `log(0)` | **`-inf`** | 对 0 取对数得负无穷 |
| `0 * log(0)` = `0 * (-inf)` | **`nan`** | 浮点运算里 0 乘无穷是 `nan` |

数学上 $\sigma(-100)$ 是一个**很小的正数**，它的对数是有限的（约 $-100$）。问题出在：我们先算 `sigmoid`，那个小正数**在取对数之前就丢掉了**。把 $s$ 从 $-100$ 到 $100$ 画出 `log(sigmoid(s))` 时，曲线看着正常，但前几个值已经是 `-inf`（我复现的前 5 个值都是 `-inf`），所以「图看起来没问题」不代表数值没问题。

损失里每一项都乘了标签系数（$y_i$ 或 $1-y_i$，其中总有一个是 0）。`0 * (-inf)` 变成 `nan`，求和时 `nan` 不会被抹掉，整个损失就成了 `nan`。

### 1.4 解决：用 `logsigmoid`

`t.nn.functional.logsigmoid(s)` 计算的是同一个量 $\log\sigma(s)$，但**不经过「中间的 sigmoid 值」**，所以不会丢失。

| 表达式 | 复现结果 |
| --- | --- |
| `logsigmoid(-100)` | $-100.0$ |
| `logsigmoid(-1000)` | $-1000.0$ |

新的损失函数的思路：结构和 1.1 一样，只是把「`log` 套 `sigmoid`」换成直接对打分调用 `logsigmoid`。

> 按课程讲师要求，本文不收录练习解答代码；Question 1.5 的参考实现见[官方 notebook](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_04/Lab/Week_04_linear_classification.ipynb) 中对应题目的 Show answer。

在 $\boldsymbol\theta=(20,20,20)^\top$ 处我得到有限值 $\approx21.67$，而朴素版给出 `nan`。**要点**：把**打分**传给 `logsigmoid`，不是传 sigmoid 之后的概率。数学上的损失没有变，**只是换了算法**。之后整个 Lab 都用这个稳定的 `loss`。

> [!TIP]
> **补充说明：这条经验在后面还会遇到**
> 凡是「先做一个可能变成 0 或无穷大的中间量，再取对数」的写法，都应当警惕。更一般的原则是：数学上等价的两种写法，在计算机里不一定等价。

## 2. 手写梯度，训练分类器，检查结果

### 2.1 梯度

$$\nabla\mathcal L_{\log}(\boldsymbol\theta)=\mathbf X^\top\big(\mathbf p(\boldsymbol\theta)-\mathbf y\big),\qquad \mathbf p(\boldsymbol\theta)=\boldsymbol\sigma(\mathbf X\boldsymbol\theta).$$

结构和回归的梯度类似：**转置的设计矩阵** × 「**预测 − 真实**」。区别是这里的「预测」是概率，所以括号里是「预测概率 − 观测标签」。写成代码时，先算打分和概率，再用 `X.mT` 乘「概率 − 标签」。

> 按课程讲师要求，本文不收录练习解答代码；梯度函数的参考实现见[官方 notebook](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_04/Lab/Week_04_linear_classification.ipynb) 中对应题目的 Show answer。

**形状检查**：`X.mT` 是 $(3,100)$，括号里是 $(100,1)$，乘出来是 $(3,1)$，和 `theta` 同形状。第一个分量是对截距的导数，另外两个是对两个特征权重的导数。

**自检**：`grad(t.zeros(3, 1))` 应约为 $(0,\ -53.81,\ -51.29)^\top$。我复现得到 $(0.0,\ -53.810,\ -51.289)$。第一个分量是 0，因为数据里两类各 50 个，零打分时 $p=0.5$，$y=1$ 与 $y=0$ 的差 $\pm0.5$ 在截距这一列恰好抵消。

> [!TIP]
> **补充说明：为什么这里直接用 `sigmoid` 没问题**
> 梯度里的 $p-y$ 不需要取对数，所以不会出现 1.3 里的 `log(0)` 问题。数值不稳定的只是**损失**这一步。

### 2.2 训练并检查

从全 0 的参数开始，学习率 $\eta=0.01$，用 $\boldsymbol\theta\leftarrow\boldsymbol\theta-\eta\nabla\mathcal L_{\log}$ 更新 100 次，每 10 步打印一次稳定的损失。我复现的结果（只核对了这两个点）：第 10 步 $12.3538$，第 100 步 $6.8786$。

之后用 $\boldsymbol p>0.5$ 的规则预测类别（**严格大于**，平局判类 0，和 W03 更新后的规则一致），数出错误数（预测标签与真实标签不同的个数），再和上界比较。

> 按课程讲师要求，本文不收录练习解答代码；训练与检查部分的参考实现见[官方 notebook](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_04/Lab/Week_04_linear_classification.ipynb) 中对应题目的 Show answer。

| 量 | notebook 给出的 | 我的复现 |
| --- | --- | --- |
| 学到的参数 $(b,w_1,w_2)$ | $\approx(1.1950,\ 2.4771,\ 2.6349)$ | $(1.19497,\ 2.47708,\ 2.63492)$ |
| 总 logistic loss | $\approx6.88$ | $6.8786$ |
| 训练错误数 | $3$ | $3$ |
| $\mathcal L_{\log}/\log 2$（错误数的上界） | $\approx9.92$ | $9.9237$ |

**这说明两点**：

1. **logistic loss 不是错误数**：损失 $6.88$，错误只有 $3$ 个。
2. **上界成立**：$3\le9.92$。这就是 W03 的结论 $\mathcal E(\boldsymbol\theta)\le\mathcal L_{\log}(\boldsymbol\theta)/\log2$，在这次训练里的一个具体验证。损失小，错误数就受控；但**不是等式**。

最后，notebook 提供了画决策边界的函数 `plot_decision_boundary`（标注为 supplied、non-examinable，不用自己写）。它在网格上算出打分，画出打分为 0 的曲线。

> [!NOTE]
> **补充说明：这里的边界是一条直线**
> 因为打分是仿射的 $b+x_1w_1+x_2w_2$，决策边界是直线。要让边界弯曲，就需要 W04 讲座里的隐藏层，见 [FOAI-W04-神经网络](FOAI-公开版-W04-神经网络.md) §5.2 的 XOR 例子。

## 常见卡点

- **`nan` 不是代码报错**，是一个正常的浮点结果。看到它先想「某一步是不是出现了 `0 * inf` 或 `log(0)`」。
- **别把 `loss_sigmoid` 和 `loss` 混用**：Lab 后半段要用稳定版 `loss`。
- **形状**：`X` 是 $(100,3)$，`theta` 是 $(3,1)$，`X @ theta` 是 $(100,1)$。梯度必须回到 $(3,1)$。
- **标签张量的形状是 $(N,1)$**，画图和比较时常要取 `y[:, 0]`；计算损失和梯度时则要保持 $(N,1)$，别乱压平。
- **用求和，不是平均**：这和你以后在别的地方见到的「平均损失」不同；梯度下降的步长也因此不同。

## 关键自查点

- [ ] 能解释为什么 `sigmoid(-100)` 是 0 但数学上不是。
- [ ] 能说出 `logsigmoid` 为什么能避免问题。
- [ ] 知道 `grad(0)` 的第一个分量为什么是 0。
- [ ] 能解释为什么 6.88 的损失对应只有 3 个错误。
- [ ] 能说出打分恰好为 0 时判哪一类（类 0）。
