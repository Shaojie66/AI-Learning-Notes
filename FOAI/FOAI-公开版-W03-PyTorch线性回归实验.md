---
title: FOAI W03 Lab｜用 PyTorch 实现线性回归
course: FOAI
week: 3
type: lab
tags:
  - AI-MSc
  - TB1
  - FOAI
  - 线性回归
  - PyTorch
source: 课程材料（Conor Houghton、Roberto Colomboni）的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: published
publication: public
updated: 2026-10-09
---

# FOAI W03 Lab｜PyTorch 线性回归实验

> [!IMPORTANT]
> **非官方学习笔记**：这是我在 University of Bristol「Foundations of Artificial Intelligence（SEMTM0051, 2026/27）」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：Conor Houghton、Roberto Colomboni。原始材料：[semtm0051/2026_27](https://github.com/semtm0051/2026_27/tree/322787dec91f174303ccf90399bb6243a060eb7d/)。经两位讲师同意公开；不含练习解答与非公开材料（如课堂录像）。

> [!NOTE]
> **实验文件**
> [`Week_03/Lab/Week_03_linear_regression.ipynb`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_03/Lab/Week_03_linear_regression.ipynb)（19 md / 10 code）。
> 配套讲义 **Linear Classification**（Week 3 lecture）的 PDF 已上传，笔记见 [FOAI-W03-线性分类](FOAI-公开版-W03-线性分类.md)。本周 Lab 做的是回归，线性分类 Lab 在 Week 4。

这个 lab 把 W02 讲义里的三件事用 CPU PyTorch 从头写一遍：
① 直线拟合的解析解；② 二次特征拟合；③ 手写梯度下降。**不使用 `torch.nn` / 自动求导**——所有梯度都按讲义公式手算。

> 按课程讲师要求，本文不收录练习解答代码；各题的参考实现见[官方 notebook](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_03/Lab/Week_03_linear_regression.ipynb) 中对应题目的 Show answer。

---

## 0. 画图约定
- 观测数据：`plt.scatter(x, y)`（散点）；
- 模型预测：`plt.plot(x, yh)`（折线）；
- 此时 `x, y, yh` 都是**普通 CPU tensor**，还没到 autograd。

## 1. 拟合直线

### 1.1 构造设计矩阵
讲义里的 $\boldsymbol{X}$ 要在每个观测前加一列 1（用 `torch.stack` 把全 1 列和 `x` 拼起来，形状 $(N,2)$），对应 $\theta = (b, a)^\top$。

### 1.2 解析解
按 $\theta^* = (X^\top X)^{-1} X^\top y$ 求解，工程上用 `torch.linalg.solve` 而不是显式求逆。

> [!TIP]
> **为什么用 `solve` 而不是 `.inverse()`**
> `torch.linalg.solve(A, b)` 解 $A\theta=b$ 不显式求逆，数值更稳、更快。讲义的闭式公式是数学表达，工程上一律 `solve`。

### 1.3 / 1.4 画图与矩阵乘法预测
- 先手写出预测 `yh`，和数据画在一起；
- 再用 $\boldsymbol{X}\theta^*$ 算一遍预测，确认和手算一致。

## 2. 拟合二次函数
按讲义 §4.4，选特征 $\phi(x) = (1, x, x^2)$，设计矩阵 $\Phi$ 形状 $(N,3)$：
- 模型对 $x$ 是二次的，但对 $\boldsymbol{w}$ 仍是线性的；
- 设计矩阵多加一列 $x^2$ 而已，正规方程不变。

## 3. 手写梯度下降

### 3.1 损失
对任意系数 $\boldsymbol{w}$，损失是残差平方和，对应讲义 $\mathcal{L}(\theta) = \|\Phi w - y\|_2^2$。

### 3.2 梯度
讲义已推出 $\nabla \mathcal{L} = 2 \Phi^\top(\Phi w - y)$。
**这一步是这个 lab 的核心**——对照讲义公式 (68) 一行行写。

### 3.3 更新
从 $\boldsymbol{w}^{(0)} = \boldsymbol{0}$ 出发，学习率 $\eta=0.04$，重复 $\boldsymbol w\leftarrow\boldsymbol w-\eta\nabla\mathcal L$：
- 对照讲义 (70)：先算残差 $\boldsymbol{r}^{(t)} = \Phi w^{(t)} - y$，再更新；
- 所有系数（包括截距）在同一轮里一起更新；
- 观察损失随迭代下降的曲线，以及和解析解 $w^*$ 的距离。

## 关键自查点

- [ ] 设计矩阵第一列是不是全 1？别漏，漏了就没有截距。
- [ ] `X.T @ X` shape 是 $(d+1)\times(d+1)$，别和 `X @ X.T`（$N\times N$）搞混。
- [ ] 梯度里的因子 2 有没有漏？漏了等价于学习率减半，仍然收敛但慢。
- [ ] 学习率 $\eta$ 太大会发散（讲义 §5.1 的 $F(a)=100a^2$ 例子）。
- [ ] 比较梯度下降收敛后的 $\boldsymbol{w}$ 和解析解 $w^*$：应该几乎重合。
