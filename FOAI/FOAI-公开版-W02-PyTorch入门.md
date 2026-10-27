---
title: FOAI W02 Lab｜PyTorch 入门与 GPU
course: FOAI
week: 2
type: lab
tags:
  - AI-MSc
  - TB1
  - FOAI
  - PyTorch
source: 课程材料的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: published
publication: public
updated: 2026-10-09
---

# FOAI W02 Lab｜PyTorch 入门（Tensors / 类 Part 2 / GPU）

> [!IMPORTANT]
> **非官方学习笔记**：这是我在「Foundations of Artificial Intelligence」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：课程讲师。原始材料：[官方课程仓库](https://github.com/semtm0051/2026_27/tree/322787dec91f174303ccf90399bb6243a060eb7d/)。经课程讲师同意公开；不含练习解答与非公开材料（如课堂录像）。

> [!NOTE]
> **实验文件（按顺序做）**
> 1. [`Week_02/Lab/Week_02_intro_to_pytorch.ipynb`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_02/Lab/Week_02_intro_to_pytorch.ipynb)（27 md / 27 code）
> 2. [`Week_02/Lab/Week_02_python_classes_part2.ipynb`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_02/Lab/Week_02_python_classes_part2.ipynb)（5 md / 15 code，subclassing 标注 [Non examinable]）
> 3. [`Week_02/Lab/Week_02_gpus_with_pytorch.ipynb`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_02/Lab/Week_02_gpus_with_pytorch.ipynb)（8 md / 52 code，需在 Colab 选 GPU runtime）

---

## Notebook 1：PyTorch = NumPy + GPU + 自动求导

### 1. Tensor 是什么
`torch.Tensor` 是一个多维数字数组，标量、向量、矩阵、高维数组都是同一个类。
- 初始化：`torch.zeros(2,3)`、`torch.ones(...)`、`torch.randn(...)`（标准正态）。
- **shape**：每个维度的大小，顺序就是维度顺序；矩阵 `(3,4)` = 3 行 4 列。

### 2. Broadcasting（高频踩坑）
把不同 shape 的 tensor 按元素运算时，PyTorch 从**最右边**开始对齐维度：
- 维度相等，或其中一个是 1，才能对齐；
- size 为 1 的维度会被"广播"复制到另一边的大小。

例子：
- `X` shape `(2, 3, 4)` + `Y` shape `(4,)`：把 Y 看作 `(1,1,4)`，结果 `(2,3,4)`，$Z_{ijk} = X_{ijk} + Y_k$。
- `X` shape `(2,3,4)` + `Y` shape `(2,)`：**不能**自动对齐第一维。要让 Y 匹配第一维，得 reshape 成 `(2,1,1)`，此时 $Z_{ijk} = X_{ijk} + Y_{i00}$。
- 索引里 `:` 表示"保留这一维全部"，`Y.view(2,1,1)` 或 `Y[:, None, None]` 插入大小为 1 的维度。

> [!WARNING]
> **长度相等 ≠ 维度自动匹配**
> 长度为 2 的向量**不会**自动匹配矩阵第一维（也是 2）。必须显式 reshape 成 `(2,1)` 或 `(2,1,1)`。

### 3. 逐元素 vs 矩阵乘
- `A * B`：逐元素乘（broadcasting 后要求 shape 兼容）。
- `A @ B`（或 `A.matmul(B)`）：矩阵乘法，对应 [W02 讲义](FOAI-公开版-W02-线性回归.md) 里的 $X\theta$。

### 4. Reduction 操作
`sum / mean / prod / std / var` 把多维张量压成标量或低维张量。
- `A.mean(0)` 沿第 0 维求平均，剩下后面的维；
- **`keepdim=True`**：被消掉的维度保留为 size 1，方便后面广播回去。
  - 想从 `A`（shape `(2,3,4)`）里减去均值：
    - 沿 dim=0 求 mean 用 `keepdim=True` 得到 `(1,3,4)`，直接广播相减；
    - 沿 dim=2 求 mean 不加 keepdim 得到 `(2,3)`，和 `(2,3,4)` 对不齐，要手动插入维度。

### 5. 课堂练习
- (a) 预测几个 tensor 运算后的 shape；
- (b) `A.reshape(2,3,20)`（总元素数不变才能 reshape）；
- (c) 用 broadcasting 把 `(2,3)` 的 B 逐元素乘到 `(2,3,4)` 的 A 上；
- (d) 用 PyTorch 算 $d = \frac12 \sum_i\sum_k [\sin(\sum_j C_{ijk})]^2$（三种等价写法：循环 / 一次 sum / einsum）。

---

## Notebook 2：Python Classes Part 2

承接 [W01 的类基础](FOAI-公开版-W01-Python复习与类.md)，只加两件后面立刻要用的事：

### 1. 让对象像函数一样被调用：`__call__`
```python
class LinearModel:
    def __init__(self, w):
        self.w = w
    def __call__(self, x):
        return x @ self.w

model = LinearModel(w)
y = model(x)     # 自动调 __call__
```
- 对象自己存状态（`self.w`），又能像函数一样被 `model(x)` 调用；
- **这就是 `nn.Module` 的工作方式**：`model(x)` 实际调 `model.forward(x)`。

### 2. Subclassing（考试不要求）
一个类可以继承另一个类，复用父类的属性和方法，再加上自己的。后面 `class MyModel(nn.Module):` 就是这个机制。

---

## Notebook 3：GPU 与内存

> [!NOTE]
> **在 Colab 里跑**
> Runtime → Change runtime type → Hardware accelerator 选 GPU。

### 1. 什么时候 GPU 有用
- **大张量的整体操作**：GPU 并行算数快；
- **很多很多小操作**：CPU↔GPU 传输和 kernel launch 开销盖过并行收益，反而慢。

### 2. 两个坑
1. **在 GPU 上直接初始化**，不要在 CPU 建大 tensor 再 `.to('cuda')`——既占 CPU 内存又多一次传输。
   ```python
   torch.randn(1000, device='cuda')   # 好
   torch.randn(1000).to('cuda')      # 慢且浪费
   ```
2. **GPU 显存有限**，OOM 是常事。`del tensor` + `torch.cuda.empty_cache()` 可手动释放。

### 3. 赋值不拷贝，in-place 操作
- `b = a` 两个名字指向**同一个 tensor**（和 list 一样）；要真正复制用 `a.clone()`。
- **in-place** 操作（`a.add_(1)`、`a += 1`）直接改原 tensor，不开新内存；返回新 tensor 的是 `a.add(1)`。
- 训练时显存吃紧时，in-place 能省内存，但要小心 autograd 保留的中间引用。
