---
title: FOAI W01 Lab｜Python 复习与类（上）
course: FOAI
week: 1
type: lab
tags:
  - AI-MSc
  - TB1
  - FOAI
  - Python
source: 课程材料的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: published
publication: public
updated: 2026-10-09
---

# FOAI W01 Lab｜Python 复习 + 类 Part 1

> [!IMPORTANT]
> **非官方学习笔记**：这是我在「Foundations of Artificial Intelligence」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：课程讲师。原始材料：[官方课程仓库](https://github.com/semtm0051/2026_27/tree/322787dec91f174303ccf90399bb6243a060eb7d/)。经课程讲师同意公开；不含练习解答与非公开材料（如课堂录像）。

> [!NOTE]
> **实验文件**
> - [`Week_01/Lab/Week_01_intro_to_python.ipynb`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_01/Lab/Week_01_intro_to_python.ipynb)（29 md / 26 code cells）
> - [`Week_01/Lab/Week_01_python_classes_part1.ipynb`](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_01/Lab/Week_01_python_classes_part1.ipynb)（23 md / 18 code cells）
> 两份都可直接在 Google Colab 打开（仓库 README 里有 badge 链接），无需 GPU。

本实验是 Python 速成，目标不是从零教语法，而是补齐后面写 PyTorch 代码会立刻用到的几个点。

---

## Notebook 1：Python 速览

### 控制流与函数
- `for i in range(3):` 依次取 `0,1,2`。
- `def f(x, y):` 定义函数；`return` 才是返回值，`print` 只是打印——**Colab 会自动显示 cell 最后一个表达式的值**，别把它和 `return` 混淆。

### list vs tuple（高频踩坑点）
| | list `[1,2,3]` | tuple `(1,2,3)` |
|---|---|---|
| 可变性 | **mutable**，可改元素 | **immutable**，不能改 |
| 用途 | 同构、会变化的序列 | 记录、多返回值、dict 键 |
| 索引 | 正索引 `0..n-1`，负索引 `-1` 是最后一个 | 同 |

> [!WARNING]
> **修改原对象 vs 返回新对象**
> - `sorted(a)` 返回**新排好序的 list**，不改 `a`；
> - `a.sort()` **原地排序**，返回 `None`。
> 用任何 list 方法前先确认它是 in-place 还是返回新对象。后面 PyTorch 的 tensor 也有同样的区分（`add_` vs `add`）。

### Matplotlib 最小集
- `import matplotlib.pyplot as plt`
- `plt.scatter(x, y)` 画散点（观测数据）；
- `plt.plot(x, yh)` 画折线（模型预测）；
- Week 3 lab 会反复用这个组合画"数据 vs 拟合线"。

### 实验后要能独立写
- 反转列表逐行打印（两种写法：正索引倒着数 / 负索引）。
- 不运行代码预测 10 段代码片段的输出（题目 2，覆盖 list 别名、method 调用、mutability 等）。

---

## Notebook 2：Python Classes（Part 1）

后面整个课程的神经网络都是 `torch.nn.Module` 的子类，类是绕不开的。这一节只讲最核心的四件事。

### 1. 类与实例
```python
class Adder:
    def __init__(self, a):      # 初始化器，构造时自动调用
        self.a = a
    def add_a(self, b):         # 方法
        return self.a + b
```
- `AdderInit(3)` 会自动调 `__init__`，Python 把新实例自己作为第一个参数传进去，命名为 `self`（只是惯例）。
- 定义时有三个参数 `(self, a, b)`，调用 `obj.add_a(1, 2)` 时只传两个——**self 由 Python 自动补**。

### 2. 属性挂在实例上，不挂在类上
- `self.a = a` 把参数 `a` 存到**当前实例**的属性里；
- 不同实例各自有一份 `a`，互不相干；
- 提到 `self.example_state` 但没赋值就访问，会报 `AttributeError`。

### 3. 方法：要么改状态，要么返回值
- `initialize_a(4)`：**修改**实例状态，无返回；
- `add_a(b)`：**返回**计算结果，不改状态。
- 这就是 list 的 `a.sort()`（改自己）vs `sorted(a)`（返回新对象）的类版本。

### 4. 课堂练习
- **练习 1**：预测 `AdderInit` 两个实例各自的打印行（状态隔离）。
- **练习 2**：写一个 `Counter` 类，有三个方法：`increment()`、`decrement()`、`value()`（返回当前计数）。状态在多次调用之间保留。

> [!TIP]
> **为什么课程现在教类**
> 后面 PyTorch 里：
> - 一个模型 = 一个类（继承 `nn.Module`），把权重等状态存在 `self` 上；
> - `model(x)` 让对象像函数一样被调用（part 2 会讲 `__call__`）；
> - 状态跨多次 forward/backward 调用持续存在——和这里的 `Counter` 一模一样。
