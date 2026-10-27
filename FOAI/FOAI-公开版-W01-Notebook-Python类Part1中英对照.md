---
title: Week_01 Python Classes Part 1（中英对照）
course: FOAI
week: 1
type: lab-notebook-bilingual
tags:
  - AI-MSc
  - TB1
  - FOAI
  - notebook
  - 中英对照
source: 课程材料的个人非官方翻译与学习笔记
original: https://github.com/semtm0051/2026_27
status: published
publication: public
updated: 2026-10-09
---

# Python classes（Python 类，第 1 部分）

> [!IMPORTANT]
> **非官方学习笔记**：这是我在「Foundations of Artificial Intelligence」课程中的个人中文翻译与学习笔记，**不是官方课程材料**，可能有错漏，以官方原文为准。
> 原课程材料作者：课程讲师。原始材料：[官方课程仓库](https://github.com/semtm0051/2026_27/tree/322787dec91f174303ccf90399bb6243a060eb7d/)。经课程讲师同意公开；不含练习解答与非公开材料（如课堂录像）。

> **Week 1 · Python refresher · Notebook 2 of 2 · Classes, part 1**
> 第 1 周 · Python 复习 · 第 2/2 个 notebook · 类（上）

Complete **`Week_01_intro_to_python.ipynb`** first. Continue working in **Google Colab**, running the cells from top to bottom.

> 请先完成 `Week_01_intro_to_python.ipynb`。继续在 Google Colab 里从上到下运行 cell。

Python classes are important for us because PyTorch—the Python library we will use to implement neural networks—uses objects and classes throughout. We only need a basic understanding of them here.

> 类对我们很重要，因为 PyTorch（我们后面用来实现神经网络的 Python 库）全程都在使用对象和类。这里只需要掌握基础。

A **class** defines a type of object. For example, Python's `list` class is the type of objects such as `[1, 2, 3]`. We can also define our own classes.

> **类（class）**定义一种对象类型。例如 Python 的 `list` 类就是 `[1, 2, 3]` 这类对象的类型。我们也可以自己定义类。

An **instance** is one particular object of a class. We will distinguish:

> **实例（instance）**是某个类的一个具体对象。我们区分两件事：

* **State:** the data associated with that instance, stored in **attributes**.
* **Methods:** functions defined in the class that operate on an instance and its state.

> * **状态（state）：**与该实例关联的数据，存放在**属性（attribute）**里。
> * **方法（method）：**定义在类里、作用于某个实例及其状态的函数。

Different instances can have different state while using the same method definitions.

> 不同实例可以有不同的状态，但使用同一套方法定义。

## The simplest possible class（最简单的类）

We start with a class whose body defines no attributes or methods. `SimplestPossibleClass` is the name we have chosen.

> 我们先写一个类体里什么都没有的类，取名 `SimplestPossibleClass`。

As with `for` and `def`, the colon `:` introduces an indented block. Python requires a statement in that block, so we use `pass`, which means "do nothing".

> 和 `for`、`def` 一样，冒号 `:` 后面是一个缩进块。Python 要求块里至少有一条语句，所以用 `pass`（意思是"什么都不做"）占位。

```python
# Define the class. This does not yet create the instances used below.
class SimplestPossibleClass:
    pass  # A placeholder: the class body has nothing else to do.
```

Writing `SimplestPossibleClass()` creates a new instance of this class. The parentheses are needed even though there are no arguments.

> 写 `SimplestPossibleClass()` 就会创建这个类的一个新实例。即使没有参数，括号也不能省。

Calling it twice gives us **two distinct objects of the same class**:

> 调用两次，就得到**同一个类的两个不同对象**：

```python
# Create one instance and give us a name with which to access it.
simple_object1 = SimplestPossibleClass()

# Call the class again: this creates a SECOND, distinct instance.
simple_object2 = SimplestPossibleClass()
```

### Adding and reading attributes（添加与读取属性）

An **attribute** is a named item associated with an object. In `simple_object1.a`, the part before the dot identifies the object, and the part after it identifies the attribute.

> **属性（attribute）**是挂在对象上的一个命名条目。`simple_object1.a` 里，点号前是对象，点号后是属性名。

For this simple class, we can add instance attributes by assignment; they do not need to be declared in the class body first:

> 对这种简单类，我们可以直接赋值来添加实例属性，不需要先在类体里声明：

* `simple_object1.a = 3` creates the attribute `a` on that instance and assigns it the value `3`.
* `simple_object1.a` reads the attribute. Reading it before it exists raises an `AttributeError`.

> * `simple_object1.a = 3` 在该实例上创建属性 `a`，并赋值 `3`。
> * `simple_object1.a` 读取该属性；如果属性不存在就先读，会抛 `AttributeError`。

This behaviour is allowed for the simple classes we define here; other classes can restrict which attributes may be added.

> 我们这里写的简单类允许这样做；其他类可以限制能添加哪些属性。

```python
# Create `a` on the FIRST object, then read and print it.
simple_object1.a = 3
print(simple_object1.a)  # Output: 3.

# Create `b` on the SECOND object, then read and print it.
simple_object2.b = 2
print(simple_object2.b)  # Output: 2.

# Adding an attribute to one instance does not add it to the other.
# print(simple_object1.b)  # AttributeError: only simple_object2 has `b`.
# print(simple_object2.a)  # AttributeError: only simple_object1 has `a`.
```

```text
3
2
```

## How methods are called（方法是怎么调用的）

In the first notebook, we called the list method `a.sort()`. Now we will define methods of our own.

> 第一个 notebook 里我们调用过 list 的方法 `a.sort()`。现在我们自己定义方法。

A method is still a function, but we define it **inside a class** and call it **through an instance**. For the instance methods used here:

> 方法本质上还是函数，但它定义在**类内部**，并且要**通过实例来调用**。本课程用到的实例方法约定：

* `def mymethod(self, a, b):` defines the method with parameters `self`, `a` and `b`.
* The definition is indented inside the class; the method body is indented one level further.
* `self` receives the instance on which we call the method. The name `self` is a convention, not a Python keyword.

> * `def mymethod(self, a, b):` 定义一个方法，参数为 `self`、`a`、`b`。
> * 方法定义要缩进在类里面；方法体再缩进一层。
> * `self` 接收调用该方法的那个实例。`self` 这个名字只是惯例，不是 Python 关键字。

**String syntax in the next cell:** the `f` before a quoted string makes it an **f-string**. Python inserts the value of each expression inside `{...}` into the text. Thus, `f"State on the object: {self.example_state}"` includes the value of that instance's `example_state` attribute.

> **下一个 cell 的字符串语法：**字符串前面加 `f` 就是 **f-string**。Python 会把 `{...}` 里表达式的值插进去。例如 `f"State on the object: {self.example_state}"` 会自动换成该实例 `example_state` 属性的值。

```python
class ClassWithOneMethod:
    def mymethod(self, a, b):
        # Read data from the instance received as `self`.
        print(f"State on the object: {self.example_state}")

        # `a` and `b` are inputs to this call, not instance attributes.
        print(f"First explicit argument (a): {a}")
        print(f"Second explicit argument (b): {b}")
```

Now we create three instances and assign an `example_state` attribute to the first two.

> 现在创建三个实例，并给前两个分别赋一个 `example_state` 属性。

Mentioning `self.example_state` in the method does **not** create that attribute. The method will try to read it when called.

> 在方法里写到 `self.example_state` **不会**自动创建这个属性；调用方法时它只会去读，读不到就报错。

```python
obj1 = ClassWithOneMethod()
obj2 = ClassWithOneMethod()
obj3 = ClassWithOneMethod()

# Same attribute name, but a separate assignment on each instance.
obj1.example_state = "red"
obj2.example_state = "blue"

# We have not yet assigned obj3.example_state.
```

The usual call syntax is `object.method(arguments)`.

> 标准调用写法是 `对象.方法(参数)`。

We defined `mymethod` inside the class, not as a standalone function. A call using just the name `mymethod` would therefore fail here:

> `mymethod` 是定义在类里的，不是独立函数。所以直接写 `mymethod(...)` 会报错：

```python
# This name has not been defined as a standalone function in the notebook.
# mymethod(obj1, 1, 2)  # NameError: name 'mymethod' is not defined.
```

Instead, call the method through the instance. The dot selects the method on `obj1`; the parentheses call it with the explicit arguments `1` and `2`.

> 正确做法是通过实例调用：点号选中 `obj1` 上的方法，括号里传入显式参数 `1` 和 `2`。

```python
# Python supplies obj1 as the first argument, corresponding to `self`.
obj1.mymethod(1, 2)
```

```text
State on the object: red
First explicit argument (a): 1
Second explicit argument (b): 2
```

The definition has three **parameters** (`self`, `a`, `b`), but this call supplies only two **explicit arguments** (`1`, `2`). Python supplies the instance automatically:

> 方法定义有三个**形参**（`self`、`a`、`b`），但调用时只传了两个**实参**（`1`、`2`）。Python 自动把实例填给 `self`：

| Parameter（形参） | Value in this call（本次调用的取值） |
| --- | --- |
| `self` | the object referred to by `obj1`（即 `obj1` 这个对象） |
| `a` | `1` |
| `b` | `2` |

So `self.example_state` reads `obj1.example_state`, whose value is `"red"`. These are the values passed to this call, not assignments we need to write ourselves.

> 所以 `self.example_state` 读的就是 `obj1.example_state`，值是 `"red"`。这些值是 Python 自动帮我们填好的，不用自己写。

Do not pass `obj1` again inside the parentheses: Python already supplies it for `self`. Outside a method, use a name such as `obj1` to access the object; `self` is simply a parameter name inside the method.

> 括号里**不要再传** `obj1`：Python 已经把它作为 `self` 传进去了。在方法外面用 `obj1` 这个名字访问对象；`self` 只是方法内部的一个形参名。

## Using methods（使用方法）

A method can use the stored data to compute a result or change those data. This is the distinction we saw with `sorted(a)` and `a.sort()`.

> 方法要么用存好的数据**算出一个结果返回**，要么**直接改这些数据**。这就是我们在 list 那里见过的 `sorted(a)`（返回新列表）和 `a.sort()`（原地改）的区别。

Our `Adder` class stores a number in the instance attribute `a` and defines three methods:

> 我们的 `Adder` 类把一个数存在实例属性 `a` 里，定义三个方法：

| Method（方法） | Purpose（作用） |
| --- | --- |
| `initialize_a(a)` | Set the instance's stored number.（设置实例里存的数） |
| `add_a(b)` | Return that number plus `b`, without changing it.（返回该数加 `b` 的结果，不改动原值） |
| `increment_a()` | Increase the stored number by one.（把存的数加 1） |

In this first version, creating an `Adder` does **not** set its attribute `a`. We must initialise that attribute before the other methods can read it.

> 第一个版本里，创建 `Adder` 时**不会**自动设置属性 `a`。必须先手动初始化，其他方法才能读它。

```python
class Adder:
    def add_a(self, b):
        # Read this instance's stored number and return the sum.
        # This does NOT assign a new value to self.a.
        return self.a + b

    def initialize_a(self, a):
        # Left: an attribute on the instance. Right: this call's input.
        self.a = a

    def increment_a(self):
        # Read the current value, add 1, then store the result back.
        self.a = self.a + 1
```

### Why write `self.a = a`?（为什么要写 `self.a = a`？）

The two occurrences of `a` have different roles. In `adder1.initialize_a(4)`, `self` receives `adder1` and the parameter `a` receives `4`. The assignment `self.a = a` therefore stores `4` in the attribute `adder1.a`.

> 两个 `a` 角色不同。调用 `adder1.initialize_a(4)` 时，`self` 接的是 `adder1`，形参 `a` 接的是 `4`。所以 `self.a = a` 就是把 `4` 存进 `adder1.a` 这个属性。

The parameter `a` is a **local name** for this call's input; `self.a` is an **attribute on the instance**, available to later method calls. The matching spelling is just a naming choice: with a parameter called `value`, we would write `self.a = value` instead.

> 形参 `a` 只是这次调用输入的**局部名字**；`self.a` 是实例上的**属性**，后续方法调用都能用。两者拼写相同只是命名习惯——如果形参叫 `value`，就写成 `self.a = value`。

Similarly, `self.a = self.a + 1` reads the current attribute value, adds one, and assigns the result back to that attribute.

> 同理，`self.a = self.a + 1` 是先读当前属性值，加 1，再存回这个属性。

`initialize_a` is a name we chose. Python does **not** call it automatically when we create an `Adder`.

> `initialize_a` 只是我们自己取的名字。创建 `Adder` 时 Python **不会**自动调用它。

```python
# Create two instances. Neither has an attribute `a` yet.
adder1 = Adder()
adder2 = Adder()
```

```python
adder1.initialize_a(4)  # Now adder1.a is 4.
adder1.add_a(5)        # Returns 4 + 5 = 9; adder1.a is still 4.
```

```text
9
```

The first call set `adder1.a`; the second returned a sum without changing that attribute. Colab displays `9` because that returned value is the cell's final expression.

> 第一句把 `adder1.a` 设成 4；第二句返回了一个和，但没改属性。Colab 显示 `9` 是因为这个 cell 的最后一个表达式有返回值。

We can also set the attribute directly, without calling `initialize_a`:

> 也可以不通过方法，直接给属性赋值：

```python
adder2.a = 6     # Set the SECOND instance's state directly.
adder2.add_a(5)  # Returns 6 + 5 = 11; adder2.a is still 6.
```

```text
11
```

### Modifying the stored value（修改存储的值）

Now compare an update through a method with a direct update. The comments assume the preceding cells have been run once, in order. Rerunning an update cell applies the update again.

> 比较一下"通过方法更新"和"直接更新"。下面的注释假设前面的 cell 都按顺序跑过一遍；重跑更新 cell 会再更新一次。

```python
# Modify the first instance THROUGH A METHOD.
print(adder1.add_a(5))  # Prints 9: the stored value is still 4.
adder1.increment_a()   # Changes adder1.a from 4 to 5.
adder1.add_a(5)        # Now returns 10.
```

```text
9
```

```text
10
```

```python
# Modify the second instance DIRECTLY.
print(adder2.add_a(5))  # Prints 11: the stored value is still 6.
adder2.a = adder2.a + 1  # Changes adder2.a from 6 to 7.
adder2.add_a(5)         # Now returns 12.
```

```text
11
```

```text
12
```

At this point, `adder1.a` is `5` and `adder2.a` is `7`. Updating one instance has not changed the other.

> 此时 `adder1.a` 是 5，`adder2.a` 是 7。改一个实例不会影响另一个。

`add_a` **returns a computed number**. By contrast, `initialize_a` and `increment_a` **change the stored data** and have no `return` statement, so they return `None`. As with `a.sort()` in the first notebook, returning `None` does not mean that nothing happened.

> `add_a` **返回一个算出来的数**；而 `initialize_a` 和 `increment_a` **修改存储的数据**，没有 `return`，所以返回 `None`。和第一个 notebook 里的 `a.sort()` 一样：返回 `None` 不代表什么都没发生——数据其实已经改了。

## Initialisers（初始化器 `__init__`）

So far, setting up an `Adder` has required two separate steps:

> 到目前为止，初始化一个 `Adder` 需要分两步：

```python
adder = Adder()
adder.initialize_a(2)
```

or, using direct assignment:

> 或者直接赋值：

```python
adder = Adder()
adder.a = 2
```

Python provides a **special method** named `__init__` to initialise a newly created instance automatically. The name has **two underscores before and two after** `init`.

> Python 提供了一个**特殊方法**叫 `__init__`，用来在创建实例时自动初始化。这个名字在 `init` 前后各有**两个下划线**。

`AdderInit` below has the same `add_a` and `increment_a` methods as `Adder`, but replaces the manually called `initialize_a` with `__init__`.

> 下面的 `AdderInit` 和 `Adder` 一样有 `add_a`、`increment_a`，但把手动调用的 `initialize_a` 换成了 `__init__`。

```python
class AdderInit:
    def __init__(self, a):
        # Python calls this method automatically to initialise a new instance.
        # The print is only here to make that automatic call visible.
        print("Inside __init__!!!")
        self.a = a  # The same assignment we used in initialize_a.

    def add_a(self, b):
        return self.a + b

    def increment_a(self):
        self.a = self.a + 1
```

When we write `AdderInit(3)`, Python creates a new instance and automatically calls `__init__`, supplying that instance for `self` and `3` for `a`.

> 写 `AdderInit(3)` 时，Python 创建一个新实例，并自动调用 `__init__`，把新实例填给 `self`，把 `3` 填给 `a`。

The method prints its message and sets the new instance's attribute `a` to `3`. The assignment below gives us the name `adder` for that initialised instance:

> 这个方法打印一句提示，并把新实例的属性 `a` 设成 3。下面的赋值把这个已初始化的实例命名为 `adder`：

```python
# Create an instance and initialise it with a = 3.
adder = AdderInit(3)
```

```text
Inside __init__!!!
```

The message confirms that `__init__` ran without a separate call from us.

> 这句输出证明 `__init__` 在我们没有手动调用的情况下自动跑了。

The code **inside** `__init__` is ordinary method code: `self.a = a` works just as it did in `initialize_a`. What is special is that Python recognises the name `__init__` and calls the method automatically during instance creation. Unlike `self`, this particular spelling is required for that behaviour.

> `__init__` **里面**的代码和普通方法没区别：`self.a = a` 跟在 `initialize_a` 里一样。特殊之处在于 Python 认得 `__init__` 这个名字，会在创建实例时自动调用它。和 `self` 不同，这个拼写是**固定的**，写错就不会触发自动调用。

The attribute `adder.a` is now set, so we can use it immediately:

> 属性 `adder.a` 已经设好，可以直接用：

```python
print(adder.a)   # 3: the attribute was set by __init__.
adder.add_a(6)   # Returns 9, without changing adder.a.
```

```text
3
```

```text
9
```

## Exercises（练习）

These two short exercises use only the class syntax introduced above. First predict what existing code does; then write a small class yourself.

> 两个小练习只用到上面讲过的类语法。先预测已有代码会做什么，再自己写一个类。

Try each exercise in its code cell before opening **Show answer**. For the first exercise, write your predictions as comments before running the code. Opening an answer does not execute its code: copy any code you want to run into the exercise cell.

> 在点开"显示答案"之前，先在代码 cell 里自己做。第一题先把预测写成注释再跑。点开答案块不会自动执行代码——要把想跑的代码复制到练习 cell 里。

### 1. Two instances, two stored values（两个实例，两份各自的状态）

Use the `AdderInit` class already defined above. **Without running the following code**, predict every printed line, including the messages from `__init__`.

> 用上面已经定义好的 `AdderInit` 类。**不要运行下面这段代码**，预测每一行打印结果，包括 `__init__` 的提示信息。

```python
left = AdderInit(2)
right = AdderInit(10)

result = left.add_a(5)
left.increment_a()

print(result)
print(left.a, right.a)
print(right.add_a(left.a))
print(left.a, right.a)
```

Then answer:

> 然后回答：

1. Which instance does `self` refer to in each call to `add_a` and `increment_a`?
2. Does `left.increment_a()` change `right.a` or the number already stored in `result`? Explain why.
3. Does `right.add_a(left.a)` change either instance's stored value?

> 1. 每次调用 `add_a` 和 `increment_a` 时，`self` 分别指哪个实例？
> 2. `left.increment_a()` 会改 `right.a` 吗？会改已经存在 `result` 里的数吗？解释原因。
> 3. `right.add_a(left.a)` 会改任何一个实例存的值吗？

Write your predictions below, then copy and run the code to check them. The first two lines create fresh instances, so you can rerun the whole example from the same starting state.

> 把预测写在下面，再复制代码运行验证。前两行会创建全新实例，所以可以从同样的初始状态反复重跑。

```python
# Write your predicted output and explanations as comments here.

# Then copy and run the code from Exercise 1 to check your predictions.
```

> [!NOTE]
> 按课程讲师要求，本文不收录练习解答。Exercise 1 的答案见[官方 notebook](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_01/Lab/Week_01_python_classes_part1.ipynb) 中的 Show answer。

### 2. Write a class that keeps a running total（写一个累加器类）

Write a class named **`Counter`** with these three methods:

> 写一个叫 **`Counter`** 的类，包含三个方法：

| Method（方法） | Required behaviour（要求行为） |
| --- | --- |
| `__init__(self, start)` | Store `start` in the instance attribute `value`. Do not print anything.（把 `start` 存到实例属性 `value` 里，不要打印任何东西） |
| `add(self, amount)` | Increase this instance's stored value by `amount`. Do not return a value.（把该实例的 `value` 加上 `amount`，不返回值） |
| `read(self)` | Return this instance's current stored value, without changing it.（返回当前 `value`，不改动它） |

Here `add` must **change the stored value**. This differs from `AdderInit.add_a`, which only returns a sum. You can use the syntax from the earlier examples; no imports or inheritance are needed.

> 这里的 `add` 必须**修改存储的值**，这和 `AdderInit.add_a` 只返回一个和不同。用前面例子的语法就行，不需要 import 或继承。

After writing the class, use these calls to check it:

> 写完后用下面的调用检查：

```python
counter1 = Counter(2)
counter2 = Counter(10)

counter1.add(3)
print(counter1.read(), counter2.read())

print(counter1.add(4))
print(counter1.read(), counter2.read())

counter2.add(-2)
print(counter1.read(), counter2.read())
```

The expected output is:

> 预期输出：

```text
5 10
None
9 10
9 8
```

Explain why `print(counter1.add(4))` prints `None` even though the next line shows that the stored value has changed. Why does changing one counter leave the other unchanged?

> 解释：为什么 `print(counter1.add(4))` 打印出 `None`，而下一行却显示存储值确实变了？为什么改一个计数器不会影响另一个？

```python
# Define Counter here.

# Then copy and run the checking code from Exercise 2.
# Explain the output using comments.
```

> [!NOTE]
> 按课程讲师要求，本文不收录练习解答。Exercise 2 的答案见[官方 notebook](https://github.com/semtm0051/2026_27/blob/322787dec91f174303ccf90399bb6243a060eb7d/Week_01/Lab/Week_01_python_classes_part1.ipynb) 中的 Show answer。

This completes Week 1's Python notebooks.

> 第 1 周的 Python notebook 到此结束。
