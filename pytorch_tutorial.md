- [先决条件](#先决条件)
- [NumPy和PyTorch的关系](#numpy和pytorch的关系)
- [什么是PyTorch](#什么是pytorch)
- [PyTorch与TensorFlow](#pytorch与tensorflow)
- [什么是张量](#什么是张量)
    - [张量数据类型](#张量数据类型)
    - [张量的形态](#张量的形态)
    - [张量构造函数](#张量构造函数)
    - [张量的操作](#张量的操作)
        - [张量的降维](#张量的降维)
        - [张量的升维](#张量的升维)
        - [张量的转置](#张量的转置)
        - [张量的逐元素算术运算](#张量的逐元素算术运算)
        - [张量的统计聚合操作](#张量的统计聚合操作)
        - [PyTorch v.s. NumPy](#pytorch-vs-numpy)
        - [张量的设备](#张量的设备)
- [How to Calculate Gradient?](#how-to-calculate-gradient)
- [深度神经网络训练过程概述](#深度神经网络训练过程概述)
    - [数据集与数据加载器](#数据集与数据加载器)
    - [全连接层](#全连接层)
    - [激活函数](#激活函数)
    - [损失函数](#损失函数)
    - [优化](#优化)
    - [模型训练](#模型训练)
        - [训练循环](#训练循环)
    - [验证](#验证)
    - [测试](#测试)
    - [build your own neural network](#build-your-own-neural-network)


## 先决条件

1. 熟悉 Python 语法。
2. 熟悉 NumPy 库。

## NumPy和PyTorch的关系

NumPy 是 CPU 科学计算的基础，而 PyTorch 是在 NumPy 理念基础上重构的、专为深度学习与 GPU 加速设计的框架。

核心区别与联系
1. 设计定位不同：
    - NumPy：通用科学计算库，核心数据结构是 ndarray，只能在 CPU 上运行。
    - PyTorch：深度学习框架，核心数据结构是 Tensor，原生支持 GPU/TPU 并行加速与自动求导（Autograd）。

2. 接口与语法极度相似：
    - PyTorch 在设计 Tensor API 时大幅参考了 NumPy 的习惯。数学运算、切片、广播机制（Broadcasting）以及维度变换的逻辑基本一致（例如 NumPy 的 axis 对应 PyTorch 的 dim）。

3. 无缝互转（零拷贝）：
    - 两者在 CPU 内存上共享底层数据块，转换几乎不消耗性能：

```python
import numpy as np
import torch

# NumPy -> PyTorch
np_arr = np.array([1, 2, 3])
tensor_data = torch.from_numpy(np_arr)

# PyTorch -> NumPy
back_to_np = tensor_data.numpy()
```

典型合作工作流，在实际 AI 项目开发中，两者通常协作互补：

1. 预处理：用 NumPy（或 Pandas）加载 CSV、图像或点云数据，完成数据清洗、标准化与特征拼接。

2. 训练：将 NumPy 数组转成 PyTorch Tensor 扔进 GPU/NPU 中，利用 PyTorch 的自动求导进行模型反向传播与权重更新。

3. 后处理：将模型的预测输出转回 NumPy，借助 Matplotlib 做可视化，或存入本地文件系统。

## 什么是PyTorch

1. 一个开源的机器学习框架。
2. 这是一个 Python 包，拥有两个高级功能：
    - 支持张量计算（类似 NumPy），且具备强大的 GPU 加速性能。
    - 能够构建基于自动微分系统的深度神经网络。

## PyTorch与TensorFlow

| 对比维度 | PyTorch | TensorFlow | 详细含义 |
| :--- | :--- | :--- | :--- |
| **Developer (开发者)** | Facebook AI (现 Meta AI) | Google Brain (谷歌大脑) | 两家顶尖科技巨头各自主导发起的开源框架。 |
| **Interface (编程接口/语言)** | Python & C++ | Python, C++, JavaScript, Swift | 都以 Python 为主流；TensorFlow 支持的跨平台语言更多。 |
| **Debug (调试难易度)** | Easier (更容易) | Difficult (较难，但在 2.0 版本后有所改善) | PyTorch 采用动态计算图，代码像普通 Python 一样直观，打断点单步调试极方便；TensorFlow 早期的静态图机制非常难调试，2.0 引入 Eager Execution 后才好转。 |
| **Application (主要应用场景)** | Research (学术研究) | Production (工业落地/生产环境) | PyTorch 在高校、科研界和学术论文中占绝对主导；TensorFlow 在大型企业服务部署、移动端/嵌入式落地（TensorFlow Lite）上生态更成熟。 |

## 什么是张量

![alt text](./img/pytorch/tensor1.png)

在计算机和数学里，“张量”只是多维数据的一个统称。你可以根据它的维数（Dimensions / Rank）来直观理解：

| 维数 (Rank) | 数学称呼 | 计算机里的概念 | 举个例子 |
| :--- | :--- | :--- | :--- |
| **0 维张量** | 标量 (Scalar) | 单个数字 | `5` |
| **1 维张量** | 矢量 / 向量 (Vector) | 一维数组 / 列表 | `[1.0, 2.0, 3.0]` （如：一排数据） |
| **2 维张量** | 矩阵 (Matrix) | 二维表格 / 矩阵 | `[[1, 2], [3, 4]]` （如：包含行和列的 Excel 表格） |
| **3 维张量** | 3D 张量 | 3 维数据块 | 一张彩色图片 `(高, 宽, RGB 3通道)` |
| **4 维张量** | 4D 张量 | 4 维数据阵列 | 一批彩色图片 `(批次大小, 高, 宽, 通道)`，深度学习中最常见的数据输入格式 |

### 张量数据类型

![alt text](./img/pytorch/tensor_data1.png)

在 PyTorch 中，用来处理“小数”和“整数”的两种最核心数据类型，以及它们对应的两种代码写法。

第一行：处理带小数点的数字（浮点数）

1. 物理本质 (Data type)：32-bit floating point（32 位单精度浮点数）。

2. 应用场景：神经网络里的权重参数（Weights）、梯度（Gradients）、连续的特征输入（如图像像素缩放值、特征向量）。

3. 写法一 (dtype)：torch.float
```python
# 使用通用函数，传入 dtype 参数（推荐）
x = torch.tensor([1.5, 2.3], dtype=torch.float)
```

4. 写法二 (tensor)：torch.FloatTensor
```python
# 直接使用浮点数专用类（老式/直接写法）
x = torch.FloatTensor([1.5, 2.3])
```

第二行：处理不带小数点的数字（整数）

1. 物理本质 (Data type)：64-bit integer (signed)（64 位带符号长整型）。

2. 应用场景：分类任务的分类标签（Labels，如 0 代表猫，1 代表狗）、词表/嵌入层的索引（Indices）。

3. 写法一 (dtype)：torch.long
```python
# 使用通用函数，传入 dtype 参数（推荐）
y = torch.tensor([0, 1, 2], dtype=torch.long)
```

4. 写法二 (tensor)：torch.LongTensor
```python
# 直接使用长整型专用类（老式/直接写法）
y = torch.LongTensor([0, 1, 2])
```

### 张量的形态

![alt text](./img/pytorch/tensor_shape1.png)

PyTorch 中张量形状（Shape）的几何表示与维度索引（dim）的对应关系。

三个示例拆解
1. 1D 张量（向量）：shape = (5, )
    - 一排共 5 个元素。
    - 只有 1 个维度：dim 0 的大小为 5。
2. 2D 张量（矩阵）：shape = (3, 5)
    - 一个 3 行 5 列的网格（共 15 个元素）。
    - 最高维度 dim 0 代表行数（3 行），下一个维度 dim 1 代表列数（5 列）。
3. 3D 张量（数据块）：shape = (4, 5, 3)
    - 可以直观理解为由 4 个 5 x 3 的矩阵堆叠而成的立体块。
    - 从外层向内层依次为：dim 0 = 4（切片数/深度）、dim 1 = 5（行）、dim 2 = 3（列）。

底部 Note 关键说明，dim in PyTorch == axis in NumPy

1. PyTorch 里的维度叫 dim（例如做求和运算 torch.sum(x, dim=0)）。

2. NumPy 里的维度叫 axis（例如 np.sum(x, axis=0)）。

3. 两者的概念完全相同，都是从外层中括号（最高维 0）向内层递增计数。


### 张量构造函数

![alt text](./img/pytorch/tensor_constructor1.png)

PyTorch 中最常见的几种张量构造方法（Tensor Constructor），也就是如何在代码里创建一个 Tensor。

三种常用构造方式
1. From list / NumPy array（从现有数据创建）
    - 用 Python 嵌套列表初始化：`x = torch.tensor([[1, -1], [-1, 1]])`
    - 从 NumPy 数组转换：`x = torch.from_numpy(np.array([[1, -1], [-1, 1]]))`
    - 作用：把已有的本地数据（如 Python 列表或 NumPy 矩阵）直接包装成 PyTorch Tensor。

2. Zero tensor（全零张量）
    - 代码：`x = torch.zeros([2, 2])`
    - 作用：创建一个元素全为 0. 的张量，传入的参数 [2, 2] 指定其 Shape（形状），即 2 行 2 列。常用于初始化累加变量或掩码（Mask）。

3. Unit tensor / Ones tensor（全一张量）
    - 代码：`x = torch.ones([1, 2, 5])`
    - 作用：创建一个元素全为 1. 的张量，传入的参数 [1, 2, 5] 代表 Shape 为 3 维（深度 1，行数 2，列数 5）。常用于权重初始化或偏置向量搭建。


### 张量的操作

#### 张量的降维

![alt text](./img/pytorch/tensor_operation_squeeze1.png)

把长度为 1 的冗余维度压掉，给 Tensor 降维。

代码逻辑与几何拆解
1. 原始状态：
    - 代码：`x = torch.zeros([1, 2, 3])`
    - 形状：`torch.Size([1, 2, 3])`，这是一个 3D 张量。
    - 几何意义：如右上图，表示有 1 个 2 x 3 的数据块（最外层 dim=0 的长度只有 1）。

2. 执行挤压：
    - 代码：`x = x.squeeze(0)`
    - 参数：0 代表指定的维度 dim = 0。
    - 作用：告诉系统“检查 dim=0，如果它的长度是 1，就把这个维度抹掉”。

3. 挤压后状态：
    - 形状：torch.Size([2, 3])，变成了 2D 张量（矩阵）。
    - 几何意义：如右下图，原先那个厚度为 1 的 3D 块，被压缩成了一个标准的 2 x 3 平面网格。

#### 张量的升维

![alt text](./img/pytorch/tensor_operation_unsqueeze1.png)

这是 squeeze 的反向操作——unsqueeze（扩维/升维），用来在指定位置硬塞入一个长度为 1 的新维度。

代码与几何逻辑拆解
1. 原始状态（2D 平面）：
    - 代码：`x = torch.zeros([2, 3])`
    - 形状：`torch.Size([2, 3])`
    - 维度索引：`dim=0` 是 2（行），`dim=1` 是 3（列）。

2. 执行升维（unsqueeze(1)）：
    - 参数 1 代表：在索引为 1 的位置插入一个长度为 1 的新维度。
    - 形状变化：`[2, 3]` $\rightarrow$ `[2, 1, 3]`
        - 原来的 dim=0（2）依然在 dim=0。
        - 新插入的 dim=1 大小为 1。
        - 现在的 dim=2 变成了原来的 3。

3. 几何图形对应：
    - 如右下图所示，原来的 2 × 3 扁平网格，在中间强行拉扯出一个厚度为 1 的新轴（红圈标出），使它从 2D 矩阵升维成了 3D 矩阵。

#### 张量的转置

![alt text](./img/pytorch/tensor_operation_transpose1.png)

transpose（转置 / 交换维度） 操作：指定交换张量的某两个维度。

代码与几何逻辑拆解
1. 原始状态（2 行 3 列）：
    - 代码：`x = torch.zeros([2, 3])`
    - 形状：`torch.Size([2, 3])`
    - 维度划分：`dim 0 = 2`（行数），`dim 1 = 3`（列数）。
2. 执行转置（`transpose(0, 1)`）：
    - 参数 (0, 1) 表示：把 dim 0 和 dim 1 这两个轴对调位置。
    - 形状变化：`[2, 3]` $\rightarrow$ `[3, 2]`
        - 原来的行（大小 2）变成了现在的列。
        - 原来的列（大小 3）变成了现在的行。
3. 几何图形对应：
    - 如图所示，原本横着放的 2 × 3 扁平网格，经转置后旋转顺应变成了竖着放的 3 × 2 矩阵。

#### 张量的拼接

![alt text](./img/pytorch/tensor_operation_concat1.png)

torch.cat（Concatenate，拼接） 操作：沿指定的维度（dim）将多个张量拼接成一个更大的张量。

代码与几何逻辑拆解
1. 参与拼接的三个张量：
    - 粉色 x：`shape = [2, 1, 3]`（dim 1 的长度为 1）
    - 黄色 y：`shape = [2, 3, 3]`（dim 1 的长度为 3）
    - 绿色 z：`shape = [2, 2, 3]`（dim 1 的长度为 2）
2. 拼接操作（dim=1）：
    - 代码：`w = torch.cat([x, y, z], dim=1)`
    - 规则：除了指定的 `dim=1` 之外，其他维度的大小必须完全一致（这里 dim 0 都是 2，dim 2 都是 3）。
    - 维度计算：dim 1 的新长度直接累加：$1 + 3 + 2 = 6$。
3. 拼接后的结果 w：
    - 形状：`torch.Size([2, 6, 3])`
    - 几何图形：如最下方的图所示，粉色、黄色、绿色三个块在 dim 1（横向/宽度方向）挨个并排贴合，拼成了一个宽度为 6 的完整大长方体。

#### 张量的逐元素算术运算

![alt text](./img/pytorch/tensor_operation_operator1.png)

PyTorch 张量的基础逐元素算术运算（Element-wise Operations）。

1. Addition（加法）：`z = x + y`
    - 作用：将 x 和 y 中对应位置的元素逐个相加（也可写成 torch.add(x, y)）。

2. Subtraction（减法）：`z = x - y`
    - 作用：将 x 中每个元素减去 y 中对应位置的元素（也可写成 torch.sub(x, y)）。

3. Power（幂运算）：`y = x.pow(2)`
    - 作用：计算 x 中每个元素的指定次方，这里 2 代表平方（也可写成 x ** 2）。

#### 张量的统计聚合操作

![alt text](./img/pytorch/tensor_operation_operator2.png)

统计聚合操作（Aggregation Operations）。

1. Summation（求和）：`y = x.sum()`
    - 作用：计算张量 x 中所有元素相加的总和。不传参数时，会将所有维度压缩，返回一个标量（0 维 Tensor）。

2. Mean（求平均值）：`y = x.mean()`
    - 作用：计算张量 x 中所有元素的算术平均值（即 `x.sum() / x.numel()`）。同样返回一个标量。

#### PyTorch v.s NumPy

![alt text](./img/pytorch/pytorch_vs_numpy1.png)

PyTorch 在设计 Tensor 的 API 时故意模仿了 NumPy 的习惯，让两者在查看数据基本属性时的语法保持完全一致。

| 属性访问 | 语法（ `PyTorch` / `NumPy` ） | 含义与作用 |
| :--- | :--- | :--- |
| **形状 / 维度** | `x.shape` | 返回张量或数组的维度（尺寸），如 `(3, 224, 224)`。（注：*PyTorch* 也可以用方法 `x.size()`） |
| **数据类型** | `x.dtype` | 返回元素的数据类型，如 `float32`，`int64` 等。 |

![alt text](./img/pytorch/pytorch_vs_numpy2.png)

| 操作类型 / 含义 | PyTorch | NumPy |
| :--- | :--- | :--- |
| **重塑形状 (Reshape)** | `x.reshape(...)` / `x.view(...)` | `x.reshape(...)` |
| **压缩维度 (Squeeze)**<br>*(移除长度为 1 的维度)* | `x.squeeze()` | `x.squeeze()` |
| **扩充维度 (Unsqueeze)**<br>*(在指定轴插入长度为 1 的维度)* | `x.unsqueeze(1)` | `np.expand_dims(x, 1)` |

#### 张量的设备

![alt text](./img/pytorch/tensor_device1.png)

PyTorch 中张量（Tensor）与模型模块（Module）的设备分配与迁移机制（Device Management）。

1. 默认机制（Default）：
    - 在 PyTorch 中，默认创建的所有 Tensor 和神经网络模型（nn.Module）都会保存在 CPU 内存中并在 CPU 上执行计算。
2. 设备迁移 API (.to())：
    - 迁移到 CPU：x = x.to('cpu')，将数据转移至主机内存（系统 RAM）。
    - 迁移到 GPU (CUDA)：x = x.to('cuda')，将数据从 CPU 转移到显卡显存（VRAM），利用 GPU 并行加速计算。

![alt text](./img/pytorch/tensor_device2.png)

关键细节与工程实践

1. torch.cuda.is_available()：
    - 返回布尔值 True 或 False。
    - 它不仅检查你的电脑上有没有 NVIDIA 显卡，还会验证是否正确安装了匹配的 CUDA Driver 和 PyTorch CUDA 版本。

2. 多卡指定 ('cuda:X')：
    - 这里的 cuda 指的是 NVIDIA CUDA 软件驱动框架及设备识别前缀，而不是指单个物理计算单元。
    - 编号从 0 开始，cuda:0 代表第一张默认 GPU。
    - 也可以通过环境变量在终端临时限制运行卡号：CUDA_VISIBLE_DEVICES=0,1 python train.py。

3. 为什么 GPU 适合深度学习 / 矩阵运算？：
    - CPU：适合做高复杂的顺序逻辑控制（少量强大的 Core）。
    - GPU：拥有数千个小计算单元（CUDA Cores），非常适合对大规模矩阵/张量同时执行相同的加乘指令（SIMD/SIMT），从而将矩阵乘法加速几十甚至上百倍。


## How to Calculate Gradient?

![alt text](./img/pytorch/tensor_howtocalcgradient1.png)

PyTorch 中使用 Autograd（自动求导引擎）计算梯度的完整过程

1. 步骤 1：创建需要求导的张量
    - 代码：`x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)`
    - 数学：定义矩阵 $x = \begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix}$。
    - 关键点：requires_grad=True 告诉 PyTorch 开始追踪对 $x$ 的所有计算操作，以便后续计算梯度。
2. 步骤 2：前向传播（计算目标标量）
    - 代码：`z = x.pow(2).sum()`
    - 数学：$z = \sum_{i}\sum_{j} x_{i,j}^2 = 1^2 + 0^2 + (-1)^2 + 1^2 = 3$。
    - 含义：先对 $x$ 的每个元素平方，再求和得到标量 $z$（通常充当损失函数 Loss）。
3. 步骤 3：反向传播（自动求导）
    - 代码：`z.backward()`
    - 数学：根据微积分求导法则，对偏导数公式求解：$\frac{\partial z}{\partial x_{i,j}} = \frac{\partial (x_{i,j}^2)}{\partial x_{i,j}} = 2 x_{i,j}$。
    - 含义：触发自动求导引擎，利用链式法则计算 $z$ 关于 $x$ 内部每个元素的偏导数。
4. 步骤 4：提取计算得到的梯度
    - 代码：`x.grad`
    - 数学：将 $x$ 的具体数值代入导数公式 $2x$ 中：$$\frac{\partial z}{\partial x} = 2 \cdot \begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ -2 & 2 \end{bmatrix}$$
    - 结果：输出打印出的梯度张量 tensor([[2., 0.], [-2., 2.]])。

## 深度神经网络训练过程概述

![alt text](./img/pytorch/OverviewoftheDNNTrainingProcedure1.png)

把整个流程拆解为三大核心部分：

1. 数据准备（Load Data），对应右顶部的蓝色虚线框，负责把磁盘上的数据加载并打包成模型能吃的 Batch 格式：
    - torch.utils.data.Dataset：负责定义数据的读取逻辑（如读取图片、标注文件、转换类型）。
    - torch.utils.data.DataLoader：负责对数据做 Batch 分组、打乱（Shuffle）以及多线程并行加载。
    - 数据去向：虚线分别指向了训练（Training）、验证（Validation）和测试（Testing）阶段。

2. 模型与优化器构建（网络三大件），对应左侧的蓝色虚线框，包含了模型训练的核心配置，主要依赖两个库：
    - Define Neural Network（定义网络结构）：使用 torch.nn.Module 构建
    - Loss Function（损失函数）：使用 torch.nn 里的损失函数（如 MSELoss, CrossEntropyLoss）评估模型预测结果与真实标签的误差。
    - Optimizer（优化器）：使用 torch.optim 模块（如 SGD, Adam）根据计算出的梯度来更新网络中的权重参数。

3. 训练与评估循环（Training & Evaluation Loop），对应中间和右侧的三个核心运行模块：
    - Training（训练）：将数据喂给网络，算 Loss，做反向传播（loss.backward()），用 Optimizer 更新权重。
    - Validation（验证）：在训练过程中每隔几轮验证一次，用来调参或判断模型是否过拟合（通过指标反馈回 Training 决定是否继续训练或调整策略）。
    - Testing（测试）：模型彻底训练好后，用从未见过的独立测试集评估最终的泛化性能。

### 数据集与数据加载器

![alt text](./img/pytorch/dnn_procedure_dataset_dataloader1.png)

如何把硬盘里的文件（如图片、文本、3D模型数据等）规范地喂给神经网络。

1. __init__(self, file)：初始化与加载数据
    - 作用：实例化这个类时最先执行。
    - 通常在这里做：读取路径、加载文件列表、做初始的数据预处理（比如读取图片路径、加载 CSV 等）。
2. __getitem__(self, index)：根据索引取单条数据
    - 作用：支持像数组一样用 dataset[i] 去取第 index 条数据。
    - 通常在这里做：把第 index 项数据从磁盘/内存中拉出来，转换成 torch.Tensor 类型返回。
3. __len__(self)：获取数据集总长度
    - 作用：支持用 Python 的 len(dataset) 查看里面一共有多少条样本。

![alt text](./img/pytorch/dnn_procedure_dataset_dataloader2.png)

DataLoader 如何自动化地把 Dataset 里的单条数据批量打包（Batching）并喂给模型 的工作原理。

核心参数与机制解析

1. dataset：传入上一张图写好的数据容器。

2. batch_size：指定每个批次包含多少条数据（例如图中一次抽取 5 条，合成一个 mini-batch）。

3. shuffle=True / False：控制是否打乱抽取顺序。
    - 训练阶段（Training: True）：必须设为 True。每轮训练前随机打乱索引，防止模型记住数据的先后顺序，避免过拟合。
    - 测试阶段（Testing: False）：通常设为 False。评估模型时按固定顺序读取即可，方便复现结果。

回顾：

![alt text](./img/pytorch/dnn_procedure_dataset_dataloader3.png)

简单来说，如果你的样本数据（$N$）非常多，直接把所有数据一次性丢给模型训练是非常慢且占内存的。所以，我们把数据切分成很多小块，一块一块地喂给模型。

1. 以下是核心概念的拆解：
    - 什么是 Batch (批次)？Batch：就是把庞大的训练数据集 $N$ 分成一个个小份，每一份叫一个 Batch，大小为 $B$。
    - 为什么需要 Batch？
        - 内存限制：如果一次性把 100 万条数据全部读进内存，显卡（GPU）可能会直接崩溃。
        - 训练速度：分批次训练可以让模型在还没看完所有数据时，就已经开始通过梯度下降更新参数了，这样学习效率更高。

2. 训练流程：Update 与 Epoch
    - Update (参数更新)：
        - 每次模型读取一个 Batch 的数据。
        - 计算这一个 Batch 的损失 $L^i$。
        - 计算梯度 $g$。
        - 更新参数 $\theta$。
        - 重点：只要算完一个 Batch 并更新一次参数，就叫一次 Update。如果一个 Epoch 里有 100 个 Batch，那么一次 Epoch 就对应 100 次 Update。
    - Epoch：
        - 指的是“把所有的数据（即 $N$ 个数据）完整地看了一遍”。
        - 如果你的数据集被分成了 4 个 Batch，那么当你把这 4 个 Batch 全部训练过一次之后，就完成了 1 个 Epoch。

3. 为什么 Batch 大小也是“超参数(人为设置)”？
    1. Batch 太小：参数更新太频繁，虽然学得快，但受单个噪声数据影响大，训练不稳定。
    2. Batch 太大：更新次数少，利用不到大数据量带来的高效计算优势，且可能卡在局部最优解。
    3. 找平衡：我们需要人为去设置一个合适的 Batch 大小（比如 32, 64, 128），这就是所谓的“超参数”。
    
什么时候停止呢？
- 自己设置上限（超参数），这是最简单粗暴的方法。人为设定一个“训练轮数（Epochs）”或者“迭代次数”。
    - 例子：设定迭代 1000 次，无论损失是否降低，到了 1000 次就强行停止。
    - 缺点：这是一个“超参数”，如果设置得太小，模型还没收敛；设置太大，则浪费算力。
- 理想情况：微分值为 0（极小值点），从数学角度看，当损失函数的导数（梯度）为 0 时，说明到达了函数的波谷（极小值点）。此时参数不再更新，算法自然停止。
- 在高维的复杂神经网络中，相比于局部最小值（Local Minima），鞍点（Saddle Points）和平坦区域（Plateaus，梯度极其微小，参数几乎不再发生改变，看起来就像训练“停滞”了一样）确实是更常见且更棘手的阻碍。

### 全连接层

![alt text](./img/pytorch/dnn_procedure_fullconnected1.png)

线性层（nn.Linear，也叫全连接层 Fully-connected Layer）。它的核心机制是执行线性变换方程 $y = xW^T + b$。

核心要素拆解

1. nn.Linear(32, 64)
    - in_features = 32：要求输入的特征维度（最后一个维度）必须是 32。
    - out_features = 64：经过这一层的矩阵计算后，把特征维度映射转换成 64。

2. 前面的星号 *（批次及其他维度）
    - 线性层只关心最后一个维度是否与 in_features 匹配。
    - 前面的维度（星号 *）可以是任意形状（如 Batch Size、序列长度等），nn.Linear 会自动保持前面所有维度不变。

3. 图中的尺寸对应示例无论输入的张量维度多复杂，只有最后一维会从 $32$ 变成 $64$：
    - 输入 (10, 32) $\rightarrow$ 输出 (10, 64) (2D 矩阵：10 个样本，每个样本 32 维特征)
    - 输入 (10, 5, 32) $\rightarrow$ 输出 (10, 5, 64) (3D 张量：常见于序列或 3D 空间点数据)
    - 输入 (1, 1, 3, 32) $\rightarrow$ 输出 (1, 1, 3, 64) (4D 张量)

![alt text](./img/pytorch/dnn_procedure_fullconnected2.png)

![alt text](./img/pytorch/dnn_procedure_fullconnected3.png)

1. 左图：神经网络拓扑视角（人类直观图）
    - Batch Size = 1
    - 输入层（32 个绿框）：代表 1 个样本包含 32 个数值标量 $x_1, x_2, \dots, x_{32}$。
    - 隐藏/输出节点（64 个蓝圆圈）：代表 64 个独立的计算单元（神经元）。
    - 连接线（全连接 Fully-Connected）：
        - 每个蓝圈都连接了全部 32 个绿框。
        - 每条连线上都有一个可以学习的权重参数 $w$。
        - 第 1 个蓝圈计算出 $y_1$，第 2 个算出 $y_2$，直到第 64 个算出 $y_{64}$。
    - 标量计算公式（以输出的第 1 个数 $y_1$ 为例）：$$y_1 = (w_{1,1}x_1 + w_{1,2}x_2 + \dots + w_{1,32}x_{32}) + b_1$$

2. 右图：矩阵计算视角（GPU 实际运行的公式）如果用上面的标量公式写 64 次，代码效率极低。右图展示了如何用线性代数写成一次矩阵加乘：

$$\boldsymbol{y} = \boldsymbol{W} \cdot \boldsymbol{x} + \boldsymbol{b}$$

| 符号 | 图中形状 | 几何/代数含义 | 详细说明 |
| :--- | :--- | :--- | :--- |
| **$\boldsymbol{W}$** | **$64 \times 32$** | 权重矩阵 | 包含 $64 \times 32 = 2048$ 个参数。**矩阵的第 $i$ 行（共 32 个数），就是专门用来计算 $y_i$ 的那 32 条连线的权重**。 |
| **$\boldsymbol{x}$** | **$32 \times 1$** | 输入列向量 | 包含 1 个样本的 32 个特征数字。 |
| **$\boldsymbol{b}$** | **$64 \times 1$** | 偏置列向量 | 给 64 个神经元各自加上一个偏置量（偏移截距）。 |
| **$\boldsymbol{y}$** | **$64 \times 1$** | 输出列向量 | 矩阵乘法结果：$(64 \times 32) \times (32 \times 1) + (64 \times 1) = (64 \times 1)$，得到 64 个新特征。 |

![alt text](./img/pytorch/dnn_procedure_fullconnected4.png)

代码与图的精准对照

1. `layer = torch.nn.Linear(32, 64)`，初始化一个全连接层，输入特征 32 维，输出特征 64 维。
2. `layer.weight.shape $\rightarrow$ torch.Size([64, 32])`，对应右图中的 $W (64 \times 32)$。证实了之前讨论的细节：PyTorch 内部保存的权重矩阵形状就是 (out_features, in_features)，即 64 行 32 列。
3. `layer.bias.shape $\rightarrow$ torch.Size([64])`对应右图中的 $b$。输出维度是 64，所以偏置向量长度为 64，给每个输出节点各配一个偏置量。

### 激活函数

![alt text](./img/pytorch/dnn_procedure_activationfunctions1.png)

这张图展示的是神经网络中的激活函数（Activation Functions），它们的作用是引入非线性（Non-linearity）。

1. Sigmoid 激活函数 (nn.Sigmoid())数学公式：
    - $\sigma(x) = \frac{1}{1 + e^{-x}}$
    - 曲线特征：如右图上方所示，是一条平滑的 S 型曲线。
    - 输出范围：把任意实数压缩到 $(0, 1)$ 之间。
    - 主要用途：
        - 二分类问题（Binary Classification）：用于网络最后一层，将输出转化为概率（比如判断是不是猫，输出 $0.85$ 表示有 $85\%$ 的概率是猫）。
    - 缺点：当输入太小或太大时，曲线非常平缓，梯度接近 0，容易导致深层网络出现梯度消失（Gradient Vanishing）。

2. ReLU 激活函数 (nn.ReLU())
    - 数学公式：$\text{ReLU}(x) = \max(0, x)$
    - 曲线特征：如右图下方所示，小于 0 的部分全压成 0，大于 0 的部分保持不变（$y = x$）。
    - 输出范围：$[0, +\infty)$
    - 主要用途：
        - 中间隐藏层（Hidden Layers）：现代深度学习中间层最推荐、最通用的激活函数。
    - 优点：计算极其简单高效（CPU/GPU 只需要判断是否大于 0）；在正半轴导数恒等于 1，能很好地缓解梯度消失问题，加速网络收敛。

### 损失函数

![alt text](./img/pytorch/dnn_procedure_lossfunctions1.png)

这张图展示的是 PyTorch 中最基础的两类损失函数（Loss Functions）。损失函数用来衡量模型预测值（$\hat{y}$）与真实标签（$y$）之间的差距，也是反向传播算梯度的源头。

1. 均方误差损失：nn.MSELoss()
    - 适用场景：回归问题（Regression），比如预测房价、预测坐标偏移量、估算物理载荷等连续数值。
    - 数学公式：$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$
    - 核心逻辑：计算预测值与真实值差值的平方和再求平均。取平方一方面消除了负号，另一方面会放大较大误差的惩罚权重。
2. 交叉熵损失：nn.CrossEntropyLoss()
    - 适用场景：分类问题（Classification），比如判断图片属于猫/狗/兔中的哪一类。
    - 数学公式（多分类）：$$y'_i = \frac{\exp(y_i)}{\sum_j \exp(y_j)}$$
        - $y_i$：模型原始输出（logit），比如 3, 1, -3
        - $y'_i$：经过 softmax 后的“概率”，比如 0.88, 0.12, 0 $\approx 0$
        - 分母是所有 exp 的和 → 保证总和为 1
    - 核心逻辑：PyTorch 内置把 Softmax（把网络输出值转为概率）和 负对数似然（NLL Loss） 结合在了一起。只要模型对真实类别的预测概率越接近 1，Loss 就越接近 0；反之如果预测错了，Loss 会急剧飙升。

3. 回归 vs 分类 经典组合总结

| 任务类型 | 模型最后一层输出维度 | 激活函数 | 常用损失函数 |
| :--- | :--- | :--- | :--- |
| **回归任务** | `1`（或预测变量的个数） | 通常无需激活函数 | **`nn.MSELoss()`** |
| **分类任务** | `C`（类别总数） | 无（PyTorch 交叉熵自带 Softmax） | **`nn.CrossEntropyLoss()`** |

### 优化

![alt text](./img/pytorch/dnn_procedure_optimizers1.png)

PyTorch 中的优化器（Optimizer），负责根据计算出的梯度来更新网络中的权重和偏置参数。

1. 核心作用：执行梯度下降公式
    - 我们在前面讨论过梯度下降更新公式：$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot g$$
    - 优化器的工作就是自动化执行这个公式，把模型里的权重 $\theta$ 沿着梯度的反方向推进一小步。

2. 代码参数拆解 (`torch.optim.SGD`)
    - params（关键参数）：
        - 指图中的 `model.parameters()`。它把模型中所有需要训练的参数（即所有 nn.Linear 层的 $W$ 和 $b$ 矩阵）打包交给优化器，告诉优化器“你要更新哪些权重”。
        - lr（Learning Rate，学习率 $\eta$）：
            - 控制每一步迈多大（比如 lr=0.01）。
        - momentum（动量）：物理学冲量的概念，帮助优化器冲过局部极小值点或平坦区域。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 在每个训练 Step 中：
optimizer.zero_grad()  # 1. 清空上一次累积的梯度
loss.backward()        # 2. 反向传播，计算当前梯度 g
optimizer.step()       # 3. 触发优化器，按 W_new = W_old - lr * g 更新所有 W 和 b
```

### 模型训练

代码逐行拆解

1. `dataset = MyDataset(file)`
    - 数据集读取：实例化自定义的 Dataset 类，负责从本地文件（CSV、图像、3D mesh 等）中读取原始数据并转成 Tensor。
2. `tr_set = DataLoader(dataset, 16, shuffle=True)`
    - 数据加载器：将 dataset 打包成可迭代的批处理器。
    - 16：Batch Size，表示每次训练喂入 16 条数据。
    - shuffle=True：每轮训练打乱数据顺序，防止模型记住样本的排列位置。
3. `model = MyModel().to(device)`
    - 模型实例化与设备迁移：创建网络实例，并用 .to(device) 把权重矩阵推送到 GPU（cuda）或 CPU 上进行加速计算。
4. `criterion = nn.MSELoss()`
    - 设置损失函数：这里选择了均方误差损失，说明当前训练任务是一个回归任务（Regression）。
5. optimizer = torch.optim.SGD(model.parameters(), 0.1)
    - 设置优化器：使用随机梯度下降算法，绑定模型的权重参数，设置学习率 $\eta = 0.1$。
    
一句话总结

这张图把训练所需要的所有“基础设施”全建好了：数据准备完毕（DataLoader） $\rightarrow$ 模型准备完毕（Model） $\rightarrow$ 评价标准准备完毕（MSELoss） $\rightarrow$ 求解器准备完毕（SGD）。下一步就是直接写 for epoch in range(...) 循环进入正式训练了。

#### 训练循环

![alt text](./img/pytorch/dnn_procedure_trainloop1.png)

神经网络模型训练的核心循环（Training Loop），也就是把前面准备好的数据、模型、损失函数和优化器真正run起来的过程。

代码逐行拆解与逻辑
1. `for epoch in range(n_epochs)`:
    - 外层循环控制全局迭代轮次。1 个 Epoch 意味着把完整数据集训练一遍。
2. `model.train()`
    - 将模型切换为训练模式。主要用来开启 Dropout 和 Batch Normalization 等特定层在训练时的行为。
3. `for x, y in tr_set`:
    - 内层循环按 Batch（比如上一张图设置的 16）从 DataLoader 中分批次提取数据 $x$ 和对应的标签 $y$。
    - 在每一次内层循环中，提取出来的 $x$ 和 $y$ 不再是单条数据，而是打包好的一批数据（Batch）：
        - $x$（特征矩阵）：形状为 (16, 32) $\rightarrow$ 这一包里有 16 个样本，每个样本有 32 个输入特征。
        - $y$（标签矩阵）：形状为 (16, 1) $\rightarrow$ 这一包里有 16 个真实答案，对应着这 16 个样本的目标值。 
        - “按 Batch 分批次提取”本质上就是：利用 GPU 的并行计算能力，每次抓 16 个样本打包成矩阵一次性算完，既不撑爆显存，又能快速更新网络参数。
4. `optimizer.zero_grad()`
    - 清空旧梯度。PyTorch 的梯度默认会在 .grad 属性中累加，为了防止上一个 Batch 的梯度影响本次计算，必须在计算新梯度前手动清零。
5. `x, y = x.to(device), y.to(device)`
    - 将当期 Batch 的特征和标签推送到指定设备（如 GPU cuda）上，确保与模型的参数处在同一个计算设备上。
6. `pred = model(x)`
    - 前向传播（Forward Pass）。把数据 $x$ 喂入网络，计算出当前 Batch 的预测值 pred。
7. `loss = criterion(pred, y)`
    - 计算损失。通过损失函数（如 MSELoss）对比预测值 pred 与真实标签 $y$，得到当前 Batch 的标量损失值。
8. `loss.backward()`
    - 反向传播（Backpropagation）。基于链式法则自动计算损失关于各个权重矩阵 $W$ 和偏置 $b$ 的梯度 $g = \frac{\partial \text{Loss}}{\partial W}$，并填充到参数的 .grad 字段中。
9. `optimizer.step()`
    - 更新参数。优化器（如 SGD）触发梯度下降更新法则：$W_{new} = W_{old} - \eta \cdot g$，完成一次权重的修整。

#### 验证

![alt text](./img/pytorch/dnn_procedure_validate1.png)

神经网络的评估/验证过程（Validation/Evaluation Loop）。

与前面训练循环（Training Loop）最大的不同是：验证阶段只测成绩、不更新参数，因此关闭了梯度计算。

代码逐行拆解与逻辑

1. `model.eval()`
    - 切换模型为评估模式。主要用来关闭 Dropout（让所有神经元都参与预测）和冻结 Batch Normalization 的均值/方差更新。
2. `total_loss = 0`
    - 初始化累加器：用来累加整个验证集上所有样本的总损失值。
3. `for x, y in dv_set:`
    - 遍历验证集：从验证集的数据加载器（dv_set）中按 Batch（如每包 16 个）分批提取数据 x 和标签 y。
4. `x, y = x.to(device), y.to(device)`
    - 数据迁移：把当前 Batch 的数据推送到 GPU/CPU 上。
5. `with torch.no_grad():`（最核心）
    - 禁用梯度计算：告诉 PyTorch 接下来不需要构建计算图，也不用保存反向传播用的中间激活值。这可以大幅节省显存，并提高推理速度。
6. `pred = model(x)`
    - 前向传播：仅做网络的前向计算，求出预测值 pred。
7. `loss = criterion(pred, y)`
    - 计算当前 Batch 损失：计算当前批次的平均损失（注意这里的 loss 是当前 Batch 中单个样本的平均损失。
8. `total_loss += loss.cpu().item() * len(x)`
    - 累加样本总 Loss：
        - loss.cpu().item() 把 Tensor 标量转为 Python 纯数字。
        - len(x) 是当前 Batch 的实际样本数量（如 16）。
        - 两者相乘还原出“该 Batch 内所有样本的总损失”，并加到 total_loss 里。
9. `avg_loss = total_loss / len(dv_set.dataset)`
    - 计算全集平均 Loss：用累加的绝对总损失，除以验证集的数据总条数 len(dv_set.dataset)，得到整个验证集上的最终平均损失。

#### 测试

![alt text](./img/pytorch/dnn_procedure_test1.png)

最终推理/测试阶段（Testing Set / Inference Loop）。

与上一张验证集（Validation Set）代码相比，最关键的区别在于：测试集没有标签 $y$，不计算损失 Loss，只负责生成并收集预测结果 pred 用于最终提交或导出。

代码逐行拆解

1. `model.eval()`
    - 评估模式：冻结 Dropout 和 Batch Normalization，确保推理过程的确定性。
2. `preds = []`
    - 容器初始化：创建一个列表，用来顺序收集所有 Batch 的预测输出。
3. `for x in tt_set`:
    - 遍历测试集：注意这里是 for x 而不是 for x, y，因为测试集通常没有真实标签 $y$（比如 Kaggle 比赛中的 test.csv）。
4.  `x = x.to(device)`
    - 数据迁移：将特征数据 x 送入 GPU/CPU 计算设备。
5. `with torch.no_grad()`:
    - 关闭梯度计算：仅做纯前向推断，大幅省显存并加速计算。
6. `pred = model(x)`
    - 前向预测：计算当前 Batch 的输出（如 x.shape 是 (16, 32)，得到的 pred.shape 就是 (16, 64) 或回归任务的 (16, 1)）。
7. `preds.append(pred.cpu())`
    - 收集预测值：先用 .cpu() 将预测张量从 GPU 挪回 CPU 内存（避免 GPU 显存爆掉），再存入 preds 列表中。

### Build your own neural network

![alt text](./img/pytorch/dnn_procedure_buildselfDNN1.png)

![alt text](./img/pytorch/dnn_procedure_buildselfDNN2.png)

在 PyTorch 中如何动手搭建一个完整的神经网络模型。

核心结构拆解

1. 继承基类`nn.Module`
    - 自定义模型必须继承 nn.Module，这是所有神经网络模块的母类。
2. 初始化函数`__init__(self)`
    - 作用：定义和实例化网络需要的组件（零件库）。
    - `super().__init__()`：调用父类初始化，注册 PyTorch 内部的参数追踪机制。
    - `nn.Sequential(...)`：一个容器，把内部的层按顺序串联起来形成数据流水线：
        - `nn.Linear(10, 32)`：输入 10 维，升维提取出 32 个特征。
        - `nn.Sigmoid()`：非线性激活。
        - `nn.Linear(32, 1)`：降维映射到 1 维输出。
3. 前向传播函数 forward(self, x)
    - 作用：定义数据 $x$ 经过网络的真实计算流程（组装流水线）。
    - `self.net(x)`：将输入数据 $x$ 传入刚才在 __init__ 中组装好的 Sequential 流水线并返回计算结果。

数据流变化图解

```text
输入数据 x (Batch, 10)
    │
    ▼  nn.Linear(10, 32)   --> 特征升维 (Batch, 32)
    │
    ▼  nn.Sigmoid()        --> 非线性激活 (Batch, 32)
    │
    ▼  nn.Linear(32, 1)    --> 输出降维 (Batch, 1)
输出结果 (Batch, 1)
```

