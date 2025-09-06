# D2MVSC 环境配置指南

本文档记录了复现 [D2MVSC](https://github.com/jiaowangswust/D2MVSC) 项目时的环境配置步骤、常见问题及解决方案，帮助快速搭建运行环境。

## 项目背景

D2MVSC 是一个基于深度学习的多视图子空间聚类算法，原项目依赖特定版本的 PyTorch、CUDA 及其他库，环境配置过程中可能遇到版本兼容问题，本文档旨在解决这些问题。

## 环境要求

- 操作系统：Windows 10/11
- Python：3.7.x
- PyTorch：1.7.0+cu110
- torchvision：0.8.1+cu101
- CUDA：11.0
- 其他依赖：numpy、scipy、scikit-learn（0.22.2）、munkres 等

## 详细配置步骤

### 步骤1：创建虚拟环境

使用 Anaconda/Miniconda 创建独立环境，避免依赖冲突：

```bash
# 创建名为 d2mvsc 的环境，指定 Python 3.7
conda create -n d2mvsc python=3.7 -y

# 激活环境
conda activate d2mvsc
```

### 步骤2：安装torch、torchvision和cuda

使用以下命令安装torchvision的时候，会一起安装版本适配的torch和cuda，故不需要使用额外的命令安装torch和cuda。

```bash
# 下载安装torch、torchvision和cuda
pip install torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

下载安装过程中，可能由于网络问题导致下载失败，可再次运行该命令，直到下载安装成功。

下载安装完成后，执行以下代码验证安装（可新建 check_env.py 运行）：

```python
import torch
import torchvision

# 检查PyTorch版本
print("PyTorch版本:", torch.__version__)  # 应输出 1.7.0+cu110
print("torchvision版本:", torchvision.__version__)  # 应输出0.8.1+cu101

# 检查CUDA是否可用
print("CUDA是否可用:", torch.cuda.is_available())  # 应输出 True
print("CUDA版本:", torch.version.cuda)  # 应输出 11.0
```

输出结果如下：

```bash
(d2mvsc) PS E:\deep multi-view subspace clustering\D2MVSC-main\D2MVSC> python check_env.py # 命令行运行check_env.py文件

PyTorch版本: 1.7.0+cu110
torchvision版本: 0.8.1+cu101
CUDA是否可用: True
CUDA版本: 11.0
```

### 步骤3：安装scikit-learn 0.22.2

运行命令`python train.py`，输出以下报错：

```bash
(d2mvsc) PS E:\deep multi-view subspace clustering\D2MVSC-main\D2MVSC> python train.py  

(2000, 216)
(2000, 76)
(2000, 64)
cuda
Traceback (most recent call last):
  File "train.py", line 4, in <module>
    import metrics as metrics
  File "E:\deep multi-view subspace clustering\D2MVSC-main\D2MVSC\metrics.py", line 4, in <module>
    from sklearn.metrics.cluster import check_clusterings
ImportError: cannot import name 'check_clusterings' from 'sklearn.metrics.cluster' (D:\miniconda3\envs\d2mvsc_new\lib\site-packages\sklearn\metrics\cluster\__init__.py)
```

这个错误是由于 默认安装scikit-learn（即 sklearn）的版本过高导致的兼容问题。具体来说，代码中使用的 from sklearn.metrics.cluster.supervised import check_clusterings 语句在当前环境的 scikit-learn 版本中找不到对应的模块。

卸载当前版本，安装0.22.2版本。

```bash
# 先卸载当前版本
pip uninstall scikit-learn -y

# 安装 0.22.2 版本（与旧路径兼容）
pip install scikit-learn==0.22.2
```

### 步骤4：运行train.py

下载预训练模型AE1.pth后，将预训练模型AE1.pth放在代码根目录（与train.py同级），因为train.py中加载路径为`./AE1.pth`。在终端运行`python train.py`命令即可。

至此，实验环境配置完成。

## 环境配置过程中出现的问题说明

### 问题1：Pytorch版本问题

D2MVSC中的readme.md文件如下写到：

```markdown
environment requirements: PyTorch-gpu 1.0, Python 3.7, numpy, scipy, sklearn, and munkres

run '''python train.py'''

pre-trained model address as following: https://github.com/jiaowangswust/D2MVSC/releases/download/v0.0.0/AE1.pth
```

原先我根据作者的readme.md文件，安装torch==1.0.1，torchvision==0.2.2和cuda==10.0，由于版本太低，许多源没有这些库，多方查找后下载成功。但当我读取预训练模型，报错如下：

```bash
(d2mvsc) PS E:\deep multi-view subspace clustering\D2MVSC-main\D2MVSC> python train.py

(2000, 216)
(2000, 76)
(2000, 64)
cuda
D:\miniconda3\envs\D2MVSC\lib\site-packages\sklearn\utils\deprecation.py:144: FutureWarning: The sklearn.metrics.cluster.supervised module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.metrics.cluster. Anything that cannot be imported from sklearn.metrics.cluster is now part of the private API.
  warnings.warn(message, FutureWarning)
cuda
Traceback (most recent call last):
  File "train.py", line 110, in <module>
    model.load_state_dict(torch.load('./AE1.pth'), strict=False)
  File "D:\miniconda3\envs\D2MVSC\lib\site-packages\torch\serialization.py", line 367, in load
    return _load(f, map_location, pickle_module)
  File "D:\miniconda3\envs\D2MVSC\lib\site-packages\torch\serialization.py", line 528, in _load
    magic_number = pickle_module.load(f)
_pickle.UnpicklingError: A load persistent id instruction was encountered,
but no persistent_load function was specified.
```

这个错误是由于 PyTorch 版本与预训练模型（AE1.pth）的序列化格式不兼容导致的。具体来说，模型文件（AE1.pth）是用较高版本的 PyTorch 保存的，而当前使用的 PyTorch 1.0.1 版本过旧，无法解析新格式的序列化数据，从而引发了 _pickle.UnpicklingError。

在搜索了资料后看到，[【报错解决】RuntimeError: xx.pt is a zip archive (did you mean to use torch.jit.load()?)](https://zhuanlan.zhihu.com/p/454415689)，我猜测pytorch的版本需要>=1.6,
重新安装高版本的pytorch，和兼容的torchvision，cuda，代码成功运行。
