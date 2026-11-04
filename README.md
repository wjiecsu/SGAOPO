# Offline Reinforcement Learning

## 1. 项目简介

本项目实现了一种面向离线强化学习（Offline Reinforcement Learning,
Offline
RL）的策略学习方法，旨在利用预先收集的离线数据集进行策略优化，避免训练过程中与环境进行实时交互。

项目基于 D4RL 基准测试环境开展实验，主要面向 Gym-MuJoCo
连续控制任务，用于评估算法在不同离线数据集上的策略学习性能。

## 2. 实验环境

本项目运行涉及以下主要软件及工具：

-   Python：算法开发与运行环境
-   PyTorch：深度学习框架
-   D4RL：离线强化学习基准数据集
-   Gym：强化学习环境接口
-   NumPy：数值计算与数据处理
-   CUDA：GPU 加速计算

请根据实际运行环境配置相应的软件依赖及 GPU
驱动。具体依赖版本以项目实际代码和依赖文件为准。

## 3. 实验基准

本项目采用 D4RL 基准测试平台开展离线强化学习实验。

### 3.1 Gym-MuJoCo 环境

  环境          任务说明
  ------------- --------------------------
  HalfCheetah   半猎豹机器人连续运动控制
  Hopper        单腿机器人跳跃与平衡控制
  Walker2d      双足机器人行走控制

### 3.2 数据集类型

  数据集          数据特点
  --------------- ----------------------------------
  Random          由随机策略收集的数据
  Medium          由中等性能策略收集的数据
  Medium-Replay   策略训练过程中累积的经验数据
  Medium-Expert   中等性能策略与专家策略混合的数据

常用环境名称示例：

-   `halfcheetah-random-v2`
-   `halfcheetah-medium-v2`
-   `halfcheetah-medium-replay-v2`
-   `halfcheetah-medium-expert-v2`
-   `hopper-random-v2`
-   `walker2d-medium-v2`

## 4. 项目结构

``` text
.
├── main.py          # 主程序及训练入口
├── README.md        # 项目说明文档
└── ...              # 其他算法模块及配置文件
```

`main.py`
是项目的主要运行入口。程序通过命令行参数指定实验环境、随机种子及算法超参数。

## 5. 运行方法

### 5.1 运行环境准备

运行前请确认：

1.  已正确配置 Python 运行环境。
2.  已安装与当前环境兼容的 PyTorch。
3.  已配置可用的 NVIDIA GPU 和 CUDA 环境。
4.  已安装项目所需依赖。
5.  对应的 D4RL 数据集能够被程序正确加载。

具体安装命令和依赖版本请以项目提供的依赖文件为准。

### 5.2 启动训练

在项目根目录下执行以下命令：

``` bash
gpu-run --gpus 1 -- python main.py --env=halfcheetah-random-v2 --seed=0 --vae_weight=1.0 --bc_weight=1.0 --discount=0.99 --alpha=40 --Is_SubOptimal=False
```

该命令使用一张 GPU 运行 `main.py`，在 `halfcheetah-random-v2`
数据集上启动实验，并按照指定参数配置模型训练。

### 5.3 命令行参数说明

  参数                                   示例值 说明
  ------------------- ------------------------- --------------------------------------
  `--env`               `halfcheetah-random-v2` 指定 D4RL 实验环境及数据集名称
  `--seed`                                  `0` 随机种子，用于控制实验随机性
  `--vae_weight`                          `1.0` VAE 模块对应的损失权重
  `--bc_weight`                           `1.0` 行为克隆相关损失的权重
  `--discount`                           `0.99` 折扣因子 γ
  `--alpha`                                `40` 算法超参数 α，具体作用以代码实现为准
  `--Is_SubOptimal`                     `False` 是否启用次优数据设置
