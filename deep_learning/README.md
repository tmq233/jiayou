# 神经网络与表征学习基础（纯 NumPy 实现）

## 简介

一套纯 NumPy 实现的神经网络构建块，目标是从第一性原理理解深度学习中与控制/RL 相关的关键设计选择。不依赖 PyTorch/TensorFlow —— 每一行梯度计算都可见。

## 核心概念

| 概念 | 关键公式 | 控制/RL 相关性 |
|------|---------|---------------|
| 反向传播 | 链式法则 $\frac{\partial L}{\partial W_l}$ | 理解策略梯度 |
| Xavier 初始化 | $\text{Var}(W) = \frac{2}{n_{in}+n_{out}}$ | 防止梯度消失 |
| He 初始化 | $\text{Var}(W) = \frac{2}{n_{in}}$ | ReLU 网络训练稳定性 |
| 正交初始化 | $W^T W \approx I$ | PPO 探索多样性 |
| LayerNorm | 按特征维度归一化 | RL 小 batch/单轨迹训练 |
| 残差连接 | $\frac{\partial H}{\partial x} = \frac{\partial F}{\partial x} + I$ | 深层策略网络梯度流 |
| Dropout | 训练时随机丢弃神经元 | 部署时**必须关闭**（否则动作含随机噪声） |

## 文件结构

```
deep_learning/
├── __init__.py                          # 模块导出
├── nn_core.py                           # 所有构建块（~400 行）
│   ├── 权重初始化: xavier / he / orthogonal
│   ├── 层: Linear, ReLU, Tanh, Sigmoid
│   ├── 归一化: LayerNorm, BatchNorm
│   ├── 正则化: Dropout (inverted)
│   ├── 容器: Sequential
│   ├── 损失: MSELoss, CrossEntropyLoss
│   └── 优化器: SGD (momentum + weight_decay)
├── demos/                               # 5 个可运行实验
│   ├── demo_01_backprop.py
│   ├── demo_02_init.py
│   ├── demo_03_norm.py
│   ├── demo_04_residual.py
│   └── demo_05_dropout.py
├── main.py                              # 统一入口
└── README.md
```

## 快速开始

```bash
# 运行所有 demo
python deep_learning/main.py

# 运行单个 demo
python deep_learning/main.py 1    # 反向传播
python deep_learning/main.py 2    # 权重初始化
python deep_learning/main.py 3    # LayerNorm vs BatchNorm
python deep_learning/main.py 4    # 残差连接
python deep_learning/main.py 5    # Dropout
```

## Demo 说明

### Demo 1: 反向传播 — 梯度流分析
用 4 层 MLP 拟合 $\sin(2x)$，可视化：
- 训练 loss 收敛曲线
- 每层梯度范数的演化（是否出现梯度消失/爆炸）
- 最终拟合效果

### Demo 2: 权重初始化策略
12 层 MLP 下对比 Xavier / He / 正交初始化的效果：
- ReLU + Xavier → 方差衰减（梯度消失）
- ReLU + He → 方差保持（正确选择）
- Tanh + Xavier → 方差保持
- Tanh + He → 方差放大

### Demo 3: LayerNorm vs BatchNorm — 为什么 RL 用 LN？
从三个维度对比：
1. **Batch size 敏感度**：BN 在小 batch 下方差剧烈波动
2. **Train/Eval 差异**：BN 训练/推理行为不一致
3. **轨迹独立性**：BN 跨样本混合，破坏 RL 轨迹独立性
4. **小 batch 训练稳定性**：LN loss 更平滑

### Demo 4: 残差连接 — 梯度高速公路
10 个残差 block vs 同深度 plain MLP：
- ResNet 的梯度比（最后/第一层）稳定在 1.0 附近
- Plain MLP 的梯度比持续下降（梯度消失）
- 验证 $\frac{\partial H}{\partial x} = \frac{\partial F}{\partial x} + I$ 的恒等映射保梯
度

### Demo 5: Dropout — 隐式集成与部署陷阱
- **训练**：$p=0.5$ 的 Dropout 有效抑制过拟合
- **推理**：`model.eval()` 关闭 Dropout，输出确定性
- **陷阱演示**：Monte Carlo 采样展示 train mode 下的输出噪声（$\sigma > 0$），eval mode 下 $\sigma = 0$

## 依赖

- numpy
- matplotlib
