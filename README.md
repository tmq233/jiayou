# jiayou — 控制理论与深度学习（面向 Sim-to-Real 机器人）

> 实事求是，按需学习。从经典控制到非线性系统，再到神经网络的表征学习，一切为 Sim-to-Real 服务。

## 项目结构

```
jiayou/
├── control/
│   ├── pid_demo/        # PID 基础与调参实验
│   ├── cascade_pid/     # 串级 PID — 多回路级联控制
│   ├── lqr/             # Linear Quadratic Regulator
│   ├── mpc/             # Model Predictive Control
│   ├── lyapunov/        # Lyapunov 稳定性与 CLF
│   ├── contraction/     # 收缩分析与增量稳定性
│   └── cbf/             # Control Barrier Functions
├── deep_learning/
│   ├── nn_core.py       # 纯 numpy 神经网络构建块
│   └── demos/           # 可运行实验
│       ├── demo_01_backprop.py    # 前向/反向传播
│       ├── demo_02_init.py        # 权重初始化策略
│       ├── demo_03_norm.py        # LayerNorm vs BatchNorm
│       ├── demo_04_residual.py    # 残差连接
│       └── demo_05_dropout.py     # Dropout 与正则化
├── README.md
├── .gitignore
└── docs/
    └── work_log.md      # 开发决策日志
```

## 学习路线

### 第一阶段：经典控制 ✅ 已完成

| 主题 | 状态 |
|------|------|
| PID 控制与调参 | ✅ `control/pid_demo` |
| 串级 PID | ✅ `control/cascade_pid` |
| LQR | ✅ `control/lqr` |
| MPC | ✅ `control/mpc` |

### 第二阶段：稳定性分析与非线性控制 ✅ 已完成

| 主题 | 状态 |
|------|------|
| Lyapunov 直接法与 CLF | ✅ `control/lyapunov` |
| 收缩分析与增量稳定性 | ✅ `control/contraction` |
| Control Barrier Functions (CBF) | ✅ `control/cbf` |

**学习要点：**
- **Lyapunov 直接法**：从线性 Lyapunov 方程 $A^T P + PA = -Q$ 到非线性 Lie 导数 $L_f V = \nabla V \cdot f$，理解稳定性判据从局部到全局的推广路径
- **Control Lyapunov Functions (CLF)**：Sontag 通用公式提供了一种构造性方法来设计镇定控制器 —— $\dot V < 0$ 是唯一的硬约束
- **收缩分析**：比 Lyapunov 方法更强的性质 —— 所有轨迹指数收敛到同一条（与初始条件无关），特别适合观测器和同步问题
- **CBF 安全滤波器**：在标称控制器输出端加一层 QP 优化，将安全性编码为硬约束（$h(x) \ge 0$），稳定性编码为软约束（CLF relaxed by slack）

### 第三阶段：神经网络与表征学习 ✅ 已完成

**核心问题：什么样的网络结构能让 RL 策略在 POMDP 和时序控制任务中稳定收敛？**

| 主题 | 状态 |
|------|------|
| 前向传播与反向传播 | ✅ `deep_learning/demos/demo_01_backprop.py` |
| 权重初始化 | ✅ `deep_learning/demos/demo_02_init.py` |
| 归一化层 | ✅ `deep_learning/demos/demo_03_norm.py` |
| 残差连接 | ✅ `deep_learning/demos/demo_04_residual.py` |
| 正则化与 Dropout | ✅ `deep_learning/demos/demo_05_dropout.py` |

**关键理论：**

- **前向传播与反向传播**
  - 链式法则下的梯度流分析
  - 梯度消失/爆炸的数学临界条件：激活函数导数 $\times$ 权重矩阵范数在深层网络中的指数级传播

- **权重初始化**
  - **Xavier (Glorot) 初始化**：保持 Tanh 各层输入输出方差一致 —— $\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$
  - **He (Kaiming) 初始化**：针对 ReLU 的修正 —— $\text{Var}(W) = \frac{2}{n_{\text{in}}}$
  - **正交初始化**：在 PPO 策略网络中维持初始探索多样性，防止梯度崩塌

- **归一化层**
  - **Layer Normalization (LN)**：在 feature 维度上归一化，不依赖 batch 统计量，是 RL/POMDP 场景的首选
  - **为什么 RL 中 LN 优于 BN？** Batch Normalization 严重依赖 batch size，且会混淆不同环境轨迹的独立性，导致策略更新时产生高方差振荡

- **残差连接**
  - $\frac{\partial H}{\partial x} = \frac{\partial F}{\partial x} + I$ —— 恒等映射 $I$ 保证即使 $\frac{\partial F}{\partial x} \to 0$，梯度依然可以无损回传

- **正则化与 Dropout**
  - Dropout 本质：训练时做隐式集成（Ensemble），推理时做缩放补偿
  - **控制系统中的陷阱**：推理阶段若保持 Dropout 开启，动作输出会引入非确定性随机噪声，部署时必须关闭

## 快速运行

```bash
# 经典控制
python control/pid_demo/main.py
python control/cascade_pid/main.py
python control/lqr/car_demo.py
python control/mpc/demo.py

# 非线性控制
python control/lyapunov/demo.py
python control/contraction/demo.py
python control/cbf/demo.py

# 深度学习基础
python deep_learning/main.py          # 运行所有 demo
python deep_learning/main.py 1        # 运行第 1 个
```

## 协作约定

- 代码注释和变量名使用英文
- 文档与讨论使用中文
- Git 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范
- 张量操作必须向量化，关键维度变换（`view` / `permute` / `einsum`）后标注维度注释 `# [B, T, dim]`
