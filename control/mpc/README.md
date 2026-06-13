# MPC Car Tracking Demo

非线性 Model Predictive Control (MPC) 实现，使用 kinematic bicycle model 控制车辆跟踪参考轨迹。提供完整的数学推导、参数调优指南，以及交互式可视化 Demo。

## 快速开始

```bash
pip install -r requirements.txt
python demo.py
```

- **拖动底部滑块**实时调整 MPC 参数
- **按钮 / 键盘**切换轨迹、复位、加减噪声
- 所有参数即时生效，无需重启

---

## 1. 数学模型

### 1.1 车辆模型 — Kinematic Bicycle

$$
\begin{aligned}
p_x^{k+1} &= p_x^{k} + v^{k} \cos \theta^{k} \cdot \Delta t \\[4pt]
p_y^{k+1} &= p_y^{k} + v^{k} \sin \theta^{k} \cdot \Delta t \\[4pt]
\theta^{k+1} &= \theta^{k} + \frac{v^{k}}{L} \tan \delta^{k} \cdot \Delta t \\[4pt]
v^{k+1} &= v^{k} + a^{k} \cdot \Delta t
\end{aligned}
$$

**状态** $x = [p_x,\, p_y,\, \theta,\, v]^\top$ — 二维位置、航向角、纵向速度

**控制** $u = [a,\, \delta]^\top$ — 纵向加速度、前轮转向角

**参数** $L$ — 轴距，$\Delta t$ — 离散时间步长

### 1.2 MPC 优化问题

在每个控制周期求解以下有限时域最优控制问题：

$$
\begin{aligned}
\min_{u_0,\,\dots,\,u_{N-1}} \quad &
\sum_{k=0}^{N-1} \Big[ (x_k - r_k)^\top Q (x_k - r_k) + u_k^\top R\, u_k \Big]
\;+\; (x_N - r_N)^\top Q_f (x_N - r_N) \\[6pt]
\text{s.t.} \quad & x_{k+1} = f(x_k, u_k), \quad k = 0,\dots,N-1 \\
& u_{\min} \le u_k \le u_{\max} \\
& 0 \le v_k \le v_{\max}
\end{aligned}
$$

- **$N$** — 预测时域 (Horizon)，控制精度与计算量的权衡
- **$Q = \operatorname{diag}(q_{xy},\, q_{xy},\, q_\theta,\, q_v)$** — 状态跟踪权重
- **$R = \operatorname{diag}(r_a,\, r_\delta)$** — 控制代价权重
- **$Q_f = 2Q$** — 终端代价（增大以保证闭环稳定性）
- **Warm-start**：利用上一时刻的最优解 `shift` 后作为当前初值，大幅加速收敛

### 1.3 求解器

使用 `scipy.optimize.minimize` + **SLSQP** 方法求解带约束非线性规划。决策变量维度 $2N$（每个时刻 2 维控制）。

---

## 2. 参考轨迹

| 轨迹 | 表达式 | 说明 |
|------|--------|------|
| **Circle** | $(R\cos\omega t,\; R\sin\omega t)$ | 定速圆周，速度 $v_{\text{des}}$ 恒定 |
| **Figure-8** | $(A\sin\omega t,\; B\sin 2\omega t)$ | Lemniscate of Gerono，速度按弧长变化 |

参考状态 $r(t) = [p_x(t),\, p_y(t),\, \theta(t),\, v(t)]^\top$ 解析生成，$\theta$ 由路径切线方向决定。

---

## 3. 参数调优指南

### 3.1 滑条参数一览

| 滑块 | 范围 | 默认值 | 含义 |
|------|------|--------|------|
| **Horizon N** | 3 – 30 | 15 | 预测步数，越大越"远见"但计算越慢 |
| **dt** | 0.05 – 0.2 s | 0.1 | 离散时间步长 |
| **Target speed** | 2 – 20 m/s | 8 | 期望车速 |
| **Q position** | 0.1 – 100 | 10 | 位置误差代价 → 轨迹跟踪紧度 |
| **Q heading** | 0.1 – 50 | 5 | 航向误差代价 → 车头朝向对齐 |
| **Q velocity** | 0.01 – 20 | 1 | 速度误差代价 → 速度跟踪 |
| **R accel** | 0.01 – 10 | 0.1 | 加速度代价 → 越大越"温柔" |
| **R steer** | 0.01 – 10 | 0.1 | 转向代价 → 越大转向越平滑 |
| **Wheelbase L** | 1.5 – 4.0 m | 2.5 | 轴距 |

### 3.2 典型调参策略

| 目标 | 调法 |
|------|------|
| **跟踪更紧** | ↑ Q_xy, ↓ R_ctrl |
| **转向更平滑** | ↑ R_steer, ↑ Q_theta |
| **速度跟踪优先** | ↑ Q_v, ↓ R_accel |
| **加速更柔和** | ↑ R_accel, ↓ Q_v |
| **计算更快** | ↓ N, ↑ dt |
| **更远见（切弯提前）** | ↑ N, ↓ dt |

### 3.3 代码接口

```python
from mpc_core import MPCController
import numpy as np

mpc = MPCController(
    dt=0.1,                    # 时间步长 [s]
    N=15,                      # 预测时域
    Q_diag=np.array([10, 10, 5, 1]),   # px, py, theta, v
    R_diag=np.array([0.1, 0.1]),       # a, delta
    L=2.5,                     # 轴距 [m]
)

# 运行时更新参数（会重置 warm-start）
mpc.update_params(N=20, dt=0.08,
                  Q_diag=np.array([20, 20, 10, 2]),
                  R_diag=np.array([0.05, 0.05]))

# 求解
x0 = np.array([0.0, 0.0, 0.0, 5.0])      # 当前状态 (4,)
x_ref = np.zeros((mpc.N + 1, 4))           # 参考轨迹 (N+1, 4)
u_opt, u_seq, x_pred, info = mpc.solve(x0, x_ref)
```

---

## 4. 项目结构

```
mpc/
├── mpc_core.py         # MPC 求解器核心
│   └── MPCController   #   非线性 MPC, SLSQP + warm-start
├── vehicle_model.py    # 车辆动力学
│   └── VehicleModel    #   Kinematic bicycle, 可选噪声
├── demo.py             # 交互式 Demo
│   └── MPCDemo         #   可视化、滑条/按钮/键盘控制
├── requirements.txt    # Python 依赖
└── README.md           # 本文档
```

---

## 5. 键盘快捷键

| 键 | 功能 |
|----|------|
| `Space` | 暂停 / 继续 |
| `R` | 复位到起点 |
| `1` | 切换到圆形轨迹 |
| `2` | 切换到 Figure-8 轨迹 |
| `N` | 切换过程噪声（模拟真实传感器误差） |

---

## 6. 依赖

- Python ≥ 3.8
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Matplotlib ≥ 3.5
