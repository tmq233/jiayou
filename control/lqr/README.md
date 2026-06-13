# LQR 算法使用文档

## 算法概述

离散时间线性二次型调节器（Discrete-Time Linear Quadratic Regulator, DLQR）求解最优状态反馈控制律，使以下二次代价函数最小：

$$\min_{u_k} \sum_{k=0}^{\infty} \left( x_k^T Q x_k + u_k^T R u_k \right)$$

受限于线性动力学：$x_{k+1} = A x_k + B u_k$

最优控制律为状态反馈：$u_k^* = -K x_k$，其中增益 $K$ 通过求解离散代数 Riccati 方程（DARE）得到。

---

## API 参考

### `lqr.dlqr(A, B, Q, R)`

无限时域离散 LQR。

| 参数 | 类型 | 说明 |
|------|------|------|
| `A` | `ndarray [n, n]` | 状态转移矩阵 |
| `B` | `ndarray [n, m]` | 输入矩阵 |
| `Q` | `ndarray [n, n]` | 状态代价矩阵（半正定） |
| `R` | `ndarray [m, m]` | 控制代价矩阵（正定） |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `K` | `ndarray [m, n]` | 最优反馈增益 $u^* = -Kx$ |
| `P` | `ndarray [n, n]` | DARE 解矩阵 $J^*(x) = x^T P x$ |

### `lqr.finite_horizon_lqr(A, B, Q, R, Qf, N)`

有限时域离散 LQR，返回时变增益序列。

| 参数 | 类型 | 说明 |
|------|------|------|
| `A`, `B`, `Q`, `R` | 同上 | 同上 |
| `Qf` | `ndarray [n, n]` | 终端状态代价矩阵 |
| `N` | `int` | 时域长度（步数） |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `K_seq` | `list[ndarray]` | 时变增益序列 $[K_0, \dots, K_{N-1}]$ |
| `P_seq` | `list[ndarray]` | 时变代价序列 $[P_0, \dots, P_{N-1}, P_N]$ |

### `lqr.solve_dare(A, B, Q, R, max_iters=2000, tol=1e-10)`

直接求解 DARE（内部函数，`dlqr` 调用此函数）。

$$P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q$$

通过 Riccati 迭代求解，直到 $\max|P_{k+1} - P_k| < \text{tol}$。

---

## 使用示例

### 基本用法

```python
import numpy as np
from lqr import dlqr

# 双积分器系统: ẋ = v, v̇ = a
# 离散化 (dt=0.1):
dt = 0.1
A = np.array([[1, dt], [0, 1]])
B = np.array([[0.5 * dt**2], [dt]])

# 代价矩阵: 惩罚位置 > 速度，轻度惩罚控制
Q = np.diag([10.0, 1.0])
R = np.array([[0.1]])

# 求解 LQR
K, P = dlqr(A, B, Q, R)

# 仿真
x = np.array([1.0, 0.0])  # 初始状态: x=1, v=0
for step in range(100):
    u = -K @ x
    x = A @ x + B @ u
```

### 有限时域 LQR

```python
from lqr import finite_horizon_lqr

K_seq, P_seq = finite_horizon_lqr(A, B, Q, R, Qf=np.diag([100, 10]), N=50)

# 使用时变增益
x = x0
for k in range(50):
    u = -K_seq[k] @ x
    x = A @ x + B @ u
```

### 车辆控制演示 (`car_demo.py`)

2D 车辆 LQR 控制交互式演示，支持**命令行调参**和**交互 REPL 模式**，便于通过动手调节 $Q/R$ 权重来建立 LQR 直觉。

车辆模型为 2D 双积分器：

$$A = \begin{bmatrix} 1 & 0 & dt & 0 \\ 0 & 1 & 0 & dt \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad B = \begin{bmatrix} 0.5dt^2 & 0 \\ 0 & 0.5dt^2 \\ dt & 0 \\ 0 & dt \end{bmatrix}$$

状态 $[x, y, v_x, v_y]^T$，控制 $[a_x, a_y]^T$，输出四象限图 `car_lqr_demo.png`（轨迹、位置时程、速度时程、控制量）。

#### 四种运行模式

| 模式 | 命令 | 说明 |
|------|------|------|
| **默认** | `python car_demo.py` | 使用预设参数运行，输出 K/P/rho/cost 和图片 |
| **CLI 调参** | `python car_demo.py --qx 50 --rx 2.0` | 命令行覆盖 $Q/R/dt/x_0$ 任意参数 |
| **参数扫描** | `python car_demo.py --sweep qx 0.1 100 20` | 扫描单个参数从 lo 到 hi，绘制 cost 与 rho 变化曲线 |
| **交互模式** | `python car_demo.py -i` | 进入 REPL，逐次调参、立即看到结果对比 |

#### CLI 参数一览

```
--qx / --qy        Q 对角线位置权重（默认: 10）
--qvx / --qvy      Q 对角线速度权重（默认: 1）
--rx / --ry        R 对角线控制权重（默认: 0.5）
--dt               离散化步长 [s]（默认: 0.05）
--T                仿真时长 [s]（默认: 8.0）
--x0 X Y VX VY    初始状态（默认: 5 3 2 -1.5）
-i, --interactive  进入交互调参模式
--sweep P LO HI N  参数扫描（P: qx/qy/qvx/qvy/rx/ry）
--no-plot          抑制绘图输出
```

#### 交互模式命令

```
qx 50           增大 x 位置惩罚 → 更激进的 x 方向控制
rx 5.0          增大控制代价 → 更平滑、更慢的收敛
dt 0.02         更细的离散化步长
x0 8 0 -1 0     修改初始状态 [x, y, vx, vy]
run             用当前参数跑一次
show            回顾上一次运行结果
sweep qx 0.1 100 20  扫描 qx 并绘制趋势图
reset           恢复默认参数
help            显示所有命令
quit            退出
```

#### 学习路径建议

按以下顺序逐步建立对 $Q/R$ 的直觉：

1. `python car_demo.py` — 看默认行为（输出的是什么矩阵？rho 多大？）
2. `python car_demo.py --qx 100` — 把位置权重放大 10 倍，K 矩阵怎么变？
3. `python car_demo.py --rx 5.0` — 控制代价拉满，轨迹和控制量有什么变化？
4. `python car_demo.py --sweep qx 0.1 100 20` — 扫一整条曲线，看 rho 和 cost 随 $Q$ 的单调趋势
5. `python car_demo.py -i` — 自由探索，验证自己的猜想

---

## 关键调参指南

| 矩阵 | 作用 | 调参建议 |
|------|------|---------|
| $Q$ | 状态跟踪权重 | 增大 $Q_{ii}$ → 状态 $i$ 收敛更快，但控制量增大 |
| $R$ | 控制代价权重 | 增大 $R$ → 控制更平滑/保守，但收敛变慢 |
| $Q_f$ | 终端代价（有限时域） | 近似无限时域代价以减小末端误差 |

**经验法则**：$Q$ 与 $R$ 的相对比例决定闭环带宽，而非绝对值。Bryson 规则提供合理初值：

$$Q_{ii} = \frac{1}{\text{状态 } x_i \text{ 可接受的最大偏差}^2}, \quad R_{jj} = \frac{1}{\text{控制 } u_j \text{ 可接受的最大值}^2}$$

---

## 数学推导（简）

给定 $J^*(x) = x^T P x$，Bellman 最优性方程：

$$x^T P x = \min_u \left[ x^T Q x + u^T R u + (Ax + Bu)^T P (Ax + Bu) \right]$$

对 $u$ 求导并令为零：

$$\frac{\partial}{\partial u} = 2R u + 2B^T P (Ax + Bu) = 0 \;\Rightarrow\; u^* = -(R + B^T P B)^{-1} B^T P A \, x$$

代回得 DARE：

$$P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q$$
