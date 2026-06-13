# MPC Car Demo — 使用手册

## 1. 环境安装

```bash
# Python ≥ 3.8
pip install -r requirements.txt
```

依赖：`numpy` `scipy` `matplotlib`（均为标准科学计算库，无特殊依赖）。

---

## 2. 启动 Demo

```bash
python demo.py
```

启动后弹出两个区域：

```
┌─────────────────────────────────┬──────────────────┐
│                                 │   ═══ MPC ═══    │
│   主视图（车辆 + 轨迹）          │   参数状态面板     │
│                                 │                  │
│   · 橙色虚线 = 参考路径          │   键盘快捷键表    │
│   · 绿色实线 = 实际轨迹          │                  │
│   · 黄色圆点 = MPC 预测轨迹      │                  │
│   · 蓝色三角 = 车辆位置/朝向      │                  │
│                                 │                  │
├─────────────────────────────────┴──────────────────┤
│  [滑条 × 9]  Horizon N  dt  Speed  Q_xy  Q_θ  ...  │
│  [按钮 × 5]  Reset  Pause  Circle  Figure-8  Noise │
└─────────────────────────────────────────────────────┘
```

---

## 3. 交互控制

### 3.1 鼠标操作

| 控件 | 作用 |
|------|------|
| **拖拽滑条** | 实时调整 MPC 参数，松手即生效 |
| **Reset 按钮** | 车辆回到起点，清空轨迹 |
| **Pause 按钮** | 暂停 / 继续模拟 |
| **Circle 按钮** | 切换到圆形轨迹 |
| **Figure-8 按钮** | 切换到 8 字形轨迹 |
| **+Noise 按钮** | 开关过程噪声（模拟传感器误差） |

### 3.2 键盘快捷键

| 键 | 功能 |
|----|------|
| `Space` | 暂停 / 继续 |
| `R` | 复位 |
| `1` | 圆形轨迹 |
| `2` | Figure-8 轨迹 |
| `N` | 开关噪声 |

---

## 4. 参数滑条详解

### 4.1 规划参数

| 滑条 | 范围 | 默认 | 含义 |
|------|------|------|------|
| **Horizon N** | 3 – 30 | 15 | MPC 向前预测的步数 |
| **dt** | 0.05 – 0.2 s | 0.1 | 离散时间步长 |
| **Target speed** | 2 – 20 m/s | 8 | 参考轨迹期望车速 |
| **Wheelbase L** | 1.5 – 4.0 m | 2.5 | 车辆轴距 |

**Horizon N 的影响：**
- N 越大 → "看得越远"，过弯更平顺提前，但计算量线性增长
- N 越小 → 响应快但可能"短视"，急弯容易切出路径
- 一般 10–20 平衡效果与性能

**dt 的影响：**
- dt 越大 → 控制频率低，计算轻松，但离散化误差增大
- dt 越小 → 控制精细，但 N 相同时预测总时长变短
- 实际控制频率 = 1/dt Hz（默认 10 Hz）

### 4.2 代价权重

| 滑条 | 范围 | 默认 | 控制什么 |
|------|------|------|----------|
| **Q position** | 0.1 – 100 | 10 | 位置误差的惩罚 → 越大，跟踪越紧 |
| **Q heading** | 0.1 – 50 | 5 | 航向角误差的惩罚 → 越大，车头越贴合路径方向 |
| **Q velocity** | 0.01 – 20 | 1 | 速度误差的惩罚 → 越大，车速越贴近目标 |
| **R accel** | 0.01 – 10 | 0.1 | 加速度的惩罚 → 越大，加/减速越柔和 |
| **R steer** | 0.01 – 10 | 0.1 | 转向角的惩罚 → 越大，方向盘动作越平滑 |

**代价函数：**
$$J = \sum_{k=0}^{N-1} \Big[ (x_k - r_k)^\top Q (x_k - r_k) + u_k^\top R u_k \Big] + \text{terminal cost}$$

- $Q$ 大 → 优先减小跟踪误差（激进跟踪）
- $R$ 大 → 优先省力省能量（平滑保守）

---

## 5. 典型调参场景

### 5.1 "我要车死死咬住参考线"

```
Q_xy:     10 → 50
Q_theta:   5 → 20
R_steer: 0.1 → 0.01
```

效果：位置偏差压缩到毫米级，但转向会频繁微调（可能抖动）。

### 5.2 "我要丝滑如自动驾驶"

```
R_accel: 0.1 → 2.0
R_steer: 0.1 → 2.0
Q_xy:    10 → 5
```

效果：加速柔和、转向平滑，代价是弯道可能轻微切出。

### 5.3 "我要它提前看路、预判过弯"

```
N:       15 → 25
dt:     0.1 → 0.08
```

效果：预测窗口从 1.5s 扩展到 2.0s。车会在入弯前就开始转向。

### 5.4 "速度必须严格锁死在目标值"

```
Q_v:      1 → 10
R_accel: 0.1 → 0.05
```

效果：速度跟踪精度大幅提升，油门响应更积极。

---

## 6. 代码接口

### 6.1 快速上手

```python
import numpy as np
from mpc_core import MPCController

# 创建 MPC
mpc = MPCController(
    dt=0.1,                          # 时间步长 [s]
    N=15,                            # 预测时域步数
    Q_diag=np.array([10, 10, 5, 1]), # Q: [px, py, theta, v]
    R_diag=np.array([0.1, 0.1]),     # R: [accel, steer]
    L=2.5,                           # 轴距 [m]
)

# 每个控制周期调用
x0 = np.array([0.0, 0.0, 0.0, 5.0])    # 当前状态 (px, py, heading, speed)
x_ref = np.zeros((mpc.N + 1, 4))        # 未来 N+1 步参考状态
u_opt, u_seq, x_pred, info = mpc.solve(x0, x_ref)
# u_opt = [accel, steering_angle]  — 施加到车辆
```

### 6.2 运行时修改参数

```python
# 无需重新创建对象，直接更新
mpc.update_params(
    dt=0.08,
    N=20,
    Q_diag=np.array([50.0, 50.0, 10.0, 2.0]),
    R_diag=np.array([0.05, 0.05]),
)
# 注意：update_params 会重置 warm-start 缓存
```

### 6.3 运行 Benchmark

```bash
# 默认参数，两条轨迹各跑 300 步
python benchmark.py

# 自定义参数
python benchmark.py --path circle --steps 500 --N 20 --Q_xy 50 --v_des 12

# 加噪声
python benchmark.py --path figure8 --noise_std 0.05 --steps 200

# 查看所有参数
python benchmark.py --help
```

输出示例：
```
═══ Benchmark Results: circle ═══
  Position error — mean: 0.0004 m,  max: 0.0128 m
  Velocity error — mean: 0.0003 m/s
  Solve time — mean: 39.9 ms,  max: 85.7 ms
  Solve rate: 25.1 Hz
```

---

## 7. 车辆模型参数

Kinematic Bicycle Model：

| 参数 | 符号 | 默认值 | 单位 |
|------|------|--------|------|
| 轴距 | $L$ | 2.5 | m |
| 最大加速度 | $a_{\max}$ | $\pm 3$ | m/s² |
| 最大转向角 | $\delta_{\max}$ | $\pm 45^\circ$ | rad |
| 最高车速 | $v_{\max}$ | 20 | m/s |

可通过 `MPCController` 构造函数参数 `u_min`, `u_max`, `v_max` 自定义：
```python
mpc = MPCController(
    ...,
    u_min=np.array([-2.0, -np.pi/6]),   # 更保守的控制限幅
    u_max=np.array([2.0, np.pi/6]),
    v_max=15.0,
)
```

---

## 8. 常见问题

**Q: Demo 窗口卡顿？**
- 减小 Horizon N（如 10），或增大 dt（如 0.15s），降低优化问题规模。
- 关闭其他 GPU 密集型应用释放资源。

**Q: 车辆在急弯处飞出参考线？**
- 增大 Q_xy 或 Q_theta，让控制器更重视跟踪精度。
- 降低 Target speed，弯道速度过快会超出转向限幅。

**Q: 方向盘来回抖动？**
- 增大 R_steer，惩罚转向变化。
- 适当降低 Q_theta。

**Q: 想在代码中使用自己的参考轨迹？**
- 参考轨迹格式为 `(N+1, 4)` 的 numpy 数组：`[px, py, theta, v]`
- 传给 `mpc.solve(x0, x_ref)` 即可，生成逻辑完全开放。

**Q: 求解器报错 / 返回 success=False？**
- 检查初始状态是否在合理范围。
- 参考轨迹是否包含不可能的机动（如瞬间变向、超速）。
- 增大 `maxiter`（在 `mpc_core.py` 的 `minimize` 调用中修改）。
