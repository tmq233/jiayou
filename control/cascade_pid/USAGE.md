# 串级 PID 演示程序 — 使用手册

## 环境要求

- Python 3.8+
- numpy
- matplotlib

```bash
pip install numpy matplotlib
```

## 快速开始

```bash
cd cascade_pid
python main.py
```

运行后控制台打印性能指标对比表，同时生成 5 张 PNG 对比图。

---

## 命令行参数

当前版本无命令行参数，所有配置在源码中直接修改。以下为常用修改入口。

---

## 修改被控对象参数

编辑 `scenarios.py` 中的 `_make_plants()`：

```python
def _make_plants():
    return (DCMotorPlant(J=0.05,   # 惯量 kg·m²（越小越快）
                         B=0.05,   # 粘性摩擦 Nm·s/rad
                         K=1.0,    # 力矩常数 Nm/控制单位
                         max_torque=5.0),  # 力矩饱和 (Nm)
            DCMotorPlant(J=0.05, B=0.05, K=1.0, max_torque=5.0))
```

**调参建议**：
| 参数 | 增大效果 | 减小效果 |
|------|----------|----------|
| `J` | 响应变慢，超调增加 | 响应变快，可能振荡 |
| `B` | 阻尼增加，超调减少 | 欠阻尼，振荡加剧 |
| `K` | 控制效率提高 | 控制效率降低 |
| `max_torque` | 允许更大加减速 | 限制加减速能力 |

---

## 修改控制器参数

编辑 `scenarios.py` 中的 `_make_controllers()`：

```python
def _make_controllers(dt_single=0.01, dt_outer=0.01, dt_inner=0.001,
                      vel_limit=8.0, torque_limit=5.0):

    # === 单级 PID ===
    single = PIDController(
        kp=2.4,   # 比例增益（位置刚度）
        ki=3.2,   # 积分增益（消静差）
        kd=0.55,  # 微分增益（阻尼）
        dt=dt_single,           # 控制周期 (s)
        output_limits=(-torque_limit, torque_limit),  # 输出限幅
        tau_f=0.01              # 微分低通滤波时间常数
    )

    # === 内环速度 PI ===
    inner = PIDController(
        kp=2.0,   # 速度比例增益
        ki=20.0,  # 速度积分增益（要高，快速消速差）
        kd=0.05,  # 速度微分（通常很小）
        dt=dt_inner,
        output_limits=(-torque_limit, torque_limit),
        tau_f=0.002
    )

    # === 外环位置 PI ===
    outer = PIDController(
        kp=2.0,   # 位置比例（映射误差→速度指令）
        ki=0.3,   # 位置积分（要小，防 windup）
        kd=0.0,   # 位置微分（内环已提供阻尼）
        dt=dt_outer,
        output_limits=(-vel_limit, vel_limit),
        tau_f=0.01
    )

    cascade = CascadePID(outer, inner, vel_limit=vel_limit)
    return single, cascade
```

### 调参原则

1. **先调内环，再调外环**：内环速度 PI 独立调好（阶跃响应快且无超调），再调外环
2. **外环 Ki 要小**：外环位置 I 过大会导致积分 windup → 严重超调。本演示用 $K_{ip}=0.3$
3. **外环 Kd 通常为 0**：内环已提供速度阻尼，外环 P 控制即可

### 公式参考（二项式极点配置法）

单级 PID（$\omega_0=4$）：$K_d = 3J\omega_0 - B$，$K_p = 3J\omega_0^2$，$K_i = J\omega_0^3$

内环速度 PI（$\zeta=1, \omega_v=20$）：$K_{pv} = 2J\omega_v - B$，$K_{iv} = J\omega_v^2$

---

## 修改仿真参数

编辑 `scenarios.py` 中各场景函数的 `run_comparison()` 调用：

```python
(res_s, res_c) = run_comparison(
    p_s, p_c, single_pid, cascade_pid,
    ref_func,                  # 参考轨迹函数 ref_func(t) -> float
    dist_func=dist_func,       # 扰动函数 dist_func(t) -> float (None = 无扰动)
    noise_func=noise_func,     # 噪声函数 noise_func(t) -> float (None = 无噪声)
    param_change_func=param_change,  # 参数变化函数 f(plant, t)
    t_final=5.0,               # 仿真时长 (s)
    sim_dt=1e-4,               # 积分步长 (s)，默认 10 kHz
    single_ratio=100,          # 单级 PID 每 N 个积分步更新一次 (100 → 100 Hz)
    cascade_outer_ratio=100,   # 外环更新间隔 (100 → 100 Hz)
    cascade_inner_ratio=10,    # 内环更新间隔 (10 → 1 kHz)
)
```

### 多速率控制

| 参数 | 默认值 | 频率 | 含义 |
|------|--------|------|------|
| `sim_dt` | 1e-4 | 10 kHz | RK4 积分步长 |
| `single_ratio` | 100 | 100 Hz | 单级 PID 更新频率 |
| `cascade_outer_ratio` | 100 | 100 Hz | 外环位置更新频率 |
| `cascade_inner_ratio` | 10 | 1 kHz | 内环速度更新频率 |

内环比外环快 10× ——模拟真实硬件中编码器 (kHz) vs 视觉 (100Hz) 的速率差异。

---

## 噪声场景的 vel_ref 滤波

编辑 `scenario_noise()` 中的 `vel_ref_alpha`：

```python
cascade_pid = CascadePID(outer, inner, vel_limit=8.0,
                         vel_ref_alpha=0.3)   # 0.3 = 强滤波, 1.0 = 无滤波
```

$|\alpha| \le 1$，越小滤波越强。这是串级结构独有的自由度——平滑 $\omega_{ref}$ 不影响内环扰动抑制能力。

---

## 添加自定义场景

```python
def scenario_my_custom():
    single_pid, cascade_pid = _make_controllers()
    p_s, p_c = _make_plants()

    def ref_func(t):
        return np.sin(2 * np.pi * 0.5 * t)  # 正弦跟踪

    def dist_func(t):
        return 0.5 * np.sin(2 * np.pi * 3 * t)  # 周期扰动

    (res_s, res_c) = run_comparison(p_s, p_c, single_pid, cascade_pid,
                                    ref_func, dist_func=dist_func,
                                    t_final=10.0)
    return res_s, res_c, "Custom Scenario: Sine Tracking"

# 在 SCENARIOS 列表末尾添加
SCENARIOS = [
    scenario_step_response,
    scenario_disturbance,
    scenario_parameter_change,
    scenario_noise,
    scenario_my_custom,  # ← 新增
]
```

---

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `comparison_S1.png` | 阶跃响应：位置、速度、力矩、误差 4 子图对比 |
| `comparison_S2.png` | 扰动抑制：含扰动前/后细节 |
| `comparison_S3.png` | 参数鲁棒性：惯量变化前后对比 |
| `comparison_S4.png` | 噪声场景：控制信号高频分量对比 |
| `metrics_summary.png` | 超调量、调节时间、ISE、IAE 柱状图汇总 |

每张图的 2×2 布局：**左上** 位置跟踪 | **右上** 速度 | **左下** 控制力矩 | **右下** 跟踪误差

单级 PID = 蓝色 (`C0`)，串级 PID = 橙色 (`C1`)，速度参考 = 绿色虚线 (`C2`)。

---

## 常见问题

**Q: 串级超调比单级还大？**
A: 外环积分增益 $K_{ip}$ 过大导致 windup。减小 $K_{ip}$ 至 0.3 以下。

**Q: 扰动抑制效果不明显？**
A: 增大扰动力矩（`dist_func` 中改大 $\tau_{dist}$），或减小内环带宽（降低 $K_{pv}, K_{iv}$）。

**Q: 想用自己调好的参数运行？**
A: 直接改 `_make_controllers()` 中的增益值即可，不需要修改其他文件。

**Q: 仿真太慢？**
A: 增大 `sim_dt`（如 5e-4），减少 `t_final`。注意积分精度会下降。
