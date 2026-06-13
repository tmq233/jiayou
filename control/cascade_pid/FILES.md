# 文件夹结构与文件说明

```
cascade_pid/
│
├── 📄 README.md              ← 理论推导 + 仿真结果（先看这个）
├── 📄 USAGE.md               ← 使用手册（改参数、加场景看这个）
├── 📄 FILES.md               ← 本文件（文件夹结构说明）
│
├── 🐍 main.py                ← 程序入口
├── 🐍 pid_controller.py      ← PID 控制器模块
├── 🐍 plant_model.py         ← 被控对象模块
├── 🐍 simulation.py          ← 仿真引擎模块
├── 🐍 scenarios.py           ← 场景定义模块
├── 🐍 plot_results.py        ← 可视化与指标模块
│
├── 📊 comparison_S1.png      ← 场景 1 对比图（阶跃响应）
├── 📊 comparison_S2.png      ← 场景 2 对比图（扰动抑制）
├── 📊 comparison_S3.png      ← 场景 3 对比图（参数鲁棒性）
├── 📊 comparison_S4.png      ← 场景 4 对比图（测量噪声）
├── 📊 metrics_summary.png    ← 四场景性能指标汇总柱状图
│
├── 📁 .claude/               ← Claude Code 计划文件（可忽略）
└── 📁 __pycache__/           ← Python 字节码缓存（可忽略）
```

---

## 源文件详解

### `main.py` — 程序入口 (185 行)

**职责**：串联所有模块，运行 4 组场景，打印指标对比表，调用绘图。

**文档**：文件头部的 docstring 包含完整的串级 PID 理论推导（传递函数、极点配置、优势分析）。阅读源码即可理解全部数学背景。

**关键代码路径**：
```
导入场景 → 逐一运行 → compute_metrics() → print_metrics_table() → plot_scenario()
```

### `pid_controller.py` — PID 控制器 (70 行)

**导出类**：
| 类 | 行数 | 说明 |
|----|------|------|
| `PIDController` | 1-47 | 通用 PID：微分低通滤波、anti-windup、输出限幅、derivative-on-measurement |
| `CascadePID` | 50-70 | 串级组合：外环→速度参考 EMA 滤波→内环→力矩指令 |

**关键设计选择**：
- 微分项默认 `deriv_on_meas=False`（对误差微分），可切为对测量值微分（避免 setpoint kick）
- Anti-windup 采用条件积分法（输出饱和时冻结积分器）
- `CascadePID.vel_ref_alpha` 实现外环输出 EMA 低通滤波（噪声场景专用）

**依赖**：`numpy`

### `plant_model.py` — 被控对象 (58 行)

**导出类**：
| 类 | 说明 |
|----|------|
| `DCMotorPlant` | 直流电机 + 惯性负载，状态 $[\theta, \omega]$ |

**关键方法**：
| 方法 | 说明 |
|------|------|
| `dynamics(state, u, tau_dist)` | 返回 $[\dot{\theta}, \dot{\omega}]$ |
| `step(u, tau_dist, dt)` | RK4 积分一步 |
| `set_params(J=None, B=None, K=None)` | 在线修改参数（模拟负载变化） |

**动力学方程**：
$$\dot{\theta} = \omega, \quad \dot{\omega} = (K \cdot \text{clamp}(u) - B\omega - \tau_{dist}) / J$$

**依赖**：`numpy`

### `simulation.py` — 仿真引擎 (95 行)

**导出类和函数**：
| 符号 | 说明 |
|------|------|
| `SimulationResult` | 日志容器（time, pos, vel, u, ref, vel_ref） |
| `run_comparison()` | 并行运行单级 vs 串级仿真 |

**`run_comparison()` 参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `plant_single` | DCMotorPlant | — | 单级 PID 控制的被控对象 |
| `plant_cascade` | DCMotorPlant | — | 串级 PID 控制的被控对象 |
| `single_pid` | PIDController | — | 单级控制器 |
| `cascade_pid` | CascadePID | — | 串级控制器 |
| `ref_func` | callable | — | 参考轨迹 $f(t) \to \theta_{ref}$ |
| `dist_func` | callable | None | 扰动函数 $f(t) \to \tau_{dist}$ |
| `noise_func` | callable | None | 噪声函数 $f(t) \to \eta$ |
| `param_change_func` | callable | None | 参数变化 $f(plant, t)$ |
| `t_final` | float | 5.0 | 仿真时长 (s) |
| `sim_dt` | float | 1e-4 | 积分步长 (s) |
| `single_ratio` | int | 100 | 单级 PID 更新间隔（sim_dt × 100 = 0.01s） |
| `cascade_outer_ratio` | int | 100 | 外环更新间隔 |
| `cascade_inner_ratio` | int | 10 | 内环更新间隔（sim_dt × 10 = 0.001s） |

**返回值**：`(SimulationResult, SimulationResult)` — 单级和串级各自的仿真数据。

**依赖**：`numpy`, `pid_controller`, `plant_model`

### `scenarios.py` — 场景定义 (150 行)

**导出**：`SCENARIOS: list[callable]` — 4 个场景函数的列表

**内部函数**：
| 函数 | 说明 |
|------|------|
| `_make_controllers()` | 创建单级 + 串级控制器（含调参） |
| `_make_plants()` | 创建两个相同的被控对象 |
| `scenario_step_response()` | 场景 1：阶跃响应 |
| `scenario_disturbance()` | 场景 2：扰动抑制 |
| `scenario_parameter_change()` | 场景 3：参数鲁棒性 |
| `scenario_noise()` | 场景 4：测量噪声 |

**每个场景函数返回值**：`(res_single, res_cascade, title_string)`

**依赖**：`numpy`, `pid_controller`, `plant_model`, `simulation`

### `plot_results.py` — 可视化与指标 (200 行)

**导出函数**：
| 函数 | 说明 |
|------|------|
| `compute_metrics(time, pos, ref, u, step_t)` | 计算标准性能指标 |
| `compute_disturbance_metrics(time, pos, ref, u, t_dist)` | 扰动抑制专用指标 |
| `compute_noise_metrics(time, pos, ref, u, t_start)` | 噪声灵敏度专用指标 |
| `print_metrics_table(name, m_single, m_cascade)` | 控制台打印对比表 |
| `plot_scenario(res_s, res_c, title, filepath)` | 生成单场景 2×2 对比图 |
| `plot_metrics_summary(all_metrics, filepath)` | 生成四场景指标柱状图 |

**计算的指标**：
| 指标 | 含义 | 越小越好？ |
|------|------|-----------|
| `overshoot_%` | 超调百分比 | ✓ |
| `settling_time_s` | 2% 调节时间 | ✓ |
| `ISE` | 平方误差积分 | ✓ |
| `IAE` | 绝对误差积分 | ✓ |
| `u_rms` | 控制量有效值 | ✓ |
| `max_dev_after_dist` | 扰动后最大位置偏差 | ✓ |
| `iae_after_dist` | 扰动后 IAE | ✓ |
| `pos_std` | 位置标准差（噪声场景） | ✓ |
| `u_std` | 控制量标准差（噪声场景） | ✓ |

**依赖**：`numpy`, `matplotlib`

---

## 数据流图

```
scenarios.py                    main.py                    plot_results.py
    │                              │                              │
    ├─ _make_controllers() ────────┤                              │
    ├─ _make_plants() ────────────┤                              │
    ├─ run_comparison() ──────────┤                              │
    │   │                          │                              │
    │   ├─ PIDController.update()  │                              │
    │   ├─ CascadePID.update_outer()                             │
    │   ├─ CascadePID.update_inner()                             │
    │   ├─ DCMotorPlant.step()     │                              │
    │   └─ SimulationResult        │                              │
    │                              │                              │
    └─ (res_s, res_c, title) ──────┤                              │
                                   │                              │
                                   ├─ compute_metrics() ──────────┤
                                   ├─ compute_disturbance_metrics() ─────┤
                                   ├─ print_metrics_table() ──────┤
                                   ├─ plot_scenario() ────────────┤
                                   └─ plot_metrics_summary() ─────┤
                                                                  │
                                                                  ├─ comparison_S*.png
                                                                  └─ metrics_summary.png
```

---

## 阅读顺序建议

1. **`README.md`** — 先理解理论背景和设计原理
2. **`main.py`** 顶部 docstring — 数学推导详情
3. **`pid_controller.py`** — 理解 `PIDController` 和 `CascadePID` 的工作方式
4. **`plant_model.py`** — 理解被控对象模型
5. **`simulation.py`** — 理解多速率仿真循环
6. **`scenarios.py`** — 理解 4 个场景的设定
7. **`USAGE.md`** — 动手修改参数、添加场景
