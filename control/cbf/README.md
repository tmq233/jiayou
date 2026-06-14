# Control Barrier Functions — 安全临界控制

## 简介

Control Barrier Functions（CBF）提供了一种将安全性编码为控制约束的数学框架。通过在标称控制器输出端叠加一个 Quadratic Program（QP），CBF 保证系统状态永远不会离开安全集 $C = \{x \mid h(x) \ge 0\}$。

## 核心概念

| 概念 | 数学表达 | 说明 |
|------|---------|------|
| 安全集 | $C = \{x \mid h(x) \ge 0\}$ | $h$ 为 barrier function |
| CBF 条件 | $\dot h(x,u) + \alpha h(x) \ge 0$ | 前向不变性的充分条件 |
| CBF-QP | $\min \|u - u_{nom}\|^2$ s.t. CBF | 最小侵入安全滤波器 |
| ECBF | $\psi = \dot h + k h$ | 处理相对度 $r > 1$ |
| CLF-CBF-QP | CLF(松弛) + CBF(硬约束) | 稳定性 + 安全性联合 QP |

## 文件结构

```
cbf/
├── __init__.py    # 模块导出
├── cbf.py         # 核心函数（CBFConstraint、ECBFConstraint、QP 求解器、系统模型）
├── demo.py        # 3 个交互式 demo
└── README.md
```

## 快速开始

```bash
# 运行所有 demo
python control/cbf/demo.py
```

## Demo 说明

### Demo 1: 自适应巡航控制 (ACC) — CBF 安全滤波器
- **场景**：本车希望以 24 m/s 行驶，前车以 20 m/s 行驶
- **CBF**：$h(x) = D - D_{safe} - T \cdot v_{ego} \ge 0$（安全跟车距离）
- **对比**：有 CBF → 始终保持安全距离；无 CBF → 严重碰撞风险
- 输出：距离/速度曲线、$h(x)$ 安全函数、控制输入

### Demo 2: 障碍物避免 — ECBF (高阶相对度)
- **系统**：二维双积分器（点质量），控制量为加速度
- **关键问题**：$h = \|p - p_{obs}\|^2 - R^2$ 的相对度为 2
- **ECBF 方案**：构造 $\psi = \dot h + k h$ 降为相对度 1
- 输出：2D 轨迹图（含障碍物）、$h(x)$ 时间曲线

### Demo 3: CLF-CBF-QP — 稳定性 + 安全性
- **CLF**：$V = \frac{1}{2}(v - v_{des})^2$ 跟踪期望速度（松弛约束）
- **CBF**：$h(x) = D - D_{safe} - T v \ge 0$ 保持安全距离（硬约束）
- **对比**：CLF-only 严重不安全，CLF-CBF-QP 兼顾二者
- 输出：距离/速度、$V(x)$、$h(x)$、松弛变量 $\delta$

## 依赖

- numpy
- scipy
- matplotlib
