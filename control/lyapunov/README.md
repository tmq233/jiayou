# Lyapunov 稳定性分析与 Control Lyapunov Functions

## 简介

非线性系统的稳定性判据工具箱，从线性 Lyapunov 方程出发，逐步深入到非线性系统的 Lie 导数分析法，最终引出 Control Lyapunov Functions（CLF）——一种构造性镇定控制器设计方法。

## 核心概念

| 概念 | 数学表达 | 说明 |
|------|---------|------|
| Lyapunov 方程 | $A^T P + PA = -Q$ | 线性系统的稳定性判据 |
| Lyapunov 函数 | $V(x) = x^T P x$ | 广义能量函数 |
| Lie 导数 | $L_f V = \nabla V \cdot f$ | $V$ 沿向量场 $f$ 的变化率 |
| LaSalle 不变集 | $\{x \mid \dot V(x) = 0\}$ | 渐近稳定性的终极判据 |
| Sontag 通用公式 | $u = -\frac{a + \sqrt{a^2 + \|b\|^4}}{\|b\|^2} b^T$ | CLF → 镇定控制器的闭环公式 |

## 文件结构

```
lyapunov/
├── __init__.py      # 模块导出
├── lyapunov.py      # 核心函数（Lyapunov 方程、CLF、Sontag、数值仿真）
├── demo.py          # 4 个交互式 demo
└── README.md
```

## 快速开始

```bash
# 运行所有 demo
python control/lyapunov/demo.py
```

## Demo 说明

### Demo 1: 线性系统 Lyapunov 稳定性
- 给定 Hurwitz 矩阵 $A$，求解 $A^T P + PA = -I$
- 验证 $P \succ 0$（正定性）和 $\dot V(x) \le 0$
- 输出：$V(x)$ 等高线图、$\dot V(x)$ 热力图、相轨迹

### Demo 2: 摆的能量法 Lyapunov 函数
- 自然能量 $E = \frac{1}{2} m L^2 \dot\theta^2 + mgL(1-\cos\theta)$ 作为 Lyapunov 候选
- 展示次水平集 $\{x \mid E(x) \le c\}$ 作为吸引域估计
- 输出：能量等高线 + 相轨迹、不同次水平集的吸引域

### Demo 3: Van der Pol 振荡器 — LaSalle 不变集
- $\ddot x - \mu(1-x^2)\dot x + x = 0$
- 展示 $\dot V$ 符号不定区域的轨迹收敛到极限环
- 输出：$\dot V(x)$ 符号热力图、多条轨迹收敛到极限环

### Demo 4: CLF + Sontag 公式镇定倒立摆
- 不稳定平衡点 $\theta = \pi$（倒立）
- CLF: $V = \frac{1}{2} k_p (\theta-\pi)^2 + \frac{1}{2} k_d \dot\theta^2$
- Sontag 控制器自动计算镇定力矩
- 输出：状态轨迹、控制输入、$V(x)$ 单调递减验证、相图

## 依赖

- numpy
- scipy
- matplotlib
