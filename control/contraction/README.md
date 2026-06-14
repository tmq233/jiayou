# 收缩分析与增量稳定性

## 简介

收缩理论（Contraction Theory）是比 Lyapunov 稳定性更强的性质：它保证**所有**轨迹指数收敛到**同一条**轨迹，与初始条件无关。这使得收缩分析特别适合观测器设计、同步问题和增量稳定性验证。

## 核心概念

| 概念 | 数学表达 | 说明 |
|------|---------|------|
| 收缩度量 | $M(x) \succ 0$ | 状态相关的 Riemann 度量 |
| 广义 Jacobian | $F = \dot M + MJ + J^T M$ | 收缩的充要条件 |
| 收缩条件 | $F \preceq -2\lambda M$ | $\lambda > 0$ 时指数收缩 |
| 部分收缩 | 仅对子空间收缩 | 观测器、同步问题的数学基础 |

## 文件结构

```
contraction/
├── __init__.py        # 模块导出
├── contraction.py     # 核心函数（Jacobian、广义 Jacobian、收缩度量求
解）
├── demo.py            # 4 个交互式 demo
└── README.md
```

## 快速开始

```bash
# 运行所有 demo
python control/contraction/demo.py
```

## Demo 说明

### Demo 1: 线性收缩度量
- 对 Hurwitz 矩阵 $A$，求解 $A^T M + MA = -Q$
- $M$ 即为收缩度量，$F = A^T M + MA = -Q \prec 0$
- 输出：$M$、$F$ 矩阵、多条轨迹的指数收敛

### Demo 2: 非线性收缩
- 系统 $\dot x_1 = -x_1 + x_2^2,\; \dot x_2 = -2x_2$
- Jacobian $J(x) = \begin{bmatrix}-1 & 2x_2 \\ 0 & -2\end{bmatrix}$
- 对称部分 $J+J^T$ 在带状区域 $\|x_2\| < \sqrt{2}$ 内负定
- 输出：网格收缩性验证、$\|x\|$ 对数衰减

### Demo 3: FitzHugh-Nagumo 神经元
- 时间尺度分离导致收缩：$\tau$ 足够大时系统收缩
- 检查极限环附近的 Jacobian 对称部分最大特征值
- 输出：相图 + 极限环、$\lambda_{\max}(J_{sym})$ 沿轨迹

### Demo 4: 耦合振荡器 — 部分收缩
- $\dot\theta_i = \omega_i + K \sum_j \sin(\theta_j - \theta_i)$
- 相位差 $\Delta\theta = \theta_1 - \theta_2$ 构成收缩子系统
- 临界耦合强度 $K_{crit} = |\omega_1 - \omega_2| / 2$
- 输出：不同 $K$ 下的相位差收敛曲线

## 依赖

- numpy
- scipy
- matplotlib
