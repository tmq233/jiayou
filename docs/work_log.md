# 开发工作日志

## 2026-06-14: Phase 2 & 3 完整实现

### 决策记录

**1. Phase 2: 稳定性分析与非线性控制**
- **Lyapunov 模块 (control/lyapunov/)**
  - 决策：实现四个核心概念 —— 线性 Lyapunov 方程、能量法（摆）、LaSalle 不变集原理（Van der Pol）、CLF+Sontag 通用公式（倒立摆）
  - 理由：从线性到非线性的渐进学习路径，每个概念都有可运行的数值验证
  - 技术选择：纯 numpy+scipy，不使用 casadi 或 cvxpy，保持依赖最小化

- **收缩分析模块 (control/contraction/)**
  - 决策：聚焦于数值 Jacobian + 广义 Jacobian 的网格验证方法
  - 理由：收缩理论在工程界不如 Lyapunov 普及，用数值方法降低入门门槛
  - 示例选择：线性系统、简单非线性（二次项）、FitzHugh-Nagumo 神经元、耦合振荡器同步

- **CBF 模块 (control/cbf/)**
  - 决策：实现了三种 CBF 应用 —— CBF-QP（安全滤波器）、ECBF（高阶相对度）、CLF-CBF-QP（稳定性+安全性联合）
  - 关键技术点：障碍物避免场景需要 ECBF（相对度 2），因为 h 依赖位置而 u 影响加速度
  - QP 求解策略：单约束用闭式投影，多约束 fallback 到 SLSQP

**2. Phase 3: 神经网络与表征学习**
- **架构决策：纯 numpy 实现**
  - 理由：不依赖 PyTorch/TensorFlow，让学习者看清每一行梯度计算
  - 所有层遵循 `forward()/backward()/parameters()/gradients()` 接口协议

- **Dropout 实现使用 inverted dropout**
  - 推理时无需缩放，`model.eval()` 后直接恒等映射
  - Demo 5 明确展示了 train/eval mode 的输出方差差异 —— 控制系统中在部署阶段必须关闭 Dropout

- **残差连接 block 的最后一层零初始化**
  - 理由：$F(x) \approx 0$ 使得初始时 block ≈ identity，训练初期梯度流无衰减
  - 这是 ResNet 原始论文中的关键技巧

### 遇到并解决了的问题

1. **Windows GBK 编码**：中文路径和 Unicode 字符（✓）在 Windows 终端无法正确输出 → 用 `PYTHONIOENCODING=utf-8` 和 `[OK]` 替代
2. **倒立摆 CLF 初始设计错误**：V(x) 设计在 θ=0 而非 θ=π → 修正为误差状态 e = θ-π
3. **CBF 障碍物避免相对度问题**：简单 CBF 对双积分器无效（L_g h = 0）→ 实现 ECBF
4. **深层 ResNet 梯度爆炸**：无归一化时 20 层残差网络输出发散 → 减小宽度(8)、深度(10)、学习率(0.0005)、输出层零初始化
5. **Dropout 模型过拟合 NaN**：128 宽网络在大学习率下发散 → 减小宽度(32)、学习率(0.003)

### 未完成/待推进

- Phase 4: RL 基础（PPO/SAC 从零实现）
- Phase 5: Sim-to-Real 完整 pipeline（domain randomization, system identification）
- 可能的改进：CBF 模块添加基于 casadi 的高效 QP 求解器（当前 SLSQP 在实时场景下偏慢）
