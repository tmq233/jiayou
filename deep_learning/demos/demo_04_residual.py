"""
Demo 4: Residual Connections — The Gradient Highway

dH/dx = dF/dx + I

The identity I guarantees gradients can bypass vanishing layers.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deep_learning.nn_core import (
    Linear, ReLU, Sequential, MSELoss, SGD,
)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ResidualBlock:
    """F(x) + x.  Output layer near-zero init → block ≈ identity."""

    def __init__(self, dim, init='he'):
        self.fc1 = Linear(dim, dim, init=init)
        self.act = ReLU()
        self.fc2 = Linear(dim, dim, init=init)
        self.fc2.W *= 0.01
        self.fc2.b *= 0.0

    def forward(self, x):
        self._cache = {'x': x}
        residual = self.fc2.forward(self.act.forward(self.fc1.forward(x)))
        return residual + x

    def backward(self, dout):
        x = self._cache['x']
        dresidual = self.fc2.backward(dout)
        dresidual = self.act.backward(dresidual)
        dresidual = self.fc1.backward(dresidual)
        return dresidual + dout  # <-- THE KEY: skip gradient = dout * I

    def parameters(self):
        p = {}
        for name in ['fc1', 'fc2']:
            for k, v in getattr(self, name).parameters().items():
                p[f'{name}_{k}'] = v
        return p

    def gradients(self):
        g = {}
        for name in ['fc1', 'fc2']:
            for k, v in getattr(self, name).gradients().items():
                g[f'{name}_{k}'] = v
        return g


class DeepResNet:
    def __init__(self, dim, depth, init='he'):
        self.blocks = [ResidualBlock(dim, init) for _ in range(depth)]

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

    def backward(self, dout):
        for block in reversed(self.blocks):
            dout = block.backward(dout)
        return dout

    def parameters(self):
        p = {}
        for i, block in enumerate(self.blocks):
            for k, v in block.parameters().items():
                p[f'block{i}_{k}'] = v
        return p

    def gradients(self):
        g = {}
        for i, block in enumerate(self.blocks):
            for k, v in block.gradients().items():
                g[f'block{i}_{k}'] = v
        return g


class DeepMLPNoSkip:
    def __init__(self, dim, depth, init='he'):
        layers = []
        for _ in range(depth):
            layers.append(Linear(dim, dim, init=init))
            layers.append(ReLU())
        layers.append(Linear(dim, dim, init=init))
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model.forward(x)

    def backward(self, dout):
        return self.model.backward(dout)

    def parameters(self):
        return self.model.parameters()

    def gradients(self):
        return self.model.gradients()


def run():
    print("=" * 60)
    print("  Demo 4: Residual Connections — The Gradient Highway")
    print("=" * 60)
    print()
    print("核心公式: dH/dx = dF/dx + I")
    print("  F(x): 残差子网络的输出")
    print("  I:   单位矩阵 (恒等映射的梯度)")
    print("  即使 dF/dx → 0, 梯度仍通过 I 无损回传")
    print()
    print("场景: 10 个残差 block 的 ResNet vs 同深度的 plain MLP")
    print("      观察最后一层和第一层的梯度比值")
    print()

    np.random.seed(42)
    D = 8
    depth = 10

    np.random.seed(42)
    resnet = DeepResNet(D, depth)
    np.random.seed(42)
    plain_mlp = DeepMLPNoSkip(D, depth)

    X = np.random.randn(128, D) * 0.1
    y = np.random.randn(128, D) * 0.1

    loss_fn = MSELoss()
    optimizer_res = SGD(resnet.parameters(), lr=0.0005)
    optimizer_plain = SGD(plain_mlp.parameters(), lr=0.0005)

    losses_res, losses_plain = [], []
    grad_ratios_res, grad_ratios_plain = [], []
    steps_recorded = []

    for step in range(500):
        y_pred = resnet.forward(X)
        loss = loss_fn.forward(y_pred, y)
        resnet.backward(loss_fn.backward())
        grads = resnet.gradients()
        optimizer_res.step(grads)
        losses_res.append(loss)

        y_pred_p = plain_mlp.forward(X)
        loss_p = loss_fn.forward(y_pred_p, y)
        plain_mlp.backward(loss_fn.backward())
        grads_p = plain_mlp.gradients()
        optimizer_plain.step(grads_p)
        losses_plain.append(loss_p)

        if step % 100 == 0:
            g_res_first = [v for k, v in grads.items()
                           if 'block0' in k and 'W' in k]
            g_plain_first = [v for k, v in grads_p.items()
                             if 'layer_0' in k and 'W' in k]
            g_res_last = [v for k, v in grads.items()
                          if f'block{depth-1}' in k and 'W' in k]
            g_plain_last = [v for k, v in grads_p.items()
                            if f'layer_{2*depth}' in k and 'W' in k]

            ratio_res = (float(np.linalg.norm(g_res_last[0])) /
                         max(float(np.linalg.norm(g_res_first[0])), 1e-10))
            ratio_plain = (float(np.linalg.norm(g_plain_last[0])) /
                           max(float(np.linalg.norm(g_plain_first[0])), 1e-10))

            grad_ratios_res.append(ratio_res)
            grad_ratios_plain.append(ratio_plain)
            steps_recorded.append(step)

    # ── Interpretation ──
    grad_ratio_res_avg = np.mean(grad_ratios_res)
    grad_ratio_plain_avg = np.mean(grad_ratios_plain)

    print()
    print("=" * 60)
    print("  结果解读")
    print("=" * 60)
    print(f"""
  ResNet blocks = {depth}, hidden dim = {D}

【核心指标: 最后一层 / 第一层的梯度比值】
  理想值 = 1.0 (梯度无衰减)
  ResNet 平均比值:      {grad_ratio_res_avg:.3f}
  Plain MLP 平均比值:   {grad_ratio_plain_avg:.3f}

  解释:
  - ResNet 的比值接近 1.0 → 深层和浅层收到的梯度尺度一致
    → 所有 block 都在学习, 网络可以堆到 100+ 层
  - Plain MLP 的比值偏离 1.0 → 深层收到的梯度远小于浅层
    → 深层几乎学不到东西, 超过 ~20 层就退化

【Loss 曲线】
  ResNet 最终 loss:  {losses_res[-1]:.6f}
  Plain MLP 最终 loss: {losses_plain[-1]:.6f}
  (ResNet 收敛更快)

【为什么残差能工作?】
  反向传播时 dL/dx = dL/dH * (dF/dx + I)
                     = dL/dH * dF/dx + dL/dH    ← 第二项永远是 dL/dH
  即使中间 10 个 block 的 dF/dx 都很小,
  下游的梯度 dL/dH 仍可以通过 +I 原封不动传到上游
""")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Residual Connections: $\\partial H/\\partial x = \\partial F/\\partial x + I$",
                 fontsize=14)

    axes[0, 0].plot(losses_res, 'b-', linewidth=1.0, label='ResNet')
    axes[0, 0].plot(losses_plain, 'r-', linewidth=1.0, alpha=0.7, label='Plain MLP')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].set_title(f"Training ({depth}-block network)")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps_recorded, grad_ratios_res, 'b-o', linewidth=1.5, label='ResNet')
    axes[0, 1].plot(steps_recorded, grad_ratios_plain, 'r-s', linewidth=1.5, label='Plain MLP')
    axes[0, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Gradient Ratio (last / first)")
    axes[0, 1].set_title("Gradient Preservation")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Per-layer gradient norms
    grad_per_layer_res = []
    grad_per_layer_plain = []
    for i in range(depth):
        key = f'block{i}_fc1_W'
        if key in grads:
            grad_per_layer_res.append(float(np.linalg.norm(grads[key])))
    for i in range(2 * depth + 1):
        key = f'layer_{i}_W'
        if key in grads_p:
            grad_per_layer_plain.append(float(np.linalg.norm(grads_p[key])))

    axes[1, 0].plot(grad_per_layer_res, 'b-o', markersize=5, label='ResNet')
    axes[1, 0].plot(grad_per_layer_plain, 'r-s', markersize=5, label='Plain MLP')
    axes[1, 0].set_xlabel("Layer/Block Index"); axes[1, 0].set_ylabel("Gradient Norm")
    axes[1, 0].set_title("Per-Layer Gradient Norms")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].text(0.5, 0.5,
        "Residual:  y = F(x) + x\n\n"
        "dL/dx = dL/dy * dF/dx + dL/dy\n"
        "         --------/   ----/\n"
        "         可能很小    永远存在!\n\n"
        "这就是为什么 ResNet\n"
        "可以 100+ 层, 而 plain\n"
        "网络超过 20 层就退化",
        transform=axes[1, 1].transAxes,
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 1].set_title("Gradient Highway Principle"); axes[1, 1].axis('off')

    plt.tight_layout()
    out = 'deep_learning/demos/fig_demo04_residual.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 图片已保存: {out}")


if __name__ == '__main__':
    run()
