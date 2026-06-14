"""
Demo 4: Residual Connections — The Gradient Highway

dH/dx = dF/dx + I

The identity mapping I guarantees that even when dF/dx → 0
(vanishing gradients in deep networks), the gradient can still
propagate through the skip connection.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deep_learning.nn_core import (
    Linear, ReLU, Sequential, MSELoss, SGD, compute_gradient_stats,
)


class ResidualBlock:
    """F(x) + x  where F is a 2-layer subnetwork.

    The output layer (fc2) is initialized to near-zero so that
    initially F(x) ≈ 0, making each block behave as identity.
    """

    def __init__(self, dim, init='he'):
        self.fc1 = Linear(dim, dim, init=init)
        self.act = ReLU()
        self.fc2 = Linear(dim, dim, init=init)
        # Zero-init the output of residual → block ≈ identity
        self.fc2.W *= 0.01
        self.fc2.b *= 0.0

    def forward(self, x):
        self._cache = {'x': x}
        residual = self.fc2.forward(self.act.forward(self.fc1.forward(x)))
        return residual + x  # [N, dim]

    def backward(self, dout):
        x = self._cache['x']
        # Gradient through skip connection: dout * I = dout
        dresidual = self.fc2.backward(dout)
        dresidual = self.act.backward(dresidual)
        dresidual = self.fc1.backward(dresidual)
        return dresidual + dout  # Skip connection gradient + residual gradient

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
    """Stack of ResidualBlocks."""

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
    """Plain MLP with same depth and width (no skip connections)."""

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
    print("=" * 55)
    print("Demo 4: Residual Connections — The Gradient Highway")
    print("=" * 55)

    np.random.seed(42)
    D = 8
    depth = 10

    # Same initialization for fair comparison
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

    for step in range(500):
        # ResNet
        y_pred = resnet.forward(X)
        loss = loss_fn.forward(y_pred, y)
        resnet.backward(loss_fn.backward())
        grads = resnet.gradients()
        optimizer_res.step(grads)
        losses_res.append(loss)

        # Plain MLP
        y_pred_p = plain_mlp.forward(X)
        loss_p = loss_fn.forward(y_pred_p, y)
        plain_mlp.backward(loss_fn.backward())
        grads_p = plain_mlp.gradients()
        optimizer_plain.step(grads_p)
        losses_plain.append(loss_p)

        if step % 100 == 0:
            # Measure grad ratios for first block/layer
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

            print(f"  step {step:3d}: "
                  f"res_loss={loss:.6f}, plain_loss={loss_p:.6f}, "
                  f"res_grad_last/first={ratio_res:.4f}, "
                  f"plain_grad_last/first={ratio_plain:.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Residual Connections: $\\partial H/\\partial x = \\partial F/\\partial x + I$",
                 fontsize=14)

    axes[0, 0].plot(losses_res, 'b-', linewidth=1.0, label='ResNet (with skip)')
    axes[0, 0].plot(losses_plain, 'r-', linewidth=1.0, alpha=0.7,
                    label='Plain MLP (no skip)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].set_title(f"Training ({depth}-block deep network)")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    steps_plot = np.arange(0, 500, 100)
    axes[0, 1].plot(steps_plot, grad_ratios_res, 'b-o', linewidth=1.5,
                    label='ResNet')
    axes[0, 1].plot(steps_plot, grad_ratios_plain, 'r-s', linewidth=1.5,
                    label='Plain MLP')
    axes[0, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Gradient Ratio (last / first)")
    axes[0, 1].set_title("Gradient Preservation: Last-to-First Layer")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Gradient norm comparison across layers (at final step)
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
    axes[1, 0].set_title("Per-Layer Gradient Norms (final step)")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Theory illustration
    axes[1, 1].text(0.5, 0.5,
                    "Residual Connection: $y = \\mathcal{F}(x) + x$\n\n"
                    "$\\frac{\\partial y}{\\partial x} = "
                    "\\frac{\\partial \\mathcal{F}}{\\partial x} + I$\n\n"
                    "Key insight:\n"
                    "- The identity $I$ provides a gradient highway\n"
                    "- Even when $\\partial\\mathcal{F}/\\partial x \\to 0$\n"
                    "  (vanishing gradients in deep nets),\n"
                    "  the gradient $dL/dx$ still has $dL/dy \\cdot I$\n\n"
                    "This is why ResNets can be 100+ layers\n"
                    "while plain nets fail beyond ~20 layers.",
                    transform=axes[1, 1].transAxes,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 1].set_title("Theory: Why Skip Connections Work")
    axes[1, 1].axis('off')

    plt.tight_layout()
    out = 'deep_learning/demos/fig_demo04_residual.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {out}")


if __name__ == '__main__':
    run()
