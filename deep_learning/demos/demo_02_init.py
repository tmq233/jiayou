"""
Demo 2: Weight Initialization Strategies

Compares Xavier, He, and Orthogonal initialization on a 10-layer MLP.
Measures per-layer activation variance and gradient norm to diagnose
vanishing/exploding gradients for each strategy.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deep_learning.nn_core import (
    Linear, ReLU, Tanh, Sequential, compute_gradient_stats, MSELoss,
    xavier_init, he_init, orthogonal_init,
)


def build_deep_mlp(init_type, activation='relu', depth=10, width=64):
    act_cls = ReLU if activation == 'relu' else Tanh
    layers = [Linear(1, width, init=init_type), act_cls()]
    for _ in range(depth - 2):
        layers.append(Linear(width, width, init=init_type))
        layers.append(act_cls())
    layers.append(Linear(width, 1, init=init_type))
    return Sequential(*layers)


def run():
    print("=" * 55)
    print("Demo 2: Weight Initialization — Vanishing/Exploding Gradients")
    print("=" * 55)

    np.random.seed(42)
    X = np.random.randn(256, 1).astype(np.float32) * 2.0  # [N, 1]
    y = 0.5 * X + 0.1 * np.random.randn(256, 1)

    depth = 12
    width = 64

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Weight Initialization: {depth}-Layer MLP (width={width})", fontsize=14)

    for col, (init_name, init_label) in enumerate([
        ('xavier', 'Xavier (Glorot)'),
        ('he', 'He (Kaiming)'),
        ('orthogonal', 'Orthogonal'),
    ]):
        for row, act_name in enumerate(['relu', 'tanh']):
            act_cls = ReLU if act_name == 'relu' else Tanh
            ax = axes[row, col]

            # Check activation variance after forward pass (no training)
            model = build_deep_mlp(init_name, act_name, depth, width)
            x_in = X
            layer_variances = [float(np.var(x_in))]

            for layer in model.layers:
                if isinstance(layer, (Linear, act_cls)):
                    x_in = layer.forward(x_in)
                    if isinstance(layer, (ReLU, Tanh)):
                        layer_variances.append(float(np.var(x_in)))

            label = f'{init_label} + {act_name.upper()}'
            ax.plot(range(len(layer_variances)), layer_variances,
                    'o-', markersize=5, linewidth=1.2, label=label)
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4,
                       label='target variance')
            ax.set_xlabel("Layer index")
            ax.set_ylabel("Activation Variance")
            ax.set_title(f"{init_label} + {act_name.upper()}")
            ax.set_yscale('log')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # Gradient stats after 1 backward pass
            if row == 0 and col == 0:
                y_pred = model.forward(X)
                loss_fn = MSELoss()
                loss = loss_fn.forward(y_pred, y)
                model.backward(loss_fn.backward())
                stats = compute_gradient_stats(model, X, loss_fn, y)[0]
                for k, v in stats.items():
                    if 'W' in k:
                        print(f"  {init_name}+{act_name} {k}: "
                              f"grad_norm={v['norm']:.4f}, grad_ratio={v['grad_ratio']:.4f}")

    plt.tight_layout()
    out = 'deep_learning/demos/fig_demo02_init.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {out}")


if __name__ == '__main__':
    run()
