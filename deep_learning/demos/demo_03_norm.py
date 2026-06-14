"""
Demo 3: Layer Normalization vs Batch Normalization

Demonstrates why LN is preferred over BN in RL/POMDP settings:
  - BN relies on batch statistics → high variance with small/unbalanced batches
  - LN normalizes per-sample → invariant to batch composition
  - For sequential/temporal data, BN mixes independent trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deep_learning.nn_core import (
    Linear, ReLU, LayerNorm, BatchNorm, Sequential, MSELoss, SGD,
)


def run():
    print("=" * 55)
    print("Demo 3: LayerNorm vs BatchNorm — Why LN for RL?")
    print("=" * 55)

    np.random.seed(42)
    D = 32

    # Build two identical MLPs, one with LN, one with BN
    def make_mlp(norm_type='ln'):
        NormCls = LayerNorm if norm_type == 'ln' else BatchNorm
        layers = [
            Linear(D, 64, init='he'), NormCls(64), ReLU(),
            Linear(64, 64, init='he'), NormCls(64), ReLU(),
            Linear(64, 64, init='he'), NormCls(64), ReLU(),
            Linear(64, D, init='he'),
        ]
        return Sequential(*layers)

    model_ln = make_mlp('ln')
    model_bn = make_mlp('bn')

    # Experiment 1: Activation statistics across different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    ln_stds = []
    bn_stds = []

    for bs in batch_sizes:
        X = np.random.randn(bs, D) * 2.0 + 0.5  # shifted, scaled input

        # LN forward
        y_ln = model_ln.forward(X)
        ln_stds.append(float(np.std(y_ln)))

        # BN forward (training mode)
        y_bn = model_bn.forward(X)
        bn_stds.append(float(np.std(y_bn)))

    # Experiment 2: BN train vs eval discrepancy
    model_bn.eval()  # switch to eval mode
    X_test = np.random.randn(4, D) * 3.0 + 1.0  # distribution shift

    model_bn.train()
    y_bn_train = model_bn.forward(X_test)

    model_bn.eval()
    y_bn_eval = model_bn.forward(X_test)

    y_ln_test = model_ln.forward(X_test)  # LN has no train/eval difference

    # Experiment 3: Training stability with small batches
    def train_with_norm(norm_type, bs=4, n_steps=300):
        model = make_mlp(norm_type)
        W_target = np.random.randn(D, D) * 0.5
        loss_fn = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.01)

        losses = []
        for _ in range(n_steps):
            X_batch = np.random.randn(bs, D)
            y_batch = X_batch @ W_target

            y_pred = model.forward(X_batch)
            loss = loss_fn.forward(y_pred, y_batch)
            model.backward(loss_fn.backward())
            optimizer.step(model.gradients())
            losses.append(loss)
        return losses

    print("Training LN model (batch_size=4)...")
    losses_ln = train_with_norm('ln')
    print("Training BN model (batch_size=4)...")
    losses_bn = train_with_norm('bn')
    print(f"  LN final loss: {losses_ln[-1]:.6f}")
    print(f"  BN final loss: {losses_bn[-1]:.6f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("LayerNorm vs BatchNorm: Implications for RL", fontsize=14)

    # 1: Output std vs batch size
    axes[0, 0].plot(batch_sizes, ln_stds, 'b-o', linewidth=1.5, label='LayerNorm')
    axes[0, 0].plot(batch_sizes, bn_stds, 'r-s', linewidth=1.5, label='BatchNorm')
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].set_xlabel("Batch Size"); axes[0, 0].set_ylabel("Output Std")
    axes[0, 0].set_title("Output Std vs Batch Size")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # 2: BN train/eval discrepancy
    axes[0, 1].bar(['BN (train)', 'BN (eval)', 'LN (always)'],
                   [float(np.std(y_bn_train)), float(np.std(y_bn_eval)),
                    float(np.std(y_ln_test))],
                   color=['salmon', 'lightcoral', 'steelblue'])
    axes[0, 1].set_ylabel("Output Std")
    axes[0, 1].set_title("Train/Eval Discrepancy (batch_size=4, shifted input)")

    # 3: Per-sample output variance
    axes[0, 2].bar(['LayerNorm', 'BatchNorm'],
                   [float(np.var(y_ln, axis=1).mean()),
                    float(np.var(y_bn_train, axis=1).mean())],
                   color=['steelblue', 'salmon'])
    axes[0, 2].set_ylabel("Per-Sample Variance")
    axes[0, 2].set_title("Per-Sample Normalization (batch=4)")

    # 4: Training curves
    axes[1, 0].plot(losses_ln, 'b-', linewidth=1.0, alpha=0.8, label='LayerNorm')
    axes[1, 0].plot(losses_bn, 'r-', linewidth=1.0, alpha=0.8, label='BatchNorm')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("MSE Loss")
    axes[1, 0].set_title("Training Stability (batch_size=4)")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # 5: Gradient norms during training
    axes[1, 1].text(0.5, 0.5,
                    "Why LN for RL?\n\n"
                    "- No batch dependency: works with\n"
                    "  batch_size=1 (single trajectory)\n"
                    "- No train/eval gap: same behavior\n"
                    "  at deployment time\n"
                    "- Independent trajectories stay\n"
                    "  independent (no cross-contamination)\n"
                    "- BN mixes statistics from different\n"
                    "  environments → high variance updates",
                    transform=axes[1, 1].transAxes,
                    ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 1].set_title("Summary"); axes[1, 1].axis('off')

    # 6: empty
    axes[1, 2].axis('off')

    plt.tight_layout()
    out = 'deep_learning/demos/fig_demo03_norm.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {out}")


if __name__ == '__main__':
    run()
