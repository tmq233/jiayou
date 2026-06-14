"""
Demo 6: Overfitting — When the Network Memorizes Instead of Learns

Scenario:  train on only 15 noisy points with a 4-layer MLP for 6000 epochs.
The network has ~2000 parameters but only 15 data points — it will memorize
every noisy detail instead of capturing the true sin(2x) pattern.

Key indicator:  train_loss ↓↓↓  while  val_loss ↑↑↑  (divergence = overfitting)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deep_learning.nn_core import (
    Linear, ReLU, Sequential, MSELoss, SGD, Dropout,
)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})


def run():
    np.random.seed(42)

    # ── Generate true function + tiny training set ──
    x_all = np.linspace(-3, 3, 500).reshape(-1, 1)
    y_true = np.sin(2 * x_all[:, 0])  # ground truth, no noise

    # Only 15 training points with noise
    idx_train = np.random.choice(500, 15, replace=False)
    idx_val = np.setdiff1d(np.arange(500), idx_train)

    X_train = x_all[idx_train]              # [15, 1]
    y_train = (y_true[idx_train] + 0.15 * np.random.randn(15)).reshape(-1, 1)

    X_val = x_all[idx_val]                  # [485, 1]
    y_val = (y_true[idx_val] + 0.15 * np.random.randn(485)).reshape(-1, 1)

    # ── Over-parameterized model (same as demo 01) ──
    model = Sequential(
        Linear(1, 32, init='he'), ReLU(),
        Linear(32, 32, init='he'), ReLU(),
        Linear(32, 32, init='he'), ReLU(),
        Linear(32, 1, init='he'),
    )

    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)

    train_losses, val_losses = [], []
    snapshots = {}  # epoch -> predicted curve

    for epoch in range(6000):
        # train step
        model.train()
        y_pred = model.forward(X_train)
        loss = loss_fn.forward(y_pred, y_train)
        model.backward(loss_fn.backward())
        optimizer.step(model.gradients())
        train_losses.append(loss)

        # val step (no grad)
        model.eval()
        y_val_pred = model.forward(X_val)
        val_loss = loss_fn.forward(y_val_pred, y_val)
        val_losses.append(val_loss)

        # snapshot at key epochs
        if epoch in [0, 100, 500, 1500, 4000, 5999]:
            y_all_pred = model.forward(x_all)
            snapshots[epoch] = y_all_pred.flatten()

        if epoch % 1000 == 0:
            print(f"  epoch {epoch:4d}:  train_loss={loss:.6f},  "
                  f"val_loss={val_loss:.6f}")

    best_val_epoch = int(np.argmin(val_losses))
    print(f"\n  best val_loss at epoch {best_val_epoch}: "
          f"train={train_losses[best_val_epoch]:.6f}, "
          f"val={val_losses[best_val_epoch]:.6f}")
    print(f"  final gap: train_loss - val_loss = "
          f"{train_losses[-1] - val_losses[-1]:.4f}")

    # snapshot at best epoch for comparison
    model.eval()
    y_best = model.forward(x_all).flatten()

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Overfitting: 15 Training Points, 2000+ Parameters, 6000 Epochs",
                 fontsize=14, fontweight='bold')

    # ---- Row 1: loss curves ----
    axes[0, 0].plot(train_losses, 'b-', linewidth=0.8, alpha=0.8,
                    label='train loss')
    axes[0, 0].plot(val_losses, 'r-', linewidth=1.2, label='val loss')
    axes[0, 0].axvline(x=best_val_epoch, color='green', linestyle='--',
                       alpha=0.7, label=f'best val (epoch {best_val_epoch})')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].set_title("Train vs Val Loss — Spot the Divergence")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

    # ---- Row 1: generalization gap ----
    gap = np.array(train_losses) - np.array(val_losses)
    axes[0, 1].plot(gap, 'purple', linewidth=1.0)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].fill_between(range(len(gap)), 0, gap,
                            where=(np.array(gap) < 0), color='red', alpha=0.2,
                            label='overfit zone')
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Gap")
    axes[0, 1].set_title("Generalization Gap = train_loss - val_loss")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

    # ---- Row 1: parameter norm growth ----
    param_norms = [float(np.linalg.norm(
        model.parameters()['layer_0_W']))]  # placeholder — final snapshot only
    # Show final weight magnitudes
    for name in ['layer_0_W', 'layer_2_W', 'layer_4_W', 'layer_6_W']:
        if name in model.parameters():
            w = model.parameters()[name]
            param_norms.append(float(np.linalg.norm(w)))

    axes[0, 2].bar(['W1', 'W2', 'W3', 'W4'],
                   [float(np.linalg.norm(model.parameters()[k]))
                    for k in ['layer_0_W', 'layer_2_W',
                              'layer_4_W', 'layer_6_W']],
                   color=['steelblue', 'darkseagreen', 'goldenrod', 'salmon'])
    axes[0, 2].set_ylabel("Frobenius Norm")
    axes[0, 2].set_title("Final Weight Magnitudes — Large = Memorized Noise")

    # ---- Row 2: function snapshots ----
    snapshot_epochs = [0, 100, 500, 1500, 4000, 5999]
    colors = ['gray', 'blue', 'cyan', 'green', 'orange', 'red']
    for i, ep in enumerate(snapshot_epochs):
        axes[1, 0].plot(x_all.flatten(), snapshots[ep],
                        color=colors[i], linewidth=1.0 + i * 0.3,
                        alpha=0.8, label=f'epoch {ep}')
    axes[1, 0].scatter(X_train.flatten(), y_train.flatten(), s=30,
                       c='black', zorder=5, label='train data (15 pts)')
    axes[1, 0].plot(x_all.flatten(), y_true, 'k--', linewidth=2.5,
                    alpha=0.4, label='true sin(2x)')
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
    axes[1, 0].set_title("Function Evolution: Smooth → Wiggly (Memorized)")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_ylim(-2.5, 2.5)

    # ---- Row 2: early stop vs final ----
    axes[1, 1].scatter(X_train.flatten(), y_train.flatten(), s=30,
                       c='black', zorder=5, label='train data')
    axes[1, 1].plot(x_all.flatten(), y_best,
                    'g-', linewidth=2.0, label=f'best (epoch {best_val_epoch})')
    axes[1, 1].plot(x_all.flatten(), snapshots[5999],
                    'r-', linewidth=1.5, alpha=0.7, label='final (epoch 5999)')
    axes[1, 1].plot(x_all.flatten(), y_true, 'k--', linewidth=2.5,
                    alpha=0.4, label='true sin(2x)')
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    axes[1, 1].set_title("Early Stopping vs Overtraining")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].set_ylim(-2.5, 2.5)

    # ---- Row 2: summary panel ----
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5,
        "WHY OVERFITTING HAPPENS\n\n"
        "Model capacity:  ~2000 parameters\n"
        "Training data:   15 points\n"
        "Ratio:  ~130 params per data point\n\n"
        "The network has enough freedom to\n"
        "draw a curve that hits every training\n"
        "point EXACTLY — but the curve between\n"
        "points is completely unconstrained.\n\n"
        "FIXES:\n"
        "1. More data (or data augmentation)\n"
        "2. Smaller model (reduce capacity)\n"
        "3. Early stopping (stop at best val)\n"
        "4. Weight decay / L2 regularization\n"
        "5. Dropout (randomly disable neurons)\n"
        "6. Simpler architecture",

        transform=axes[1, 2].transAxes,
        ha='center', va='center', fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    out = 'deep_learning/demos/fig_demo06_overfit.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {out}")


if __name__ == '__main__':
    run()
