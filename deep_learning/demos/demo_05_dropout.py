"""
Demo 5: Dropout & Regularization

Dropout as implicit ensemble:
  - Training: randomly drop neurons → each forward pass is a different sub-network
  - Inference: all neurons active, no scaling needed (inverted dropout)

Control-system pitfall: if Dropout is left ON during deployment,
action outputs become non-deterministic — a critical safety issue.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deep_learning.nn_core import (
    Linear, ReLU, Dropout, Sequential, MSELoss, SGD,
)


def run():
    print("=" * 55)
    print("Demo 5: Dropout — Implicit Ensemble & Deployment Pitfalls")
    print("=" * 55)

    np.random.seed(42)
    N = 300
    X = np.random.randn(N, 1) * 3.0
    y = np.sin(2 * X[:, 0]) + 0.2 * np.random.randn(N)

    # Simpler model with LR difference to show dropout benefit
    def make_model(dropout_p=0.0):
        layers = [
            Linear(1, 32, init='he'), ReLU(),
            Linear(32, 32, init='he'), ReLU(),
            Linear(32, 32, init='he'), ReLU(),
            Linear(32, 1, init='he'),
        ]
        if dropout_p > 0:
            layers.insert(3, Dropout(dropout_p))
            layers.insert(6, Dropout(dropout_p))
        return Sequential(*layers)

    # Experiment 1: Overfitting comparison
    def train_model(model, X_tr, y_tr, X_val, y_val, steps=2000):
        loss_fn = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.003, momentum=0.9)
        train_losses, val_losses = [], []

        for step in range(steps):
            y_pred = model.forward(X_tr)
            loss = loss_fn.forward(y_pred, y_tr)
            model.backward(loss_fn.backward())
            optimizer.step(model.gradients())
            train_losses.append(loss)

            model.eval()
            y_val_pred = model.forward(X_val)
            val_loss = loss_fn.forward(y_val_pred, y_val)
            model.train()
            val_losses.append(val_loss)

            if step % 500 == 0:
                print(f"  step {step:4d}: train_loss={loss:.6f}, val_loss={val_loss:.6f}")
        return train_losses, val_losses

    # Train-test split
    idx = np.random.permutation(N)
    X_tr = X[idx[:200], :]
    y_tr = y[idx[:200]].reshape(-1, 1)
    X_val = X[idx[200:], :]
    y_val = y[idx[200:]].reshape(-1, 1)

    model_no_drop = make_model(0.0)
    model_drop = make_model(0.5)

    print("Training WITHOUT dropout...")
    tr_no, val_no = train_model(model_no_drop, X_tr, y_tr, X_val, y_val)

    print("Training WITH dropout (p=0.5)...")
    tr_drop, val_drop = train_model(model_drop, X_tr, y_tr, X_val, y_val)

    print(f"  Final: no_dropout val={val_no[-1]:.6f}, dropout val={val_drop[-1]:.6f}")

    # Experiment 2: Dropout at deployment (THE PITFALL)
    X_test = np.linspace(-4, 4, 200).reshape(-1, 1)

    # Monte Carlo: 50 forward passes to show variance
    n_mc = 50
    preds_train = np.zeros((n_mc, 200))
    preds_eval = np.zeros((n_mc, 200))

    for i in range(n_mc):
        model_drop.train()
        preds_train[i] = model_drop.forward(X_test).flatten()
        model_drop.eval()
        preds_eval[i] = model_drop.forward(X_test).flatten()

    pred_mean_train = preds_train.mean(axis=0)
    pred_std_train = preds_train.std(axis=0)
    pred_mean_eval = preds_eval.mean(axis=0)
    pred_std_eval = preds_eval.std(axis=0)

    # Experiment 3: Ensemble effect
    print(f"  Dropout train-mode std (avg): {np.mean(pred_std_train):.6f}")
    print(f"  Dropout eval-mode  std (avg): {np.mean(pred_std_eval):.6f}")
    print(f"  Deployment noise reduction: {1 - np.mean(pred_std_eval)/max(np.mean(pred_std_train),1e-10):.1%}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dropout: Implicit Ensemble & The Deployment Pitfall", fontsize=14)

    # Loss curves
    axes[0, 0].plot(tr_no, 'b-', linewidth=0.7, alpha=0.5, label='no dropout (train)')
    axes[0, 0].plot(val_no, 'b-', linewidth=1.2, label='no dropout (val)')
    axes[0, 0].plot(tr_drop, 'r-', linewidth=0.7, alpha=0.5,
                    label='dropout p=0.5 (train)')
    axes[0, 0].plot(val_drop, 'r-', linewidth=1.2, label='dropout p=0.5 (val)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_title("Overfitting: Dropout reduces train/val gap")
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(True, alpha=0.3)

    # Predictions
    axes[0, 1].scatter(X_tr.flatten(), y_tr, s=10, alpha=0.3, label='train')
    y_plot_no = model_no_drop.forward(X_test).flatten()
    model_drop.eval()
    y_plot_drop = model_drop.forward(X_test).flatten()
    axes[0, 1].plot(X_test.flatten(), y_plot_no, 'b-', linewidth=1.5,
                    label='no dropout')
    axes[0, 1].plot(X_test.flatten(), y_plot_drop, 'r-', linewidth=1.5,
                    label='dropout (eval)')
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")
    axes[0, 1].set_title("Predictions"); axes[0, 1].legend(fontsize=8)

    # MC variance: train vs eval mode
    xf = X_test.flatten()
    axes[1, 0].fill_between(xf,
                            pred_mean_train - 2 * pred_std_train,
                            pred_mean_train + 2 * pred_std_train,
                            color='red', alpha=0.3, label='train ±2σ')
    axes[1, 0].fill_between(xf,
                            pred_mean_eval - 2 * pred_std_eval,
                            pred_mean_eval + 2 * pred_std_eval,
                            color='green', alpha=0.5, label='eval ±2σ')
    axes[1, 0].plot(xf, pred_mean_eval, 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y_pred")
    axes[1, 0].set_title("Deployment Variance: train mode = RANDOM NOISE")
    axes[1, 0].legend(fontsize=8)

    # Summary
    axes[1, 1].text(0.5, 0.5,
                    "CONTROL SYSTEMS WARNING\n\n"
                    "Dropout at TRAINING time:\n"
                    "  - Drops neurons with probability p\n"
                    "  - Each forward pass is a different\n"
                    "    sub-network (implicit ensemble)\n"
                    "  - Reduces overfitting\n\n"
                    "Dropout at DEPLOYMENT time:\n"
                    "  - MUST call model.eval() to disable\n"
                    "  - If left ON: actions contain random\n"
                    "    noise → unpredictable behavior\n"
                    "  - If you see action jitter in a\n"
                    "    deployed RL policy, check whether\n"
                    "    Dropout layers are in eval mode",
                    transform=axes[1, 1].transAxes,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 1].set_title("The Pitfall")
    axes[1, 1].axis('off')

    plt.tight_layout()
    out = 'deep_learning/demos/fig_demo05_dropout.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {out}")


if __name__ == '__main__':
    run()
