"""
Demo 1: Forward & Backward Propagation Fundamentals

Visualizes gradient flow through a multi-layer MLP on a synthetic
regression task, showing the chain rule in action.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deep_learning.nn_core import (
    Linear, ReLU, Sequential, MSELoss, SGD, compute_gradient_stats
)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})


def generate_data(N=200):
    np.random.seed(42)
    X = np.linspace(-3, 3, N).reshape(-1, 1)  # [N, 1]
    y = np.sin(2 * X[:, 0]) + 0.15 * np.random.randn(N)  # [N,]
    return X, y.reshape(-1, 1)


def run():
    print("=" * 55)
    print("Demo 1: Backpropagation — Gradient Flow Analysis")
    print("=" * 55)

    X, y = generate_data(500)
    X_train, y_train = X[:400], y[:400]

    model = Sequential(
        Linear(1, 32, init='he'),
        ReLU(),
        Linear(32, 32, init='he'),
        ReLU(),
        Linear(32, 32, init='he'),
        ReLU(),
        Linear(32, 1, init='he'),
    )

    loss_fn = MSELoss()
    params = model.parameters()
    optimizer = SGD(params, lr=0.005, momentum=0.9)

    loss_history = []
    grad_history = []

    for epoch in range(2000):
        y_pred = model.forward(X_train)
        loss = loss_fn.forward(y_pred, y_train)
        model.backward(loss_fn.backward())
        grads = model.gradients()
        optimizer.step(grads)
        loss_history.append(loss)

        if epoch % 200 == 0:
            stats, _ = compute_gradient_stats(model, X_train, loss_fn, y_train)
            grad_norms = {k: v['norm'] for k, v in stats.items() if 'W' in k}
            grad_history.append((epoch, grad_norms))
            print(f"  epoch {epoch:4d}  loss = {loss:.6f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Backpropagation: Gradient Flow Through a 4-Layer MLP", fontsize=14)

    # Loss
    axes[0].plot(loss_history, 'b-', linewidth=1.0)
    axes[0].set_yscale('log')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss (log scale)")

    # Gradient norms over training
    epochs_plot = [g[0] for g in grad_history]
    for j, key in enumerate(sorted(grad_history[0][1].keys())):
        norms = [g[1].get(key, 0) for g in grad_history]
        axes[1].plot(epochs_plot, norms, 'o-', markersize=3,
                     linewidth=1.0, label=key.replace('_W', ''))
    axes[1].set_yscale('log')
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Gradient Norm")
    axes[1].set_title("Per-Layer Gradient Norms"); axes[1].legend(fontsize=7)

    # Predictions
    X_test = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
    y_test_pred = model.forward(X_test)
    axes[2].scatter(X_train.flatten(), y_train.flatten(), s=8, alpha=0.3,
                    label='train data')
    axes[2].plot(X_test.flatten(), y_test_pred.flatten(), 'r-', linewidth=2,
                 label='MLP prediction')
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    axes[2].set_title("Learned Function"); axes[2].legend()

    plt.tight_layout()
    out = 'deep_learning/demos/fig_demo01_backprop.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {out}")


if __name__ == '__main__':
    run()
