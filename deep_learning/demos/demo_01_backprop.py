"""
Demo 1: Forward & Backward Propagation Fundamentals

Visualizes gradient flow through a 4-layer MLP fitting sin(2x).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deep_learning.nn_core import (
    Linear, ReLU, Sequential, MSELoss, SGD, compute_gradient_stats
)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120,
                      'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                      'axes.unicode_minus': False})


def generate_data(N=200):
    np.random.seed(42)
    X = np.linspace(-3, 3, N).reshape(-1, 1)
    y = np.sin(2 * X[:, 0]) + 0.15 * np.random.randn(N)
    return X, y.reshape(-1, 1)


def run():
    print("=" * 60)
    print("  Demo 1: Backpropagation — Gradient Flow Analysis")
    print("=" * 60)
    print()
    print("场景: 用 4 层 MLP [1->32->32->32->1] 拟合 y = sin(2x)")
    print("      400 个训练点, 6000 轮迭代, SGD lr=0.005")
    print()

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

    for epoch in range(6000):
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

    # ── Interpretation ──
    final_loss = loss_history[-1]
    initial_loss = loss_history[0]
    first_grads = grad_history[0][1]
    last_grads = grad_history[-1][1]

    print()
    print("=" * 60)
    print("  结果解读")
    print("=" * 60)
    print(f"""
  输入 x ([-3,3])  →  网络 f_theta  →  预测 y_hat  →  MSE(y_hat, sin(2x))

【1. Loss 曲线 (图1)】
  初始 loss = {initial_loss:.2f}  →  最终 loss = {final_loss:.4f}
  loss 下降说明网络在"学习"：预测值越来越接近真实 sin(2x)。
  - loss 大: 网络预测的曲线偏离真实曲线很远
  - loss 小: 网络预测的曲线贴近真实曲线
  - log 刻度曲线若呈直线下降 → 收敛稳定

【2. 梯度范数 (图2)】
  每条线是一个 Linear 层的梯度大小 ||dL/dW||。
  - 所有层的梯度一起下降 → 网络在学习，各层都在调整
  - 某层梯度为零 → 该层"死了"，链式法则在此断裂
  - 层越深梯度越小 → 梯度消失 (vanishing gradient)，深层学不到东西
  本 demo 中各层梯度范数相近 → 4 层网络梯度流动健康

  第一层初始梯度 = {first_grads['layer_0_W']:.4f}
  最后一层初始梯度 = {first_grads['layer_6_W']:.4f}
  比值 ≈ {first_grads['layer_6_W'] / (first_grads['layer_0_W'] + 1e-10):.2f} (越接近 1.0 越好，说明梯度无衰减)

【3. 拟合效果 (图3)】
  红曲线是网络学到的函数，灰点是训练数据。
  红曲线穿过灰点云的中央 → 学到了 sin(2x) 的规律
  红曲线精确穿过每个灰点 → 过拟合 (记住噪声而非规律)
""")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Backpropagation: Gradient Flow Through a 4-Layer MLP", fontsize=14)

    axes[0].plot(loss_history, 'b-', linewidth=1.0)
    axes[0].set_yscale('log')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss (log scale)")

    epochs_plot = [g[0] for g in grad_history]
    for j, key in enumerate(sorted(grad_history[0][1].keys())):
        norms = [g[1].get(key, 0) for g in grad_history]
        axes[1].plot(epochs_plot, norms, 'o-', markersize=3,
                     linewidth=1.0, label=key.replace('_W', ''))
    axes[1].set_yscale('log')
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Gradient Norm")
    axes[1].set_title("Per-Layer Gradient Norms"); axes[1].legend(fontsize=7)

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
    print(f"[OK] 图片已保存: {out}")


if __name__ == '__main__':
    run()
