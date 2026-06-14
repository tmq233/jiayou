"""
Demo 5: Dropout & Regularization

Dropout as implicit ensemble + the deployment pitfall for control systems.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deep_learning.nn_core import (
    Linear, ReLU, Dropout, Sequential, MSELoss, SGD,
)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run():
    print("=" * 60)
    print("  Demo 5: Dropout — Implicit Ensemble & Deployment Pitfall")
    print("=" * 60)
    print()
    print("核心概念:")
    print("  训练时: 每轮随机丢弃 p% 的神经元 → 每次训练的都是一个子网络")
    print("         → 相当于集成 (ensemble) 了大量子网络 → 减少过拟合")
    print("  推理时: 所有神经元都激活, 不做任何丢弃 (inverted dropout)")
    print("         → 输出是确定性的")
    print()
    print("[!] 控制系统陷阱:")
    print("  如果在部署模型时忘记调用 model.eval() 关闭 Dropout,")
    print("  每次推理都会随机丢弃神经元 → 动作输出含随机噪声")
    print("  → 机器人抖动 / 车辆方向不定 / 机械臂震颤")
    print()

    np.random.seed(42)
    N = 300
    X = np.random.randn(N, 1) * 3.0
    y = np.sin(2 * X[:, 0]) + 0.2 * np.random.randn(N)

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
        return train_losses, val_losses

    idx = np.random.permutation(N)
    X_tr = X[idx[:200], :]
    y_tr = y[idx[:200]].reshape(-1, 1)
    X_val = X[idx[200:], :]
    y_val = y[idx[200:]].reshape(-1, 1)

    model_no_drop = make_model(0.0)
    model_drop = make_model(0.5)

    print("训练中 (WITHOUT dropout)...")
    tr_no, val_no = train_model(model_no_drop, X_tr, y_tr, X_val, y_val)
    print("训练中 (WITH dropout, p=0.5)...")
    tr_drop, val_drop = train_model(model_drop, X_tr, y_tr, X_val, y_val)

    # Monte Carlo
    X_test = np.linspace(-4, 4, 200).reshape(-1, 1)
    n_mc = 50
    preds_train = np.zeros((n_mc, 200))
    preds_eval = np.zeros((n_mc, 200))
    for i in range(n_mc):
        model_drop.train()
        preds_train[i] = model_drop.forward(X_test).flatten()
        model_drop.eval()
        preds_eval[i] = model_drop.forward(X_test).flatten()

    pred_std_train = preds_train.std(axis=0)
    pred_std_eval = preds_eval.std(axis=0)

    # ── Interpretation ──
    gap_no_drop = val_no[-1] - tr_no[-1]
    gap_drop = val_drop[-1] - tr_drop[-1]

    print()
    print("=" * 60)
    print("  结果解读")
    print("=" * 60)
    print(f"""
【1. 过拟合程度对比】
  无 Dropout:  train loss = {tr_no[-1]:.4f},  val loss = {val_no[-1]:.4f}
              → train/val 差距 = {gap_no_drop:.4f} ({"大, 过拟合明显" if abs(gap_no_drop) > 0.01 else "小"})

  有 Dropout:  train loss = {tr_drop[-1]:.4f},  val loss = {val_drop[-1]:.4f}
              → train/val 差距 = {gap_drop:.4f} ({"小, 泛化更好" if abs(gap_drop) < abs(gap_no_drop) else "大"})

  Dropout 让 train 和 val loss 更接近 → 泛化能力提升
  (有 Dropout 时 trian loss 可能更高 — 这是正常的, 因为训练被'故意搞难')

【2. 为什么 Dropout 的 train loss 更高但 val loss 更低?】
  Dropout p=0.5 意味着训练时每次随机杀死 50% 的神经元
  → 网络被迫学习"冗余"的表示 → 任何单个神经元都不能成为关键路径
  → 推理时所有神经元一起工作 → 相当于 2^N 个子网络的投票结果

【3. 部署陷阱 (图3) — 控制系统的关键问题】
  train mode 下输出标准差 (平均): {np.mean(pred_std_train):.4f}
  eval  mode 下输出标准差 (平均): {np.mean(pred_std_eval):.4f}

  如果部署时忘了 model.eval():
  → 每次推理的 50% 神经元被随机丢弃
  → 相同输入得到不同的输出 → 随机噪声动作
  → 机器人: 传感器数据一样但电机指令每次都不同!

  eval mode 下标准差为 0 → 确定性推理 → 安全

  REMEMBER: 上真机前, 检查所有 Dropout 都是 eval 模式!
""")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dropout: Implicit Ensemble & The Deployment Pitfall", fontsize=14)

    axes[0, 0].plot(tr_no, 'b-', linewidth=0.7, alpha=0.5, label='no dropout (train)')
    axes[0, 0].plot(val_no, 'b-', linewidth=1.2, label='no dropout (val)')
    axes[0, 0].plot(tr_drop, 'r-', linewidth=0.7, alpha=0.5, label='dropout p=0.5 (train)')
    axes[0, 0].plot(val_drop, 'r-', linewidth=1.2, label='dropout p=0.5 (val)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_title("Dropout reduces train/val gap")
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(True, alpha=0.3)

    y_plot_no = model_no_drop.forward(X_test).flatten()
    model_drop.eval()
    y_plot_drop = model_drop.forward(X_test).flatten()
    axes[0, 1].scatter(X_tr.flatten(), y_tr.flatten(), s=10, alpha=0.3, label='train')
    axes[0, 1].plot(X_test.flatten(), y_plot_no, 'b-', linewidth=1.5, label='no dropout')
    axes[0, 1].plot(X_test.flatten(), y_plot_drop, 'r-', linewidth=1.5, label='dropout (eval)')
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")
    axes[0, 1].set_title("Predictions"); axes[0, 1].legend(fontsize=8)

    xf = X_test.flatten()
    pred_mean_train = preds_train.mean(axis=0)
    pred_mean_eval = preds_eval.mean(axis=0)
    axes[1, 0].fill_between(xf, pred_mean_train - 2 * pred_std_train,
                            pred_mean_train + 2 * pred_std_train,
                            color='red', alpha=0.3, label='train ±2σ')
    axes[1, 0].fill_between(xf, pred_mean_eval - 2 * pred_std_eval,
                            pred_mean_eval + 2 * pred_std_eval,
                            color='green', alpha=0.5, label='eval ±2σ')
    axes[1, 0].plot(xf, pred_mean_eval, 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y_pred")
    axes[1, 0].set_title("MC variance: train = noise, eval = deterministic")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].text(0.5, 0.5,
        "DEPLOYMENT CHECKLIST\n\n"
        "model.eval() 之前:\n"
        "  - Dropout 随机杀神经元\n"
        "  → action = random noise\n\n"
        "model.eval() 之后:\n"
        "  - Dropout 全部关闭\n"
        "  → action = deterministic\n\n"
        "如果上真机后动作一直抖,\n"
        "先检查: Dropout 关了吗?",
        transform=axes[1, 1].transAxes,
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 1].set_title("Checklist"); axes[1, 1].axis('off')

    plt.tight_layout()
    out = 'deep_learning/demos/fig_demo05_dropout.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 图片已保存: {out}")


if __name__ == '__main__':
    run()
