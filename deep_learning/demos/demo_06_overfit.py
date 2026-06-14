"""
Demo 6: Overfitting — When the Network Memorizes Instead of Learns

15 training points, 2000+ parameters, 6000 epochs.
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


def run():
    print("=" * 60)
    print("  Demo 6: Overfitting — Memorize vs Learn")
    print("=" * 60)
    print()
    print("场景设计:")
    print("  训练数据: 仅 15 个带噪声的点 (来自 sin(2x))")
    print("  模型容量: 4 层 MLP [1->32->32->32->1] ≈ 2200 个参数")
    print("  比例:     2200 参数 / 15 数据点 ≈ 147 参数/数据点")
    print("  迭代:     6000 epoch (严重过度训练)")
    print()
    print("类比: 给你 15 个点让你画一条抛物线, 但你有一支笔可以")
    print("画任意复杂曲线。你完全可以让曲线精确穿过每个点——")
    print("但点之间的曲线可能完全错误。这就是过拟合。")
    print()

    np.random.seed(42)

    x_all = np.linspace(-3, 3, 500).reshape(-1, 1)
    y_true = np.sin(2 * x_all[:, 0])

    idx_train = np.random.choice(500, 15, replace=False)
    idx_val = np.setdiff1d(np.arange(500), idx_train)

    X_train = x_all[idx_train]
    y_train = (y_true[idx_train] + 0.15 * np.random.randn(15)).reshape(-1, 1)
    X_val = x_all[idx_val]
    y_val = (y_true[idx_val] + 0.15 * np.random.randn(485)).reshape(-1, 1)

    model = Sequential(
        Linear(1, 32, init='he'), ReLU(),
        Linear(32, 32, init='he'), ReLU(),
        Linear(32, 32, init='he'), ReLU(),
        Linear(32, 1, init='he'),
    )

    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)

    train_losses, val_losses = [], []
    snapshots = {}

    for epoch in range(6000):
        model.train()
        y_pred = model.forward(X_train)
        loss = loss_fn.forward(y_pred, y_train)
        model.backward(loss_fn.backward())
        optimizer.step(model.gradients())
        train_losses.append(loss)

        model.eval()
        y_val_pred = model.forward(X_val)
        val_loss = loss_fn.forward(y_val_pred, y_val)
        val_losses.append(val_loss)

        if epoch in [0, 100, 500, 1500, 4000, 5999]:
            snapshots[epoch] = model.forward(x_all).flatten()

        if epoch % 1000 == 0:
            print(f"  epoch {epoch:4d}:  train_loss={loss:.6f},  "
                  f"val_loss={val_loss:.6f}")

    best_val_epoch = int(np.argmin(val_losses))
    y_best = model.forward(x_all).flatten()

    # ── Interpretation ──
    train_best = train_losses[best_val_epoch]
    val_best = val_losses[best_val_epoch]
    train_final = train_losses[-1]
    val_final = val_losses[-1]
    gap_final = train_final - val_final

    print()
    print("=" * 60)
    print("  结果解读")
    print("=" * 60)
    print(f"""
【过拟合的经典特征: train loss ↓ 但 val loss 先↓后↑】

  初始 (epoch 0):     train = {train_losses[0]:.4f},  val = {val_losses[0]:.4f}
  最优 val (epoch {best_val_epoch}): train = {train_best:.4f},  val = {val_best:.4f}
  最终 (epoch 5999):  train = {train_final:.4f},  val = {val_final:.4f}

  关键观察:
  - train loss 持续下降 ({train_losses[0]:.2f} → {train_final:.4f})
    → 网络在持续"进步"
  - val loss 在 epoch {best_val_epoch} 达到最低点 {val_best:.4f}
    然后上升到 {val_final:.4f}
    → 但验证集上的效果反而变差了!

  这意味着:
  网络在 epoch {best_val_epoch} 之后的学习全是"背答案"
  — 它在调整参数以精确拟合 15 个训练样本的噪声,
  但这些调整对 485 个验证样本毫无帮助, 甚至有害。

  Generalization gap (最终): {gap_final:.4f}
  (负值 = train 远好于 val → 过拟合)

【图1: Loss 曲线】
  - 蓝线 (train) 一直在下降 → 网络一直在"进步"
  - 红线 (val) 先降后升 → 网络在"背答案"而非"学规律"
  - 竖线 = 最优 early stop 时机 (val loss 最低点)

【图2: 泛化差距】
  差距 = train_loss - val_loss
  差距持续变大 → 训练和验证的表现分歧越来越大

【图4: 函数曲线演变】
  灰色 epoch 0:    大致平滑, 刚开始学习
  → 蓝色 epoch 100:  开始贴近训练点
  → 绿色 epoch 1500: 曲线比较平滑, 这是"学习"的产物
  → 红色 epoch 5999: 疯狂抖动试图穿过每一个训练点
     这就是"背答案"的产物!

【图5: Early Stop vs Overtrain】
  绿色 (best epoch {best_val_epoch}): 合理的曲线, 接近真实 sin(2x)
  红色 (final): 在训练点之间剧烈振荡, 完全偏离真实函数

【修复过拟合的 6 种方法】
  {'>' if best_val_epoch < 5999 else ' '} 1. Early Stopping: 在 epoch {best_val_epoch} 就停
  {'>' if best_val_epoch < 5999 else ' '}    效果: val loss 本可以是 {val_best:.4f} 而非 {val_final:.4f}
  {'>'} 2. 更多数据: 15 个点太少, 增加到 200+ 个
  {'>'} 3. 更小模型: 2200 参数过多, 减少到 ~200
  {'>'} 4. Weight Decay (L2): optimizer 中加 weight_decay=0.001
  {'>'} 5. Dropout: 每层加 Dropout(p=0.5)
  {'>'} 6. 简化架构: 减少层数和宽度
""")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Overfitting: 15 Training Points vs 2200 Parameters",
                 fontsize=14, fontweight='bold')

    axes[0, 0].plot(train_losses, 'b-', linewidth=0.8, alpha=0.8, label='train loss')
    axes[0, 0].plot(val_losses, 'r-', linewidth=1.2, label='val loss')
    axes[0, 0].axvline(x=best_val_epoch, color='green', linestyle='--',
                       alpha=0.7, label=f'best val (epoch {best_val_epoch})')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].set_title("Train vs Val Loss")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

    gap = np.array(train_losses) - np.array(val_losses)
    axes[0, 1].plot(gap, 'purple', linewidth=1.0)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].fill_between(range(len(gap)), 0, gap,
                            where=(np.array(gap) < 0), color='red', alpha=0.2,
                            label='overfit zone')
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Gap")
    axes[0, 1].set_title("Generalization Gap = train - val")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].bar(['W1', 'W2', 'W3', 'W4'],
                   [float(np.linalg.norm(model.parameters()[k]))
                    for k in ['layer_0_W', 'layer_2_W',
                              'layer_4_W', 'layer_6_W']],
                   color=['steelblue', 'darkseagreen', 'goldenrod', 'salmon'])
    axes[0, 2].set_ylabel("Frobenius Norm")
    axes[0, 2].set_title("Final Weight Norms (large = memorized)")

    snapshot_epochs = [0, 100, 500, 1500, 4000, 5999]
    colors_snap = ['gray', 'blue', 'cyan', 'green', 'orange', 'red']
    for i, ep in enumerate(snapshot_epochs):
        axes[1, 0].plot(x_all.flatten(), snapshots[ep],
                        color=colors_snap[i], linewidth=1.0 + i * 0.3,
                        alpha=0.8, label=f'epoch {ep}')
    axes[1, 0].scatter(X_train.flatten(), y_train.flatten(), s=30,
                       c='black', zorder=5, label='train data (15 pts)')
    axes[1, 0].plot(x_all.flatten(), y_true, 'k--', linewidth=2.5,
                    alpha=0.4, label='true sin(2x)')
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
    axes[1, 0].set_title("Function: Smooth → Wiggly")
    axes[1, 0].legend(fontsize=7); axes[1, 0].set_ylim(-2.5, 2.5)

    axes[1, 1].scatter(X_train.flatten(), y_train.flatten(), s=30,
                       c='black', zorder=5, label='train data')
    axes[1, 1].plot(x_all.flatten(), y_best,
                    'g-', linewidth=2.0, label=f'best (epoch {best_val_epoch})')
    axes[1, 1].plot(x_all.flatten(), snapshots[5999],
                    'r-', linewidth=1.5, alpha=0.7, label='final (epoch 5999)')
    axes[1, 1].plot(x_all.flatten(), y_true, 'k--', linewidth=2.5,
                    alpha=0.4, label='true sin(2x)')
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    axes[1, 1].set_title("Early Stop vs Overtrain")
    axes[1, 1].legend(fontsize=7); axes[1, 1].set_ylim(-2.5, 2.5)

    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5,
        "OVERFITTING FIXES\n\n"
        "1. Early stopping\n"
        "   Stop at best val epoch\n\n"
        "2. More data\n"
        "   15 → 200+ points\n\n"
        "3. Smaller model\n"
        "   2200 → 200 params\n\n"
        "4. L2 / weight decay\n"
        "   Penalize large weights\n\n"
        "5. Dropout\n"
        "   Randomly disable neurons\n\n"
        "6. Simpler architecture\n"
        "   Fewer layers/width",
        transform=axes[1, 2].transAxes,
        ha='center', va='center', fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    out = 'deep_learning/demos/fig_demo06_overfit.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 图片已保存: {out}")


if __name__ == '__main__':
    run()
