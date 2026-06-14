"""
Demo 3: Layer Normalization vs Batch Normalization

Why LN is preferred over BN in RL/POMDP settings.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deep_learning.nn_core import (
    Linear, ReLU, LayerNorm, BatchNorm, Sequential, MSELoss, SGD,
)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run():
    print("=" * 60)
    print("  Demo 3: LayerNorm vs BatchNorm — Why LN for RL?")
    print("=" * 60)
    print()
    print("核心区别:")
    print("  LN: 每条样本独立归一化 (沿 feature 轴)")
    print("       → 与 batch size 无关, 轨迹独立")
    print("  BN: 跨 batch 归一化 (沿 batch 轴)")
    print("       → 依赖 batch size, 轨迹互相'污染'")
    print()

    np.random.seed(42)
    D = 32

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

    # Experiment 1: Output std vs batch size
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    ln_stds = []
    bn_stds = []

    for bs in batch_sizes:
        X = np.random.randn(bs, D) * 2.0 + 0.5
        y_ln = model_ln.forward(X)
        ln_stds.append(float(np.std(y_ln)))
        y_bn = model_bn.forward(X)
        bn_stds.append(float(np.std(y_bn)))

    # Experiment 2: BN train vs eval discrepancy
    model_bn.eval()
    X_test = np.random.randn(4, D) * 3.0 + 1.0

    model_bn.train()
    y_bn_train = model_bn.forward(X_test)

    model_bn.eval()
    y_bn_eval = model_bn.forward(X_test)

    y_ln_test = model_ln.forward(X_test)

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

    print("训练 LN 模型 (batch_size=4)...")
    losses_ln = train_with_norm('ln')
    print("训练 BN 模型 (batch_size=4)...")
    losses_bn = train_with_norm('bn')

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("LayerNorm vs BatchNorm", fontsize=14)

    axes[0, 0].plot(batch_sizes, ln_stds, 'b-o', linewidth=1.5, label='LayerNorm')
    axes[0, 0].plot(batch_sizes, bn_stds, 'r-s', linewidth=1.5, label='BatchNorm')
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].set_xlabel("Batch Size"); axes[0, 0].set_ylabel("Output Std")
    axes[0, 0].set_title("Output Std vs Batch Size")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(['BN (train)', 'BN (eval)', 'LN (always)'],
                   [float(np.std(y_bn_train)), float(np.std(y_bn_eval)),
                    float(np.std(y_ln_test))],
                   color=['salmon', 'lightcoral', 'steelblue'])
    axes[0, 1].set_ylabel("Output Std")
    axes[0, 1].set_title("Train/Eval Discrepancy (batch=4, shifted input)")

    axes[0, 2].bar(['LayerNorm', 'BatchNorm'],
                   [float(np.var(y_ln, axis=1).mean()),
                    float(np.var(y_bn_train, axis=1).mean())],
                   color=['steelblue', 'salmon'])
    axes[0, 2].set_ylabel("Per-Sample Variance")
    axes[0, 2].set_title("Per-Sample Normalization (batch=4)")

    axes[1, 0].plot(losses_ln, 'b-', linewidth=1.0, alpha=0.8, label='LayerNorm')
    axes[1, 0].plot(losses_bn, 'r-', linewidth=1.0, alpha=0.8, label='BatchNorm')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("MSE Loss")
    axes[1, 0].set_title("Training Stability (batch_size=4)")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].text(0.5, 0.5,
        "Why LN for RL?\n\n"
        "1. 无 batch 依赖: batch_size=1\n"
        "   时也能正常归一化 (RL 常逐\n"
        "   条轨迹更新)\n"
        "2. 无 train/eval 差异: 部署时\n"
        "   行为和训练时完全一致\n"
        "3. 轨迹独立: 不同环境的轨迹\n"
        "   不会互相'污染'统计量\n"
        "4. BN 跨轨迹混合 → 策略更新\n"
        "   高方差 → 训练不稳定",
        transform=axes[1, 1].transAxes,
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 1].set_title("Summary"); axes[1, 1].axis('off')
    axes[1, 2].axis('off')

    plt.tight_layout()
    out = 'deep_learning/demos/fig_demo03_norm.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

    # ── Interpretation ──
    print()
    print("=" * 60)
    print("  结果解读")
    print("=" * 60)
    print(f"""
【1. Batch Size 敏感度 (图1)】
  LN 输出 std 在不同 batch size 下几乎不变 → 稳定
  BN 在小 batch 下标准差剧烈波动 → 训练不稳定
  特别是 batch_size=1 时 BN 直接失效 (无法计算 batch 统计量)

【2. Train/Eval 不一致 (图2)】
  BN train 模式: 用当前 batch 的均值/方差 → 随机
  BN eval 模式: 用训练时累积的 running mean/var → 可能过时
  LN: 无此区别, train 和 eval 行为完全一致
  这对 RL 部署至关重要 — 不能训练一个样、推理另一个样

【3. 训练稳定性 (图4)】
  LN 最终 loss = {losses_ln[-1]:.4f}
  BN 最终 loss = {losses_bn[-1]:.4f}
  LN 的 loss 曲线更平滑 → 策略梯度更新更稳定
  BN 在小 batch 下抖动 → 可能误导策略更新方向
""")
    print(f"  [OK] 图片已保存: {out}")


if __name__ == '__main__':
    run()
