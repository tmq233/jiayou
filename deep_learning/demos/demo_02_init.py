"""
Demo 2: Weight Initialization Strategies

Compares Xavier, He, and Orthogonal initialization on a 12-layer MLP.
Measures per-layer activation variance — the key signal for gradient health.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deep_learning.nn_core import (
    Linear, ReLU, Tanh, Sequential, compute_gradient_stats, MSELoss,
)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 120,
                      'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                      'axes.unicode_minus': False})


def build_deep_mlp(init_type, activation='relu', depth=12, width=64):
    act_cls = ReLU if activation == 'relu' else Tanh
    layers = [Linear(1, width, init=init_type), act_cls()]
    for _ in range(depth - 2):
        layers.append(Linear(width, width, init=init_type))
        layers.append(act_cls())
    layers.append(Linear(width, 1, init=init_type))
    return Sequential(*layers)


def run():
    print("=" * 60)
    print("  Demo 2: Weight Initialization — Vanishing/Exploding Gradients")
    print("=" * 60)
    print()
    print("场景: 12 层 MLP (width=64), 仅做一次 forward pass, 不训练")
    print("      观察激活值方差逐层变化, 判断梯度是否会消失/爆炸")
    print("      目标: 方差应该 ≈ 1.0 (灰线), 方差太小→消失, 太大→爆炸")
    print()

    np.random.seed(42)
    X = np.random.randn(256, 1).astype(np.float32) * 2.0
    y = 0.5 * X + 0.1 * np.random.randn(256, 1)

    depth = 12
    width = 64

    results = {}  # (init, act) -> final_variance

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Weight Initialization: {depth}-Layer MLP (width={width})",
                 fontsize=14)

    for col, (init_name, init_label) in enumerate([
        ('xavier', 'Xavier (Glorot)'),
        ('he', 'He (Kaiming)'),
        ('orthogonal', 'Orthogonal'),
    ]):
        for row, act_name in enumerate(['relu', 'tanh']):
            act_cls = ReLU if act_name == 'relu' else Tanh
            ax = axes[row, col]

            model = build_deep_mlp(init_name, act_name, depth, width)
            x_in = X
            layer_variances = [float(np.var(x_in))]

            for layer in model.layers:
                if isinstance(layer, (Linear, act_cls)):
                    x_in = layer.forward(x_in)
                    if isinstance(layer, (ReLU, Tanh)):
                        layer_variances.append(float(np.var(x_in)))

            results[(init_name, act_name)] = {
                'final_var': layer_variances[-1],
                'variances': layer_variances,
            }

            label = f'{init_label} + {act_name.upper()}'
            ax.plot(range(len(layer_variances)), layer_variances,
                    'o-', markersize=5, linewidth=1.2, label=label)
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4,
                       label='target variance')
            ax.set_xlabel("Layer index"); ax.set_ylabel("Activation Variance")
            ax.set_title(f"{init_label} + {act_name.upper()}")
            ax.set_yscale('log'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = 'deep_learning/demos/fig_demo02_init.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

    # ── Interpretation ──
    print()
    print("=" * 60)
    print("  结果解读")
    print("=" * 60)
    print("""
  图中每条线表示激活值方差随层深的变化。
  灰线 = 1.0, 这是理想目标: 激活方差不衰减也不放大。

  WHY 方差很重要?
    方差 → 0  → 激活值全为 0 → 梯度为 0 → 深层学不到东西 (梯度消失)
    方差 → ∞  → 激活值溢出   → 梯度 NaN → 训练崩溃 (梯度爆炸)
    方差 ≈ 1.0 → 各层信号强度一致 → 正确学习
""")

    # Diagnose each combination
    print("  ┌──────────────────────┬───────────────┬───────────────────────┐")
    print("  │ 初始化 + 激活         │ 最终方差       │ 诊断                  │")
    print("  ├──────────────────────┼───────────────┼───────────────────────┤")
    diagnoses = {}
    for (init, act), data in results.items():
        fv = data['final_var']
        if fv < 0.01:
            diag = "梯度消失! 深层神经元'死'了"
        elif fv > 100:
            diag = "梯度爆炸! 训练会 NaN"
        elif 0.1 < fv < 10:
            diag = "OK, 信号传播良好"
        elif fv < 0.1:
            diag = "偏弱, 深层可能训练缓慢"
        else:
            diag = "偏强, 但有风险"
        diagnoses[(init, act)] = diag
        print(f"  │ {init:8s} + {act.upper():5s}       │ {fv:12.6f}  │ {diag:22s}│")

    print("  └──────────────────────┴───────────────┴───────────────────────┘")
    print()

    # Best pairings
    print("  【关键结论】")
    print("  ReLU 必须配 He 初始化 —— Xavier 对 ReLU 方差衰减 ~50%/层")
    print("    ReLU 把一半输入清零, Xavier 没补偿这个效应, He 补了 (÷√2)")
    print("  Tanh 适合配 Xavier —— 设计时就是针对 Tanh 的方差保持")
    print("  正交初始化在各场景下都较稳定 —— PPO 策略网络的首选")
    print()
    print(f"  [OK] 图片已保存: {out}")


if __name__ == '__main__':
    run()
