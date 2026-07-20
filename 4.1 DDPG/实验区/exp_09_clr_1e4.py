"""
============================================================
实验 09 — Critic LR: 1e-3 → 1e-4
============================================================
修改: Critic 学习率从 1e-3 降到 1e-4

预期效果:
  - Critic 更新步伐更小，Q 值估计更缓慢但可能更准确
  - DDPG 中 Critic 通常需要比 Actor 更快学习（因为 Actor 依赖准确的 Q 值）
  - Critic LR 太小 → Q 值估计不准确，Actor 学到错误方向
  - Critic LR 太大 → Q 值震荡

对比: 与基线 exp_00 (c_lr=1e-3) 和 exp_08 (a_lr=1e-4) 比较

用法: python exp_09_clr_1e4.py
============================================================
"""
from _train_base import run_experiment

run_experiment('exp_09_clr_1e4', c_lr=1e-4)
