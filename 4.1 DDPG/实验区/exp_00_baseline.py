"""
============================================================
实验 00 —— 基线（Baseline）
============================================================
所有超参数均使用默认值，作为其他实验的对比基准。

超参数一览:
    tau=0.005  gamma=0.99  noise=0.1
    a_lr=1e-3   c_lr=1e-3   batch_size=128
    net_width=400  random_steps=20000

用法: python exp_00_baseline.py
============================================================
"""
from _train_base import run_experiment

run_experiment('exp_00_baseline')
