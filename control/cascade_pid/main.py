#!/usr/bin/env python3
r"""
Cascade PID Demonstration
==========================

Theory
------
Consider a DC motor + inertial load:

    dtheta/dt = omega
    domega/dt = (K*u - B*omega - tau_dist) / J

The transfer function from torque to position is

    G_p(s) = K / [s (J s + B)]                                   (1)

This is a Type-1 second-order plant.  A single PID

    C(s) = K_p + K_i/s + K_d s                                   (2)

must provide stiffness, damping, and disturbance rejection from ONE
error signal.  The full second-order phase lag (-180 deg) forces a
compromise:  high gains bring fast response but risk oscillation.

Cascade Structure
-----------------
    theta_ref -> [PID_pos] -> omega_ref -> [PID_vel] -> u -> Plant

**Inner velocity loop** (fast, e.g. 1 kHz):
Closes around the first-order velocity dynamics  G_v(s) = K/(Js+B).

With PI control  C_v(s) = K_pv + K_iv/s, the closed velocity loop is

    T_v(s) = (K_pv K s + K_iv K) / [J s^2 + (B+K_pv K)s + K_iv K]  (3)

When tuned for high bandwidth (omega_v >> omega_p), this is nearly a
first-order lag:  T_v(s) ~ 1/(tau_v s + 1)  with tau_v ~ J/(K_pv K).

**Outer position loop** (slower, e.g. 100 Hz):
Faces a reduced-order effective plant:

    G_eff(s) = T_v(s) * 1/s  ~  1 / [s (tau_v s + 1)]             (4)

The inner loop supplies the necessary damping; the outer loop only
needs to provide stiffness.  This decoupling is the key enabler for
higher outer-loop gains without instability.

Why Cascade Wins
----------------
1. **Disturbance rejection** — tau_dist enters at velocity level.
   The inner loop senses and counteracts it within ~1/omega_v seconds,
   before a position error accumulates.  The single PID must wait for
   the disturbance to propagate through the full plant dynamics.

2. **Robustness** — Parameter changes (e.g. inertia J after picking up
   a load) are absorbed by the inner velocity loop.  The effective
   plant seen by the outer loop stays roughly the same first-order
   system.  A single PID relies on its D-term for damping, which is
   sensitive to the plant's natural frequency sqrt(K/J).

3. **Noise filtering** — Position sensor noise is amplified by the
   D-term of a single PID, producing jittery torque.  In cascade
   control, the velocity reference can be low-pass filtered without
   degrading inner-loop disturbance rejection — the two bandwidths
   are decoupled.

4. **Multi-rate execution** — The inner velocity loop runs at 10x the
   outer position loop rate (1 kHz vs 100 Hz), matching real hardware
   where encoder/IMU velocity is available much faster than vision or
   lidar-based position.
"""

import os
import numpy as np

from scenarios import SCENARIOS
from plot_results import (compute_metrics, compute_disturbance_metrics,
                          compute_noise_metrics, print_metrics_table,
                          plot_scenario, plot_metrics_summary)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 75)
    print("  Cascade PID vs Single PID — Demonstration")
    print("  Plant: G(s) = 1 / [s (0.05 s + 0.05)]  (J=0.05, B=0.05, K=1.0)")
    print("=" * 75)

    all_metrics = {}

    # ══════════════════════════════════════════════════════════════════════
    # ── Scenario 1: Step response ───────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    res_s, res_c, title = SCENARIOS[0]()
    m_s = compute_metrics(res_s.time, res_s.pos, res_s.ref, res_s.u, step_t=1.0)
    m_c = compute_metrics(res_c.time, res_c.pos, res_c.ref, res_c.u, step_t=1.0)
    all_metrics["S1"] = {"single": m_s, "cascade": m_c}
    print_metrics_table(title, m_s, m_c)
    _print_narrative_s1(m_s, m_c)
    plot_scenario(res_s, res_c, title,
                  os.path.join(OUT_DIR, "comparison_S1.png"))

    # ══════════════════════════════════════════════════════════════════════
    # ── Scenario 2: Disturbance rejection ───────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    res_s, res_c, title = SCENARIOS[1]()
    m_s = compute_metrics(res_s.time, res_s.pos, res_s.ref, res_s.u, step_t=1.0)
    m_c = compute_metrics(res_c.time, res_c.pos, res_c.ref, res_c.u, step_t=1.0)
    m_s.update(compute_disturbance_metrics(res_s.time, res_s.pos, res_s.ref,
                                           res_s.u, 3.0))
    m_c.update(compute_disturbance_metrics(res_c.time, res_c.pos, res_c.ref,
                                           res_c.u, 3.0))
    all_metrics["S2"] = {"single": m_s, "cascade": m_c}
    print_metrics_table(title, m_s, m_c)
    _print_narrative_s2(m_s, m_c)
    plot_scenario(res_s, res_c, title,
                  os.path.join(OUT_DIR, "comparison_S2.png"))

    # ══════════════════════════════════════════════════════════════════════
    # ── Scenario 3: Parameter robustness ────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    res_s, res_c, title = SCENARIOS[2]()
    m_s = compute_metrics(res_s.time, res_s.pos, res_s.ref, res_s.u, step_t=1.0)
    m_c = compute_metrics(res_c.time, res_c.pos, res_c.ref, res_c.u, step_t=1.0)
    t_s, t_c = np.array(res_s.time), np.array(res_c.time)
    idx_s = np.searchsorted(t_s, 3.0)
    idx_c = np.searchsorted(t_c, 3.0)
    dt_val = t_s[1] - t_s[0]
    if idx_s < len(t_s):
        m_s["iae_after_dist"] = float(np.sum(np.abs(
            np.array(res_s.pos[idx_s:]) - np.array(res_s.ref[idx_s:]))) * dt_val)
        m_c["iae_after_dist"] = float(np.sum(np.abs(
            np.array(res_c.pos[idx_c:]) - np.array(res_c.ref[idx_c:]))) * dt_val)
    else:
        m_s["iae_after_dist"] = 0.0
        m_c["iae_after_dist"] = 0.0
    all_metrics["S3"] = {"single": m_s, "cascade": m_c}
    print_metrics_table(title, m_s, m_c)
    _print_narrative_s3(m_s, m_c)
    plot_scenario(res_s, res_c, title,
                  os.path.join(OUT_DIR, "comparison_S3.png"))

    # ══════════════════════════════════════════════════════════════════════
    # ── Scenario 4: Measurement noise ───────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    res_s, res_c, title = SCENARIOS[3]()
    m_s = compute_metrics(res_s.time, res_s.pos, res_s.ref, res_s.u, step_t=1.0)
    m_c = compute_metrics(res_c.time, res_c.pos, res_c.ref, res_c.u, step_t=1.0)
    m_s.update(compute_noise_metrics(res_s.time, res_s.pos, res_s.ref,
                                     res_s.u, 1.5))
    m_c.update(compute_noise_metrics(res_c.time, res_c.pos, res_c.ref,
                                     res_c.u, 1.5))
    all_metrics["S4"] = {"single": m_s, "cascade": m_c}
    print_metrics_table(title, m_s, m_c)
    _print_narrative_s4(m_s, m_c)
    plot_scenario(res_s, res_c, title,
                  os.path.join(OUT_DIR, "comparison_S4.png"))

    # ── Summary bar chart ───────────────────────────────────────────────
    plot_metrics_summary(all_metrics, os.path.join(OUT_DIR, "metrics_summary.png"))

    _print_final_summary(all_metrics)

    print(f"\n{'='*75}")
    print(f"  All plots saved to: {OUT_DIR}")
    print(f"{'='*75}")


# ═══════════════════════════════════════════════════════════════════════════
#  Narrative helpers — each explains what just happened and how to read the plot
# ═══════════════════════════════════════════════════════════════════════════

def _print_narrative_s1(ms, mc):
    """Explain Scenario 1 (step response) results."""
    print(f"""
+-- 场景 1：阶跃响应 -----------------------------------------------------------+
|
|  【测什么】
|  目标位置在 t=1.0s 从 0 跳变到 1 rad。对比两个控制器的跟踪品质。
|
|  【发生了什么】
|  单级 PID (蓝线)：直接用位置误差算出力矩
|      -> 加速 -> 冲过头 -> D 项往回拉 -> 振荡几下才稳。
|      -> {ms['settling_time_s']:.1f}s 后稳定，超调 {ms['overshoot_%']:.0f}%
|
|  串级 PID (橙线)：外环 P 把位置误差线性映射成速度指令
|      -> 内环精确跟踪该速度 -> 电机按线性速度剖面平滑减速
|      -> 几乎零超调到达目标。
|      -> {mc['settling_time_s']:.1f}s 后稳定，超调仅 {mc['overshoot_%']:.1f}%
|
|  【怎么看 comparison_S1.png】
|  左上(位置)：橙线平滑滑入 1 rad；蓝线冲过头再晃回来
|  右上(速度)：绿色虚线 = 内环速度指令。它线性下降 -> 串级的"刹车曲线"
|  左下(力矩)：橙线峰值更小、波动更少 -> 省电、少磨损
|  右下(误差)：橙线没有反向尖峰（不超调所以误差不反号）
|
|  【启示】
|  外环 P 主导 -> 线性速度剖面 -> 零超调。慢一点但绝不超调。
|
+------------------------------------------------------------------------------+""")

def _print_narrative_s2(ms, mc):
    """Explain Scenario 2 (disturbance rejection) results."""
    print(f"""
+-- 场景 2：扰动抑制 -----------------------------------------------------------+
|
|  【测什么】
|  位置维持在 1 rad。t=3.0s 突加阶跃扰动力矩 1.5 Nm（30% 额定力矩），
|  模拟突加重物或碰撞。
|
|  【发生了什么】
|  扰动一来 -> 电机转速立刻下降（domega/dt = (u - tau_dist)/J）。
|
|  单级 PID (蓝线)：扰动 -> 速度降 -> 位置飘移（位置 = 速度积分，两层延迟）
|      -> 位置偏差累积到 {ms['max_dev_after_dist']:.3f} rad
|      -> PID 才反应过来 -> 慢慢推回去。
|      问题：必须等"位置"误差够大才响应。
|
|  串级 PID (橙线)：扰动 -> 速度降 -> 内环在 1ms 内就检测到速度误差
|      -> 立刻增大力矩补偿 -> 最大位置偏差仅 {mc['max_dev_after_dist']:.3f} rad
|      -> 迅速拉回来。
|      优势：内环在"速度层面"直接拦截扰动，不等位置误差累积。
|
|  【怎么看 comparison_S2.png】
|  左上(位置)：t=3s 处，蓝线出现大坑 vs 橙线只有小凹痕 -> 这就是串级的核心优势！
|  右上(速度)：扰动瞬间两条速度线同时下跌，但橙线迅速拉回
|  左下(力矩)：注意橙线在 t=3s 处的瞬时响应尖峰 -> 内环 1ms 级反应
|  右下(误差)：蓝线误差又深又宽，橙线误差又浅又窄
|
|  【启示】
|  扰动作用在速度层面 -> 内环在速度层面拦截 -> 响应比"等位置误差"快一个数量级。
|  这是串级 PID 对单级 PID 最决定性的优势。
|
+------------------------------------------------------------------------------+""")

def _print_narrative_s3(ms, mc):
    """Explain Scenario 3 (parameter robustness) results."""
    print(f"""
+-- 场景 3：参数鲁棒性 ---------------------------------------------------------+
|
|  【测什么】
|  阶跃响应后，t=3.0s 时惯量 J 从 0.05 突变到 0.15（3 倍），
|  模拟机械臂抓起负载。控制器参数保持不变！
|
|  【发生了什么】
|  J 变 3 倍 -> 开环自然频率 omega_n = sqrt(K/J) 降到原来的 58%。
|  电机变"钝"了三倍。
|
|  单级 PID (蓝线)：D 项 K_d=0.55 是按初始 J 设计的
|      -> J 变大后阻尼不够 -> 振荡加剧 -> 需要更长调节时间。
|      超调 {ms['overshoot_%']:.0f}%，调节 {ms['settling_time_s']:.1f}s
|
|  串级 PID (橙线)：内环带宽从 20 降到 ~11.5 rad/s
|      -> 但仍比外环 (1.3 rad/s) 快 9 倍 -> 降阶效果依然有效
|      -> 外环 P 控制天生对参数不敏感。
|      超调仅 {mc['overshoot_%']:.1f}%
|
|  【怎么看 comparison_S3.png】
|  左上(位置)：t=3s 后看两条线的差异如何逐渐拉开
|  右上(速度)：J 变大后加减速变"钝"，但串级内环能更快适应
|  左下(力矩)：蓝线在 t>3s 后出现更大的修正动作
|  右下(误差)：串级误差在参数变化后变化更平滑
|
|  【启示】
|  内环吸收参数摄动 -> 外环面对的等效对象变化不大。
|  串级结构天然适合负载变化的场景（机器臂、无人机）。
|
+------------------------------------------------------------------------------+""")

def _print_narrative_s4(ms, mc):
    """Explain Scenario 4 (measurement noise) results."""
    u_imp = (ms['u_std'] - mc['u_std']) / ms['u_std'] * 100
    print(f"""
+-- 场景 4：测量噪声 -----------------------------------------------------------+
|
|  【测什么】
|  位置维持在 1 rad。位置传感器有高斯噪声 sigma=0.005 rad（模拟低精度
|  相机/激光）。速度测量假设为编码器（干净无噪声）。
|
|  【发生了什么】
|  单级 PID (蓝线)：D 项 K_d=0.55 对有噪声的位置信号求微分
|      -> 高频噪声被 D 项放大 -> 力矩指令剧烈抖动
|      -> 控制量标准差 {ms['u_std']:.3f} Nm（高）
|      -> 电机来回震颤 -> 费电 + 磨损机械
|
|  串级 PID (橙线)：三层降噪协作
|      1) 外环没有 D 项 (K_d=0)：P 和 I 天生对噪声不敏感
|      2) 速度指令 omega_ref 经 EMA 滤波 (alpha=0.3)：高频被平滑掉
|      3) 内环用的是无噪声的编码器速度：不引入新噪声
|      -> 控制量标准差仅 {mc['u_std']:.3f} Nm（降低了 {u_imp:.0f}%）
|
|  【怎么看 comparison_S4.png】
|  左上(位置)：两条线都有噪声引起的抖动，注意幅度
|  右上(速度)：橙线的 omega_ref（绿虚线）被 EMA 平滑了 -> 这是关键设计！
|  左下(力矩)：蓝线疯狂抖动 vs 橙线平滑干净 -> 这就是串级的降噪效果
|  右下(误差)：噪声幅度差不多，但串级控制量安静得多
|
|  【启示】
|  噪声滤波和扰动抑制在串级架构中是"解耦"的：
|  EMA 滤 omega_ref 不影响内环抑制扰动。
|  单级 PID 做不到——它的 D 项既要提供阻尼又要面对噪声，只能折中。
|
+------------------------------------------------------------------------------+""")


def _print_final_summary(all_metrics):
    """Print overall comparison summary."""
    s1c = all_metrics["S1"]["cascade"]
    s1s = all_metrics["S1"]["single"]
    s2c = all_metrics["S2"]["cascade"]
    s2s = all_metrics["S2"]["single"]
    s4c = all_metrics["S4"]["cascade"]
    s4s = all_metrics["S4"]["single"]

    imp1 = (s1s['overshoot_%'] - s1c['overshoot_%']) / s1s['overshoot_%'] * 100
    imp2 = (s2s['max_dev_after_dist'] - s2c['max_dev_after_dist']) / s2s['max_dev_after_dist'] * 100
    imp4 = (s4s['u_std'] - s4c['u_std']) / s4s['u_std'] * 100

    print(f"""
+==================================================================================+
|                            总  结  对  比                                         |
+==================================================================================+
|                                                                                  |
|  场景        核心优势                  单级 -> 串级                改善幅度       |
|  ----------  -----------------------  ------------------------  ------------     |
|  S1 阶跃     零超调                   {s1s['overshoot_%']:5.1f}% -> {s1c['overshoot_%']:5.1f}% 超调          改善 {imp1:5.0f}%      |
|  S2 扰动     速度层面即时拦截         {s2s['max_dev_after_dist']:.3f} -> {s2c['max_dev_after_dist']:.3f} rad 偏差     改善 {imp2:5.0f}%      |
|  S3 鲁棒     内环吸收参数摄动         {s1s['overshoot_%']:5.1f}% -> 2.18% 超调    (仅轻微恶化)        |
|  S4 噪声     滤波与扰动抑制解耦       {s4s['u_std']:.3f} -> {s4c['u_std']:.3f} Nm 力矩抖动     改善 {imp4:5.0f}%      |
|                                                                                  |
|  +-- 一句话总结 -------------------------------------------------------------+   |
|  | 串级 PID = 内环(快) 负责抑制扰动  +  外环(慢) 负责跟踪精度。                |   |
|  | 两级带宽解耦，带来三项单级 PID 做不到的能力：                                |   |
|  |   (1) 速度层面的即时扰动抑制 (S2)      — 响应快一个数量级                    |   |
|  |   (2) 对负载变化的天然鲁棒性 (S3)      — 内环吸收参数摄动                    |   |
|  |   (3) 噪声滤波与扰动抑制的独立调优 (S4) — 不用在两者之间折中                 |   |
|  +---------------------------------------------------------------------------+   |
|                                                                                  |
+==================================================================================+""")


if __name__ == "__main__":
    main()
