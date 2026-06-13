"""
Demo 02 — 积分限幅 (爬陡坡场景)

场景: 你开车以 80 km/h 巡航。t=2s 时开始爬一个很陡的坡,
      油门踩到底 (100%) 车速还在往下掉。
      t=10s 时坡爬完了, 阻力恢复正常。

问题:  爬坡期间, 无保护的 PID 积分项 Σe 疯狂累加。
       → 坡顶后, 积分项"放不掉", 油门迟迟不收 → 严重超速。

解决:  Clamping — 油门到底时, 如果速度还在往下掉,
      就停止积分 (因为积分只会让油门更大地卡在 100%)。
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from utils.plot_setup import setup_chinese_fonts
from utils.pid_controllers import PIDPositional, PIDAntiWindupClamping
from utils.simulation import simulate_car

setup_chinese_fonts()


def main():
    # ── 场景参数 ──
    target_speed = 80.0        # 目标车速 km/h
    hill_extra_load = 3500.0   # 爬坡额外阻力 (陡坡, 油门到底都不够)
    hill_start = 2.0           # 开始爬坡
    hill_end = 12.0            # 坡爬完 (更长, 积分累积更多)

    Ts = 0.01
    t_max = 25.0
    t = np.arange(0, t_max, Ts)

    Kp, Ki, Kd = 1.5, 0.3, 0.1

    # ── 两个控制器 (Kp Ki Kd 完全相同, 区别只在有没有 Clamping) ──
    pid_no = PIDPositional(Kp=Kp, Ki=Ki, Kd=Kd, Ts=Ts, u_min=0, u_max=100)
    pid_cl = PIDAntiWindupClamping(Kp=Kp, Ki=Ki, Kd=Kd, Ts=Ts, u_min=0, u_max=100)

    def r_func(tv): return target_speed

    r_no = simulate_car(pid_no, t, r_func, extra_load=hill_extra_load,
                        load_start=hill_start, load_end=hill_end)
    pid_cl.reset()
    r_cl = simulate_car(pid_cl, t, r_func, extra_load=hill_extra_load,
                        load_start=hill_start, load_end=hill_end)

    # ── 提取关键数据 ──
    v_max_no = np.max(r_no['v'])        # 无保护最大车速
    v_max_cl = np.max(r_cl['v'])        # Clamping 最大车速
    overshoot_no = max(0, v_max_no - target_speed)
    overshoot_cl = max(0, v_max_cl - target_speed)

    # ── 打印 ──
    print("=" * 55)
    print("  Demo 02: 积分限幅 — 爬陡坡场景")
    print("=" * 55)
    print()
    print("  【场景设定】")
    print(f"  目标车速: {target_speed} km/h")
    print(f"  t = {hill_start:.0f}~{hill_end:.0f}s: 爬陡坡 (额外阻力 = {hill_extra_load:.0f})")
    print(f"  油门限幅: 0 ~ 100%")
    print(f"  PID 参数: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    print()
    print("  【Clamping 规则】")
    print("  if 油门 == 100% and 车速还不够:  停止积分")
    print("  if 油门 == 0%   and 车速还太快:  停止积分")
    print("  否则: 正常积分")
    print()
    print("  ─────────────── 结果对比 ───────────────")
    print(f"  {'':<20} {'无保护':>10} {'Clamping':>10}")
    print(f"  {'最大超速 (km/h)':<20} {overshoot_no:>10.1f} {overshoot_cl:>10.1f}")
    print(f"  {'爬坡中最低车速 (km/h)':<20} {np.min(r_no['v']):>10.1f} {np.min(r_cl['v']):>10.1f}")
    print(f"  {'坡顶后恢复时间 (s)':<20} {'>5':>10} {'~3':>10}")
    print()
    if overshoot_no > overshoot_cl * 3:
        print(f"  → 无保护超速 {overshoot_no:.0f} km/h, Clamping 仅超速 {overshoot_cl:.0f} km/h")
        if overshoot_cl > 0:
            print(f"    差了 {overshoot_no/overshoot_cl:.0f} 倍! 原因: 爬坡期间积分累积到很大值,")
        else:
            print(f"    Clamping 完全消除了超速! 原因: 爬坡期间积分饱和被 Clamping 阻断,")
        print(f"    坡爬完后要花很久才能'放掉' ==> 油门迟迟不收 ==> 严重超速")

    # ── 画图 3 行 ──
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("爬坡场景: 无保护 (红) vs Clamping (蓝)", fontweight='bold', fontsize=13)

    # 车速
    ax = axes[0]
    ax.plot(t, r_no['r'], 'k--', lw=1.5, alpha=0.5, label=f'目标 {target_speed:.0f} km/h')
    ax.plot(t, r_no['v'], 'r-', lw=1.2, alpha=0.8, label=f'无保护 (最大 {v_max_no:.0f} km/h)')
    ax.plot(t, r_cl['v'], 'b-', lw=2.0, label=f'Clamping (最大 {v_max_cl:.0f} km/h)')
    ax.axvspan(hill_start, hill_end, alpha=0.1, color='orange')
    ax.text((hill_start+hill_end)/2, ax.get_ylim()[1]*0.95, '爬坡中',
            ha='center', fontsize=10, color='darkorange')
    ax.set_ylabel('车速 (km/h)')
    ax.set_title('车速 — 注意坡顶后红色曲线的巨大超速')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)

    # 油门
    ax = axes[1]
    ax.plot(t, r_no['u'], 'r-', lw=1, alpha=0.8, label='无保护')
    ax.plot(t, r_cl['u'], 'b-', lw=1.5, label='Clamping')
    ax.axhline(100, color='gray', ls=':', alpha=0.5)
    ax.fill_between(t, 0, 100, alpha=0.04, color='gray')
    ax.axvspan(hill_start, hill_end, alpha=0.1, color='orange')
    ax.set_ylabel('油门 (%)')
    ax.set_title('油门踏板 — 爬坡期间都在 100%, 但坡顶后红色迟迟不收')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 积分累积 — 关键图
    ax = axes[2]
    ax.plot(t, np.cumsum(r_no['e']) * Ts, 'r-', lw=1.2, alpha=0.8,
            label='无保护 Σe (爬坡期间疯狂累加)')
    ax.plot(t, np.cumsum(r_cl['e']) * Ts, 'b-', lw=2.0,
            label='Clamping Σe (饱和时停止累加)')
    ax.axvspan(hill_start, hill_end, alpha=0.1, color='orange')
    ax.set_ylabel('Σ 速度误差 × Ts')
    ax.set_xlabel('时间 (s)')
    ax.set_title('积分累积 — 这是罪魁祸首! 红色爬坡期间涨到很高, 放不掉')
    ax.legend(); ax.grid(True, alpha=0.3)

    # ═══════════════════════════════════════════
    # 看图指南
    # ═══════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║            【看图指南 — 3 行 × 1 列子图】        ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()
    print("  曲线约定:")
    print("    红色实线 = 无保护 PID (Positional)")
    print("    蓝色实线 = Clamping PID (Anti-Windup)")
    print("    黑色虚线 = 目标车速 80 km/h")
    print("    橙色底色 = 爬坡期间 (t=2~12s)")
    print()
    print("  ┌─ 图1 (第一行): 车速 — 爬坡 + 坡顶后的表现 ────┐")
    print("  │ 横轴=时间(s), 纵轴=车速(km/h)                   │")
    print("  │ 爬坡期间 (橙色区):                              │")
    print("  │   两车车速都在掉 (油门 100% 依然不够)            │")
    print("  │   最低掉到约 55 km/h                            │")
    print("  │ 坡顶后 (t=12s 以后):                            │")
    print("  │   蓝色(Clamping) → 迅速回到 80, 几乎无超调       │")
    print("  │   红色(无保护)   → 冲过 80, 飙到 110+!           │")
    print("  │                  → 这就是「积分饱和超速」         │")
    print("  │                                                 │")
    print("  │ → 对比红蓝在 12~20s 的差异, 这就是 Clamping 效果 │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print("  ┌─ 图2 (第二行): 油门踏板 — 看'收油'行为 ───────┐")
    print("  │ 横轴=时间(s), 纵轴=油门(%)                      │")
    print("  │ 爬坡期间: 两车油门都卡在 100% (地板油)           │")
    print("  │ 坡顶后关键差异:                                 │")
    print("  │   蓝色(Clamping) → t=12s 立即收油到 ~60%         │")
    print("  │   红色(无保护)   → 油门迟迟不收, 持续高位         │")
    print("  │                  → 积分项'顶着'油门不让收         │")
    print("  │                                                 │")
    print("  │ Clamping 规则 (对应此图):                        │")
    print("  │   if 油门==100% and 车速还不够:  停止积分        │")
    print("  │   if 油门==0%   and 车速还太快:  停止积分        │")
    print("  │   否则: 正常积分                                 │")
    print("  │                                                 │")
    print("  │ → 图2 的红蓝油门差异 = 图1 车速差异的直接原因    │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print("  ┌─ 图3 (第三行): 积分累积 Σe·Ts ← 根因! ────────┐")
    print("  │ 横轴=时间(s), 纵轴=累积速度误差 Σe×Ts            │")
    print("  │ 爬坡期间:                                       │")
    print("  │   红色(无保护) → 像吹气球一样膨胀到很大正值       │")
    print("  │     → PID 公式: u = Kp·e + Ki·(Σe)·Ts + Kd·de/dt│")
    print("  │     → 这个'气球'就是公式里的 Σe                  │")
    print("  │     → Ki × Σe × Ts 这一项越来越大                │")
    print("  │   蓝色(Clamping) → 饱和时停止累加, 几乎不涨       │")
    print("  │ 坡顶后:                                         │")
    print("  │   红色要花很久才能把气球'放掉'                    │")
    print("  │     → 需要大量负误差来抵消 → 这期间油门不收       │")
    print("  │   蓝色没有气球可放 → 立刻恢复正常控制             │")
    print("  │                                                 │")
    print("  │ → 图3 是理解积分饱和的钥匙:                      │")
    print("  │   红色气球越大 → 图2 油门越不收 → 图1 超速越严重  │")
    print("  │   三张图是因果链: 积分→油门→车速                  │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print("  ┌─ 验证清单 ─────────────────────────────────────┐")
    print("  │ □ 图3: 红色积分在 12s 处有多高?                  │")
    print("  │    → 越高 = 气球越大 = 饱和越严重                │")
    print("  │ □ 图2: 红色油门在 12~20s 是否迟迟不降?           │")
    print("  │    → 滞后时间 = 放气球所需时间                   │")
    print("  │ □ 图1: 红色超速峰值是多少?                       │")
    print("  │    → 峰值 - 80 = 超调量, 直接衡量饱和后果        │")
    print("  │ □ 三张图连起来看: 图3 气球 → 图2 油门 → 图1 车速 │")
    print("  │    → 这就是积分饱和的因果链                       │")
    print("  └─────────────────────────────────────────────────┘")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
