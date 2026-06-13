"""
Demo 01 — 位置式 vs 增量式 PID (车辆定速巡航, 三种工况对比)

场景: 你开一辆车上高速, 打开定速巡航。车载电脑通过控制油门来维持目标车速。
      本 Demo 用两种 PID 算法实现巡航, 在三种典型工况下对比表现。

位置式 PID:  油门 = Kp*当前误差 + Ki*历史误差累加 + Kd*误差变化率
             → 积分项会"记住"所有过去的误差

增量式 PID:  油门变化量 = Kp*误差变化 + Ki*当前误差 + Kd*误差加速度
             → 只算"油门该调多少", 不累加历史
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from utils.plot_setup import setup_chinese_fonts
from utils.pid_controllers import PIDPositional, PIDIncremental
from utils.simulation import simulate_car, cruise_metrics

setup_chinese_fonts()


# ── 控制器参数 (两个控制器用相同的 Kp, Ki, Kd) ──
Kp, Ki, Kd = 1.5, 0.5, 0.1
Ts = 0.01  # 控制器每 0.01 秒算一次


def run_scenario(title, t_max, r_func, desc, show_metrics=True):
    """运行一个工况, 打印指标, 画图."""
    t = np.arange(0, t_max, Ts)

    pid_pos = PIDPositional(Kp=Kp, Ki=Ki, Kd=Kd, Ts=Ts, u_min=0, u_max=100)
    pid_inc = PIDIncremental(Kp=Kp, Ki=Ki, Kd=Kd, Ts=Ts, u_min=0, u_max=100)

    r_pos = simulate_car(pid_pos, t, r_func)
    r_inc = simulate_car(pid_inc, t, r_func)

    if show_metrics:
        target = r_func(t_max * 0.9)  # 取末段目标值
        m_pos = cruise_metrics(t, r_pos['v'], target)
        m_inc = cruise_metrics(t, r_inc['v'], target)

        print(f"\n{'─'*50}")
        print(f"  【{title}】")
        print(f"  目标车速: {target} km/h | 油门限幅: 0~100%")
        print(f"  {'指标':<16} {'位置式':>10} {'增量式':>10}")
        print(f"  {'超调量 (km/h)':<16} {m_pos['overshoot_kmh']:>10} {m_inc['overshoot_kmh']:>10}")
        print(f"  {'上升时间 (s)':<16} {m_pos['rise_time_s']:>10} {m_inc['rise_time_s']:>10}")
        print(f"  {'调节时间 (s)':<16} {m_pos['settling_time_s']:>10} {m_inc['settling_time_s']:>10}")
        print(f"  {'稳态误差 (km/h)':<16} {m_pos['steady_error_kmh']:>10} {m_inc['steady_error_kmh']:>10}")
        print(f"  {'最终车速 (km/h)':<16} {m_pos['v_final']:>10} {m_inc['v_final']:>10}")
    else:
        m_pos = m_inc = None

    return t, r_pos, r_inc, m_pos, m_inc


def main():
    print("=" * 55)
    print("  Demo 01: 位置式 vs 增量式 PID — 车辆定速巡航")
    print("=" * 55)
    print()
    print("  车辆参数:")
    print("    发动机响应滞后: 0.3s")
    print("    极速 (100%油门): ~119 km/h")
    print("    油门范围: 0 ~ 100%")
    print()
    print(f"  PID 参数: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    print()
    print("  位置式公式: u = Kp·e + Ki·Ts·Σe + Kd·(e-e_prev)/Ts")
    print("    → 积分项 Σe 累加全部历史误差")
    print()
    print("  增量式公式: Δu = Kp·(e-e_prev) + Ki·Ts·e + Kd·(e-2e_prev+e_prev2)/Ts")
    print("              u = u_prev + Δu")
    print("    → 只算增量, 没有 Σe 累加器")

    # ═══════════════════════════════════════════
    # 工况 A: 正常加速 0 → 100 km/h
    # ═══════════════════════════════════════════
    t_a, r_pos_a, r_inc_a, m_pos_a, m_inc_a = run_scenario(
        "工况 A — 正常加速 (0 → 100 km/h)",
        t_max=30,
        r_func=lambda tv: 100.0,
        desc="从静止加速到 100 km/h, 油门未到极限"
    )

    # ═══════════════════════════════════════════
    # 工况 B: 先追 150 (达不到), 再降到 80 → 暴露积分饱和
    # ═══════════════════════════════════════════
    def r_high_then_low(tv):
        if tv < 12: return 150.0   # 达不到 → 油门持续 100%
        else:       return 80.0    # 降目标 → 位置式积分放不掉 → undershoot

    t_b, r_pos_b, r_inc_b, m_pos_b, m_inc_b = run_scenario(
        "工况 B — 目标 150→80 (先饱和再降目标, 测积分'放不掉')",
        t_max=30,
        r_func=r_high_then_low,
        desc="先追达不到的 150, 再降到 80 — 位置式的积分'欠债'要还"
    )

    # ═══════════════════════════════════════════
    # 工况 C: 巡航中变速 (60 → 80 → 40 km/h)
    # ═══════════════════════════════════════════
    def r_multi(tv):
        if tv < 5:   return 60.0
        elif tv < 10: return 80.0
        else:         return 40.0

    t_c, r_pos_c, r_inc_c, m_pos_c, m_inc_c = run_scenario(
        "工况 C — 巡航变速 (60→80→40 km/h)",
        t_max=20,
        r_func=r_multi,
        desc="连续改变目标车速, 测试模式切换平滑性",
        show_metrics=False  # 多段目标, 单一指标无意义
    )

    # ═══════════════════════════════════════════
    # 画图: 3 行 × 3 列
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("位置式 (蓝) vs 增量式 (红虚线) — 车辆定速巡航三种工况",
                 fontweight='bold', fontsize=13)

    scenarios = [
        (0, t_a, r_pos_a, r_inc_a, "工况 A: 正常加速 0→100 km/h"),
        (1, t_b, r_pos_b, r_inc_b, "工况 B: 目标 150→80 (先饱和, 积分残留)"),
        (2, t_c, r_pos_c, r_inc_c, "工况 C: 巡航变速 60→80→40 km/h"),
    ]

    for row, t_vec, r_pos, r_inc, title in scenarios:
        # 速度曲线
        ax = axes[row, 0]
        ax.plot(t_vec, r_pos['r'], 'k--', lw=1, alpha=0.5, label='目标车速')
        ax.plot(t_vec, r_pos['v'], 'b-', lw=2, label='位置式')
        ax.plot(t_vec, r_inc['v'], 'r--', lw=2, label='增量式')
        ax.set_ylabel('车速 (km/h)')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # 油门曲线
        ax = axes[row, 1]
        ax.plot(t_vec, r_pos['u'], 'b-', lw=1.5, label='位置式')
        ax.plot(t_vec, r_inc['u'], 'r--', lw=1.5, label='增量式')
        ax.axhline(100, color='gray', ls=':', alpha=0.5)
        ax.fill_between(t_vec, 0, 100, alpha=0.03, color='gray')
        ax.set_ylabel('油门 (%)')
        ax.set_title('油门踏板位置')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # 积分累积
        ax = axes[row, 2]
        ax.plot(t_vec, np.cumsum(r_pos['e']) * Ts, 'b-', lw=1.5, label='位置式 Σe')
        ax.plot(t_vec, np.cumsum(r_inc['e']) * Ts, 'r--', lw=1.5, label='增量式 (虚拟) Σe')
        ax.set_ylabel('Σe·Ts')
        ax.set_title('误差累积 (位置式有显式积分器)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ── 结论 ──
    print()
    print("  ═══════════════════════════════════════")
    print("  总结")
    print("  ═══════════════════════════════════════")
    print("  工况 A (正常): 两者性能等价, 曲线重合")
    print("  工况 B (先饱和再降目标): 追 150 时两车油门都 100%,")
    print("         但位置式积分疯狂累积。目标降到 80 后,")
    print("         位置式积分'放不掉' → 油门迟迟不收 → 车速降不到 80")
    print("         增量式没有 Σe 累加器 → 油门立即回收 → 平稳到 80")
    print("  工况 C (变速): 增量式只算油门变化量 → 切换更平滑")
    print("                (位置式每次目标跳变, 积分项都要重新适应)")

    # ═══════════════════════════════════════════
    # 看图指南
    # ═══════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║            【看图指南 — 3×3 九张子图】           ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()
    print("  每行 = 一个工况, 每列 = 一种视角:")
    print("    第1列: 车速 (看跟踪效果)")
    print("    第2列: 油门 (看控制动作)")
    print("    第3列: 误差累积 Σe·Ts (看积分行为) ← 关键列!")
    print()
    print("  曲线约定 (全图统一):")
    print("    蓝色实线 = 位置式 PID")
    print("    红色虚线 = 增量式 PID")
    print("    黑色虚线 = 目标车速 r(t)")
    print()
    print("  ═══════════════════════════════════════")
    print("  第一行: 工况 A — 正常加速 0→100 km/h")
    print("  ═══════════════════════════════════════")
    print("    图(1,1) 车速: 蓝/红重叠 → 两者几乎一样")
    print("    图(1,2) 油门: 蓝/红重叠, 油门 0→60% 再回落")
    print("    图(1,3) Σe:   蓝/红也重叠 → 正常工况无差异")
    print()
    print("    → 结论: 没有饱和时, 两种 PID 数学上等价")
    print()
    print("  ═══════════════════════════════════════")
    print("  第二行: 工况 B — 目标 150→80 km/h ← 核心对比!")
    print("  ═══════════════════════════════════════")
    print("    图(2,1) 车速:")
    print("      t=0~12s: 蓝/红都追不上 150 (油门 100% 饱和)")
    print("      t=12s:   目标突然降到 80")
    print("      蓝色实线 → 保持在 110+ km/h, 迟迟降不到 80!")
    print("      红色虚线 → 迅速降到 80, 平稳跟踪")
    print("      ↓ 这是积分饱和的典型症状 ↓")
    print()
    print("    图(2,2) 油门:")
    print("      t=0~12s: 两车油门都在 100% (地板油)")
    print("      t=12s:   位置式(蓝实线)迟迟不收油!")
    print("               → 因为积分项 Σe '胀得太大了'")
    print("               增量式(红虚线)立即收油到 ~60%")
    print()
    print("    图(2,3) Σe ← 这张图是根因!")
    print("      位置式(蓝实线): 0~12s 积分疯狂累加到很大正值")
    print("        → 目标降到 80 后, 积分'放不掉'")
    print("        → 油门被积分项顶着不收 → 车速降不到 80")
    print("      增量式(红虚线): 没有 Σe 累加器")
    print("        → 目标变了, 只根据当前误差算油门变化量")
    print("        → 无历史包袱 → 立即响应新目标")
    print()
    print("    公式对比:")
    print("    位置式: u = Kp·e + Ki·(Σe)·Ts + Kd·(e-e_prev)/Ts")
    print("            ↑ 积分项 Σe 记住全部历史, 饱和时'胀破'")
    print("    增量式: Δu = Kp·(e-e_prev) + Ki·Ts·e + Kd·(e-2e_prev+e_prev2)/Ts")
    print("            u = u_prev + Δu")
    print("            ↑ 没有 Σe, 每次只算增量, 天然免疫饱和")
    print()
    print("  ═══════════════════════════════════════")
    print("  第三行: 工况 C — 巡航变速 60→80→40 km/h")
    print("  ═══════════════════════════════════════")
    print("    图(3,1) 车速: 两车都跟上了目标变化")
    print("              但位置式(蓝实线)每次切换有微小过冲")
    print("              增量式(红虚线)切换更平滑")
    print("    图(3,2) 油门: 增量式油门动作更'干净'")
    print("    图(3,3) Σe: 位置式积分曲线(蓝)有锯齿形的累积/释放")
    print("              增量式虚拟 Σe(红)不累积, 只反映瞬时行为")
    print()
    print("  ┌─ 验证清单 ─────────────────────────────────────┐")
    print("  │ □ 图(2,1): 位置式(蓝)停在 80 以上多远?          │")
    print("  │    → 离 80 越远, 说明积分饱和越严重             │")
    print("  │ □ 图(2,3): 位置式 Σe(蓝) 在 12s 时有多高?       │")
    print("  │    → 这个高度 = '欠债'量, 对应图(2,1)车速降不下来│")
    print("  │ □ 图(2,1) vs 图(2,3): 积分最高点 和 车速高点    │")
    print("  │    是否时间上对应? → 因果关系的视觉证据          │")
    print("  │ □ 图(1,1): 蓝红是否重合?                        │")
    print("  │    → 重合 = 无饱和时两者等价, 验证公式推导       │")
    print("  └─────────────────────────────────────────────────┘")

    plt.show()


if __name__ == '__main__':
    main()
