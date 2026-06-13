"""
PID 车辆定速巡航 — 工程要点演示
===================================
用车控场景学 PID: 位置式 vs 增量式 / 积分限幅 / 死区

Run:  python main.py
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')

DEMOS = {
    '1': {
        'name': '位置式 vs 增量式 PID (三种工况)',
        'file': 'demo_01_positional_vs_incremental',
        'desc': '正常加速 / 油门踩到底 / 巡航变速 — 三种真实工况对比',
    },
    '2': {
        'name': '积分限幅 — 爬陡坡场景',
        'file': 'demo_02_anti_windup',
        'desc': '爬坡时油门到底, 坡顶后无保护 PID 严重超速 vs Clamping 平稳恢复',
    },
}


def print_banner():
    print()
    print("=" * 55)
    print("  PID 车辆定速巡航 — 工程要点演示")
    print("=" * 55)
    print()


def print_menu():
    for key in ['1', '2']:
        d = DEMOS[key]
        print(f"  [{key}]  {d['name']}")
        print(f"       {d['desc']}")
        print()
    print("  [a]  依次运行全部")
    print("  [q]  退出")
    print()


def run_demo(key):
    d = DEMOS[key]
    print(f"\n{'='*55}")
    print(f"  [{key}]  {d['name']}")
    print(f"{'='*55}")
    mod = __import__(f"demos.{d['file']}", fromlist=['main'])
    mod.main()


def main():
    print_banner()
    print_menu()

    while True:
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if choice == 'q':
            print("  Bye.")
            break
        elif choice == 'a':
            for key in ['1', '2']:
                run_demo(key)
                input("\n  Press Enter for next demo...")
        elif choice in DEMOS:
            run_demo(choice)
        else:
            print(f"  无效输入, 请输 1-2, 'a', 或 'q'.")


if __name__ == '__main__':
    main()
