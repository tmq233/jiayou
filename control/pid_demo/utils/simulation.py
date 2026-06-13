"""
车辆定速巡航仿真引擎

车辆模型:
  发动机:  tau_e * dF/dt + F = k_e * u     油门(%) → 驱动力(N)
  车身:    m * dv/dt = F - b * v             驱动力 → 车速(km/h)

参数说明:
  u    = 油门踏板 [0~100%]
  v    = 车速 [km/h]
  F    = 发动机驱动力 (内部状态, 归一化)
  tau_e = 0.3s   发动机响应滞后
  k_e  = 50      100%油门对应的稳态驱动力
  m    = 150     车重(归一化, 越小加速越快)
  b    = 42      风阻+滚动阻力系数, 极速 = k_e*100/b ≈ 119 km/h
                机械时间常数 m/b ≈ 3.6s (合理响应速度)
"""

import numpy as np


# ── 默认车辆参数 ──
DEFAULT_CAR = dict(tau_e=0.3, k_e=50, m=150, b=42)


def simulate_car(controller, t, r_func, car=None, extra_load=0.0,
                 load_start=1e9, load_end=-1e9):
    """
    车辆定速巡航仿真.

    Parameters
    ----------
    controller : PID 实例 (有 .update(y, r), .Ts)
    t : ndarray  时间轴 [s]
    r_func : callable  目标车速 r(t) [km/h]
    car : dict  车辆参数, 默认 DEFAULT_CAR
    extra_load : float  额外阻力 (如爬坡), 单位与驱动力同量纲
    load_start : float  额外阻力开始时间 [s]
    load_end : float    额外阻力结束时间 [s]

    Returns
    -------
    dict: t, v(车速), u(油门%), e(速度误差), r(目标), F(驱动力)
    """
    if car is None:
        car = DEFAULT_CAR
    tau_e, k_e, m, b = car['tau_e'], car['k_e'], car['m'], car['b']

    N = len(t)
    Ts = controller.Ts
    dt_inner = 0.0001

    v = np.zeros(N)   # 车速 [km/h]
    u = np.zeros(N)   # 油门 [%]
    e = np.zeros(N)   # 速度误差 [km/h]
    r = np.zeros(N)   # 目标车速 [km/h]
    F_arr = np.zeros(N)  # 驱动力

    F_state = 0.0     # 发动机驱动力初始
    v_state = 0.0     # 车速初始

    for k in range(N):
        r_k = r_func(t[k])
        r[k] = r_k

        # PID 根据当前车速计算油门
        u_k = controller.update(v_state, r_k)
        u_k = np.clip(u_k, 0.0, 100.0)  # 油门物理限幅 0~100%
        u[k] = u_k

        v[k] = v_state
        e[k] = r_k - v_state
        F_arr[k] = F_state

        # 判断当前是否在额外阻力区间
        load = extra_load if (load_start <= t[k] <= load_end) else 0.0

        # 内环仿真 (欧拉法, 步长 < Ts)
        n_inner = max(1, int(round(Ts / dt_inner)))
        dt_a = Ts / n_inner
        for _ in range(n_inner):
            dF = (k_e * u_k - F_state) / tau_e
            dv = (F_state - b * v_state - load) / m
            F_state += dt_a * dF
            v_state += dt_a * dv

    return dict(t=t, v=v, u=u, e=e, r=r, F=F_arr)


def simulate_car_with_backlash(controller, t, r_func, backlash_gap=3.0,
                               car=None):
    """
    带油门拉线间隙(backlash)的车辆巡航仿真.

    backlash_gap: 油门死区宽度 [%], 即踩下/松开 gap% 以内油门实际不动.
    """
    if car is None:
        car = DEFAULT_CAR
    tau_e, k_e, m, b = car['tau_e'], car['k_e'], car['m'], car['b']

    N = len(t)
    Ts = controller.Ts
    dt_inner = 0.0001

    v = np.zeros(N)
    u_cmd = np.zeros(N)    # PID 指令油门
    u_actual = np.zeros(N) # 经过 backlash 后的实际油门
    e = np.zeros(N)
    r = np.zeros(N)
    F_arr = np.zeros(N)

    F_state = 0.0
    v_state = 0.0
    u_actual_prev = 0.0

    for k in range(N):
        r_k = r_func(t[k])
        r[k] = r_k

        u_k = controller.update(v_state, r_k)
        u_k = np.clip(u_k, 0.0, 100.0)
        u_cmd[k] = u_k

        # Backlash 模型: 油门拉线间隙
        du = u_k - u_actual_prev
        if abs(du) < backlash_gap / 2:
            u_actual_k = u_actual_prev
        elif du > 0:
            u_actual_k = u_k - backlash_gap / 2
        else:
            u_actual_k = u_k + backlash_gap / 2
        u_actual[k] = u_actual_k
        u_actual_prev = u_actual_k

        v[k] = v_state
        e[k] = r_k - v_state
        F_arr[k] = F_state

        n_inner = max(1, int(round(Ts / dt_inner)))
        dt_a = Ts / n_inner
        for _ in range(n_inner):
            dF = (k_e * u_actual_k - F_state) / tau_e
            dv = (F_state - b * v_state) / m
            F_state += dt_a * dF
            v_state += dt_a * dv

    return dict(t=t, v=v, u_cmd=u_cmd, u_actual=u_actual, e=e, r=r, F=F_arr)


def cruise_metrics(t, v, v_target):
    """
    从仿真数据中提取巡航性能指标.

    Returns
    -------
    dict: overshoot_kmh, rise_time_s, settling_time_s, steady_error_kmh
    """
    # 稳态值: 取最后 20% 数据的均值
    n_ss = len(v) // 5
    v_ss = v[-n_ss:].mean() if n_ss > 0 else v[-1]

    steady_error = abs(v_target - v_ss)
    overshoot = max(0.0, np.max(v) - v_target)

    # 上升时间: 10% → 90% 目标值 (仅对 step 响应有意义)
    v10 = 0.1 * v_target
    v90 = 0.9 * v_target
    idx10 = np.argmax(v >= v10) if np.any(v >= v10) else 0
    idx90 = np.argmax(v >= v90) if np.any(v >= v90) else len(t) - 1
    rise_time = t[idx90] - t[idx10]

    # 调节时间: 进入目标 ±2 km/h 后不再超出
    band = 2.0  # km/h
    settled = np.abs(v - v_target) <= band
    settling_time = t[-1]
    for i in range(len(t) - 1, -1, -1):
        if not settled[i]:
            settling_time = t[min(i + 1, len(t) - 1)]
            break

    return dict(
        overshoot_kmh=round(overshoot, 1),
        rise_time_s=round(rise_time, 2),
        settling_time_s=round(settling_time, 2),
        steady_error_kmh=round(steady_error, 2),
        v_final=round(v_ss, 1),
    )
