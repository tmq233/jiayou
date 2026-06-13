"""Headless benchmark — validates MPC tracking performance without GUI.

Usage:
    python benchmark.py              # default params, both paths
    python benchmark.py --path circle --steps 500
    python benchmark.py --path figure8 --N 20 --Q_xy 50
"""

import argparse
import time
import numpy as np

from mpc_core import MPCController
from vehicle_model import VehicleModel
from demo import ref_circle, ref_figure8


def run_benchmark(path_type='circle', N=15, dt=0.1, Q_xy=10, Q_theta=5,
                  Q_v=1, R_a=0.1, R_delta=0.1, L=2.5, v_des=8.0,
                  steps=300, noise_std=0.0):
    """Run a headless MPC simulation and return performance metrics."""

    mpc = MPCController(
        dt=dt, N=N,
        Q_diag=np.array([Q_xy, Q_xy, Q_theta, Q_v]),
        R_diag=np.array([R_a, R_delta]),
        L=L,
    )
    vehicle = VehicleModel(dt=dt, L=L)
    if noise_std > 0:
        vehicle.set_noise(noise_std, noise_std, noise_std * 0.3, noise_std * 0.6)

    # Initialize at reference
    ref_fn = ref_circle if path_type == 'circle' else ref_figure8
    s0 = ref_fn(0.0, v_des=v_des) if path_type == 'circle' else ref_fn(0.0)
    vehicle.reset(s0)

    t = 0.0
    pos_errors = []
    vel_errors = []
    solve_times = []
    controls = []

    for step_i in range(steps):
        ts = t + np.arange(N + 1) * dt
        if path_type == 'circle':
            x_ref = np.array([ref_circle(ti, v_des=v_des) for ti in ts])
        else:
            x_ref = np.array([ref_figure8(ti) for ti in ts])

        t0 = time.perf_counter()
        try:
            u_opt, u_seq, x_pred, info = mpc.solve(vehicle.state, x_ref)
        except Exception:
            u_opt = np.zeros(2)
            info = {'success': False}
        solve_times.append(time.perf_counter() - t0)

        vehicle.step(u_opt)
        t += dt
        controls.append(u_opt)

        # Errors — vehicle is now at t (after step + t+=dt), x_ref[1] is at t
        ref_now = x_ref[1]
        pos_err = np.linalg.norm(vehicle.state[:2] - ref_now[:2])
        vel_err = abs(vehicle.state[3] - ref_now[3])
        pos_errors.append(pos_err)
        vel_errors.append(vel_err)

        if not info.get('success'):
            print(f'  [WARN] Solve failed at step {step_i}')

    pos_errors = np.array(pos_errors)
    vel_errors = np.array(vel_errors)
    solve_times = np.array(solve_times)
    controls = np.array(controls)

    print(f'\n═══ Benchmark Results: {path_type} ═══')
    print(f'  Steps: {steps},  N={N},  dt={dt}')
    print(f'  Q=({Q_xy}, {Q_xy}, {Q_theta}, {Q_v}),  R=({R_a}, {R_delta})')
    print(f'  Noise: {noise_std}')
    print()
    print(f'  Position error — mean: {pos_errors.mean():.4f} m,  '
          f'max: {pos_errors.max():.4f} m,  std: {pos_errors.std():.4f} m')
    print(f'  Velocity error — mean: {vel_errors.mean():.4f} m/s,  '
          f'max: {vel_errors.max():.4f} m/s')
    print(f'  Control — |a| mean: {np.abs(controls[:, 0]).mean():.4f},  '
          f'|δ| mean: {np.degrees(np.abs(controls[:, 1])).mean():.1f}°')
    print(f'  Solve time — mean: {solve_times.mean()*1000:.2f} ms,  '
          f'max: {solve_times.max()*1000:.2f} ms,  '
          f'sum: {solve_times.sum():.2f} s')
    print(f'  Solve rate: {steps / solve_times.sum():.1f} Hz')
    print()

    return {
        'pos_error_mean': pos_errors.mean(),
        'pos_error_max': pos_errors.max(),
        'vel_error_mean': vel_errors.mean(),
        'solve_time_mean_ms': solve_times.mean() * 1000,
        'solve_rate_hz': steps / solve_times.sum(),
    }


def main():
    parser = argparse.ArgumentParser(description='MPC Headless Benchmark')
    parser.add_argument('--path', default='both', choices=['circle', 'figure8', 'both'])
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--N', type=int, default=15)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--Q_xy', type=float, default=10)
    parser.add_argument('--Q_theta', type=float, default=5)
    parser.add_argument('--Q_v', type=float, default=1)
    parser.add_argument('--R_a', type=float, default=0.1)
    parser.add_argument('--R_delta', type=float, default=0.1)
    parser.add_argument('--L', type=float, default=2.5)
    parser.add_argument('--v_des', type=float, default=8.0)
    parser.add_argument('--noise_std', type=float, default=0.0)
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k not in ('path', 'steps')}
    paths = ['circle', 'figure8'] if args.path == 'both' else [args.path]

    for p in paths:
        run_benchmark(path_type=p, steps=args.steps, **kwargs)


if __name__ == '__main__':
    main()
