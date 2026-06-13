"""
Car-like 2D double-integrator LQR demo --- interactive parameter tuning.

Model
-----
Continuous:  dx/dt = vx,  dy/dt = vy,  dvx/dt = ax,  dvy/dt = ay
State:       [x, y, vx, vy]^T
Control:     [ax, ay]^T

Usage
-----
  python car_demo.py                          # default run
  python car_demo.py --qx 50 --qvx 0.1        # override Q weights
  python car_demo.py --rx 2.0                 # heavier control penalty
  python car_demo.py -i                       # interactive tuning mode
  python car_demo.py --sweep qx 0.1 100 20    # sweep qx from 0.1 to 100 (20 points)
  python car_demo.py --sweep rx 0.01 10 15    # sweep rx from 0.01 to 10 (15 points)

Key intuition
-------------
- Q (state cost): larger -> more aggressive control, faster convergence, higher control effort
- R (input cost): larger -> more conservative control, smoother trajectory, slower convergence
- Q/R ratio fundamentally determines closed-loop pole locations
- Spectral radius rho(A-BK) < 1 guarantees stability; smaller rho = faster convergence
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from lqr import dlqr


# ===============================================================================
# Dynamics
# ===============================================================================

def car_dynamics(dt):
    """Discrete double-integrator in 2D (zero-order hold)."""
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ])  # [4, 4]

    B = np.array([
        [0.5 * dt**2, 0],
        [0, 0.5 * dt**2],
        [dt, 0],
        [0, dt],
    ])  # [4, 2]

    return A, B


# ===============================================================================
# Simulation & analysis
# ===============================================================================

def simulate(A, B, K, x0, n_steps):
    """Closed-loop rollout: x_{k+1} = (A - BK) x_k, u_k = -K x_k."""
    n, m = A.shape[0], B.shape[1]

    X = np.zeros((n_steps + 1, n))
    U = np.zeros((n_steps, m))

    x = x0.copy()
    X[0] = x
    for k in range(n_steps):
        u = -K @ x                         # [m]
        U[k] = u
        x = A @ x + B @ u                  # [n]
        X[k + 1] = x

    return X, U


def analyse(A, B, K, X, U, Q, R):
    """Print diagnostic summary for a closed-loop run."""
    A_cl = A - B @ K
    eigvals = np.linalg.eigvals(A_cl)
    rho = np.max(np.abs(eigvals))

    final_pos_err = np.linalg.norm(X[-1, :2])
    final_vel_err = np.linalg.norm(X[-1, 2:])
    state_cost = np.sum(np.diag(X @ Q @ X.T))
    input_cost = np.sum(np.diag(U @ R @ U.T))
    total_cost = state_cost + input_cost

    print(f"\n{'='*60}")
    print(f"  Spectral radius rho(A-BK): {rho:.6f}  {'[stable]' if rho < 1 else '[UNSTABLE]'}")
    print(f"  Final position error:  {final_pos_err:.4f} m")
    print(f"  Final velocity error:  {final_vel_err:.4f} m/s")
    print(f"  State cost (x^T Q x):   {state_cost:.2f}")
    print(f"  Input cost (u^T R u):   {input_cost:.2f}")
    print(f"  Total LQR cost:         {total_cost:.2f}")
    print(f"{'='*60}")
    return rho, total_cost


def plot_results(X, U, dt, Q, R, x0, suffix=""):
    """4-panel plot: trajectory, position, velocity, control."""
    t = np.arange(X.shape[0]) * dt
    t_u = np.arange(U.shape[0]) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Trajectory (x-y)
    ax = axes[0, 0]
    ax.plot(X[:, 0], X[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(*x0[:2], c='green', s=100, zorder=5, label='Start')
    ax.scatter(0, 0, c='red', s=120, marker='x', zorder=5, label='Target')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Vehicle Trajectory')
    ax.legend(); ax.axis('equal'); ax.grid(True, alpha=0.3)

    # Position vs time
    ax = axes[0, 1]
    ax.plot(t, X[:, 0], 'b-', label='x')
    ax.plot(t, X[:, 1], 'r--', label='y')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Position [m]')
    ax.set_title('Position vs Time')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Velocity vs time
    ax = axes[1, 0]
    ax.plot(t, X[:, 2], 'b-', label='vx')
    ax.plot(t, X[:, 3], 'r--', label='vy')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Velocity vs Time')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Control vs time
    ax = axes[1, 1]
    ax.step(t_u, U[:, 0], 'b-', where='post', label='ax')
    ax.step(t_u, U[:, 1], 'r--', where='post', label='ay')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Acceleration [m/s^2]')
    ax.set_title(f'Control Inputs  (R = {R[0,0]:.2f})')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Info text
    fig.text(0.02, 0.02,
             f"diag(Q) = [{Q[0,0]:.1f}, {Q[1,1]:.1f}, {Q[2,2]:.1f}, {Q[3,3]:.1f}]  "
             f"diag(R) = [{R[0,0]:.2f}, {R[1,1]:.2f}]  dt = {dt:.3f}",
             fontsize=8, family='monospace')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fname = f"car_lqr_demo{suffix}.png"
    plt.savefig(fname, dpi=150)
    print(f"""
  *** Figure saved to: {fname} ***
  >>> Open {fname} to view trajectory, velocity, and control plots <<<
""")


# ===============================================================================
# Run one experiment
# ===============================================================================

def run_once(A, B, Q, R, x0, n_steps, dt, verbose=True):
    """Solve LQR, simulate, analyse, return (K, P, X, U, rho, cost)."""
    K, P = dlqr(A, B, Q, R)
    if verbose:
        print("\nOptimal feedback gain K:\n", K)
    X, U = simulate(A, B, K, x0, n_steps)
    rho, cost = analyse(A, B, K, X, U, Q, R)
    return K, P, X, U, rho, cost


# ===============================================================================
# Interactive mode
# ===============================================================================

def _print_params(params):
    print(f"""
  Current parameters
  --------------------------------------------------
    dt  = {params['dt']:.4f}        T = {params['T']:.1f}
    x0  = [{params['x0'][0]:.1f}, {params['x0'][1]:.1f}, {params['x0'][2]:.1f}, {params['x0'][3]:.1f}]
    Q   = diag([{params['Q'][0,0]:.2f}, {params['Q'][1,1]:.2f}, {params['Q'][2,2]:.2f}, {params['Q'][3,3]:.2f}])
    R   = diag([{params['R'][0,0]:.2f}, {params['R'][1,1]:.2f}])
  --------------------------------------------------""")

def _print_menu():
    print("""
  Commands
  --------------------------------------------------
    qx <val> / qy <val>       set position penalty (Q diagonal)
    qvx <val> / qvy <val>      set velocity penalty
    rx <val>  / ry <val>       set control penalty (R diagonal)
    dt <val>                   set discretization step
    T  <val>                   set simulation duration
    x0 <v1> <v2> <v3> <v4>    set initial state [x, y, vx, vy]
    run                        re-run with current params
    sweep qx <lo> <hi> <n>     sweep a parameter (e.g. sweep qx 0.1 100 20)
    show                       show last results summary
    reset                      reset to defaults
    help                       show this menu
    quit                       exit
  --------------------------------------------------""")

def _sweep_param(name, lo, hi, n_pts, A, B, params):
    """Sweep one Q/R diagonal element and plot resulting spectral-radius & cost."""
    vals = np.linspace(lo, hi, n_pts)
    rho_list, cost_list = [], []
    n_steps = int(params['T'] / params['dt'])
    x0 = params['x0']

    for v in vals:
        Q = params['Q'].copy()
        R = params['R'].copy()
        if name.startswith('q'):
            idx = {'qx': 0, 'qy': 1, 'qvx': 2, 'qvy': 3}[name]
            Q[idx, idx] = v
        elif name.startswith('r'):
            idx = {'rx': 0, 'ry': 1}[name]
            R[idx, idx] = v
        else:
            print(f"Unknown sweep parameter: {name}"); return

        K, _ = dlqr(A, B, Q, R)
        A_cl = A - B @ K
        rho = np.max(np.abs(np.linalg.eigvals(A_cl)))
        rho_list.append(rho)

        X, U = simulate(A, B, K, x0, n_steps)
        total = np.sum(np.diag(X @ Q @ X.T)) + np.sum(np.diag(U @ R @ U.T))
        cost_list.append(total)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(vals, rho_list, 'b-o', markersize=4)
    ax1.axhline(1, color='r', ls='--', label='stability boundary')
    ax1.set_xlabel(name); ax1.set_ylabel('rho(A-BK)')
    ax1.set_title(f'Spectral radius vs {name}')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(vals, cost_list, 'r-s', markersize=4)
    ax2.set_xlabel(name); ax2.set_ylabel('Total LQR cost')
    ax2.set_title(f'Total cost vs {name}')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"Sweep done. {name}: rho range [{min(rho_list):.4f}, {max(rho_list):.4f}]")


def interactive_mode(defaults):
    """REPL for tuning LQR parameters and observing results."""
    params = {k: v for k, v in defaults.items()}  # copy
    A, B = car_dynamics(params['dt'])
    last_run = None  # (K, X, U, rho, cost)

    print("\n" + "=" * 60)
    print("  LQR Interactive Tuning Mode")
    print("  Tune Q/R weights and observe how the controller behaves")
    print("=" * 60)
    _print_params(params)
    _print_menu()

    while True:
        try:
            raw = input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == 'quit' or cmd == 'exit' or cmd == 'q':
            break

        elif cmd == 'help' or cmd == 'h':
            _print_menu()

        elif cmd == 'show':
            if last_run is None:
                print("No run yet. Use 'run' first.")
            else:
                K, X, U, rho, cost = last_run
                print(f"\n  Last run -- rho: {rho:.6f}  |  total cost: {cost:.2f}")

        elif cmd == 'params' or cmd == 'p':
            _print_params(params)

        elif cmd == 'reset':
            params = {k: v for k, v in defaults.items()}
            A, B = car_dynamics(params['dt'])
            print("Parameters reset to defaults.")
            _print_params(params)

        elif cmd == 'run' or cmd == 'r':
            n_steps = int(params['T'] / params['dt'])
            A, B = car_dynamics(params['dt'])
            K, P, X, U, rho, cost = run_once(A, B, params['Q'], params['R'],
                                              params['x0'], n_steps, params['dt'])
            last_run = (K, X, U, rho, cost)
            plot_results(X, U, params['dt'], params['Q'], params['R'], params['x0'],
                         suffix="_interactive")

        elif cmd in ('qx', 'qy', 'qvx', 'qvy'):
            if len(parts) < 2:
                print(f"Usage: {cmd} <value>")
                continue
            val = float(parts[1])
            idx = {'qx': 0, 'qy': 1, 'qvx': 2, 'qvy': 3}[cmd]
            params['Q'][idx, idx] = val
            print(f"  Q[{idx},{idx}] = {val:.2f}  (position penalty)" if idx < 2 else
                  f"  Q[{idx},{idx}] = {val:.2f}  (velocity penalty)")

        elif cmd in ('rx', 'ry'):
            if len(parts) < 2:
                print(f"Usage: {cmd} <value>")
                continue
            val = float(parts[1])
            idx = {'rx': 0, 'ry': 1}[cmd]
            params['R'][idx, idx] = val
            print(f"  R[{idx},{idx}] = {val:.2f}  (control penalty)")

        elif cmd == 'dt':
            if len(parts) < 2:
                print("Usage: dt <value>")
                continue
            params['dt'] = float(parts[1])
            A, B = car_dynamics(params['dt'])
            print(f"  dt = {params['dt']:.4f}  (dynamics rebuilt)")

        elif cmd == 't':
            if len(parts) < 2:
                print("Usage: T <value>")
                continue
            params['T'] = float(parts[1])
            print(f"  T = {params['T']:.1f} s")

        elif cmd == 'x0':
            if len(parts) < 5:
                print("Usage: x0 <x> <y> <vx> <vy>")
                continue
            params['x0'] = np.array([float(parts[1]), float(parts[2]),
                                     float(parts[3]), float(parts[4])])
            print(f"  x0 = {params['x0']}")

        elif cmd == 'sweep':
            if len(parts) < 5:
                print("Usage: sweep <param> <lo> <hi> <n_points>")
                print("  param: qx, qy, qvx, qvy, rx, ry")
                continue
            name = parts[1].lower()
            lo, hi, n_pts = float(parts[2]), float(parts[3]), int(parts[4])
            A, B = car_dynamics(params['dt'])
            _sweep_param(name, lo, hi, n_pts, A, B, params)

        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for available commands.")


# ===============================================================================
# Main entry point
# ===============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Car-like 2D LQR demo -- tune Q/R weights and learn LQR behaviour.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python car_demo.py                              # default run
  python car_demo.py --qx 50 --qvx 0.1            # aggressive position, cheap velocity
  python car_demo.py --rx 5.0                      # expensive control -> smoother
  python car_demo.py -i                            # interactive tuning mode
  python car_demo.py --sweep qx 0.1 100 20         # sweep position penalty
  python car_demo.py --sweep rx 0.01 10 15         # sweep control penalty
""")
    # Q weights (state cost)
    p.add_argument('--qx',  type=float, help='Q weight on x position (default: 10)')
    p.add_argument('--qy',  type=float, help='Q weight on y position (default: 10)')
    p.add_argument('--qvx', type=float, help='Q weight on x velocity (default: 1)')
    p.add_argument('--qvy', type=float, help='Q weight on y velocity (default: 1)')
    # R weights (input cost)
    p.add_argument('--rx',  type=float, help='R weight on ax control (default: 0.5)')
    p.add_argument('--ry',  type=float, help='R weight on ay control (default: 0.5)')
    # Dynamics
    p.add_argument('--dt',  type=float, help='Discretization step [s] (default: 0.05)')
    p.add_argument('--T',   type=float, help='Simulation duration [s] (default: 8.0)')
    # Initial state
    p.add_argument('--x0',  type=float, nargs=4, metavar=('X', 'Y', 'VX', 'VY'),
                   help='Initial state (default: 5 3 2 -1.5)')
    # Modes
    p.add_argument('-i', '--interactive', action='store_true',
                   help='Enter interactive parameter-tuning mode')
    p.add_argument('--sweep', type=str, nargs=4, metavar=('PARAM', 'LO', 'HI', 'N'),
                   help='Sweep a parameter and plot trends (qx/qy/qvx/qvy/rx/ry)')
    p.add_argument('--no-plot', action='store_true', help='Suppress plot output')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Build defaults ---
    defaults = {
        'dt': 0.05,
        'T':  8.0,
        'x0': np.array([5.0, 3.0, 2.0, -1.5]),
        'Q':  np.diag([10.0, 10.0, 1.0, 1.0]),
        'R':  np.diag([0.5, 0.5]),
    }

    # Override with CLI args
    if args.qx  is not None: defaults['Q'][0, 0] = args.qx
    if args.qy  is not None: defaults['Q'][1, 1] = args.qy
    if args.qvx is not None: defaults['Q'][2, 2] = args.qvx
    if args.qvy is not None: defaults['Q'][3, 3] = args.qvy
    if args.rx  is not None: defaults['R'][0, 0] = args.rx
    if args.ry  is not None: defaults['R'][1, 1] = args.ry
    if args.dt  is not None: defaults['dt'] = args.dt
    if args.T   is not None: defaults['T']  = args.T
    if args.x0  is not None: defaults['x0'] = np.array(args.x0)

    dt = defaults['dt']
    T  = defaults['T']
    x0 = defaults['x0']
    Q  = defaults['Q']
    R  = defaults['R']
    n_steps = int(T / dt)

    # --- Interactive mode ---
    if args.interactive:
        interactive_mode(defaults)
        return

    A, B = car_dynamics(dt)

    # --- Sweep mode ---
    if args.sweep is not None:
        name, lo, hi, n_pts = args.sweep[0], float(args.sweep[1]), float(args.sweep[2]), int(args.sweep[3])
        _sweep_param(name.lower(), lo, hi, n_pts, A, B, defaults)
        return

    # --- Standard single-run mode ---
    K, P, X, U, rho, cost = run_once(A, B, Q, R, x0, n_steps, dt, verbose=True)

    if not args.no_plot:
        plot_results(X, U, dt, Q, R, x0)


if __name__ == "__main__":
    main()
