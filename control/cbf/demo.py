"""
Control Barrier Functions demos.

Coverage:
  1. Adaptive Cruise Control (ACC): CBF enforces safe following distance
  2. CBF vs no-CBF comparison: safety violation without CBF
  3. Obstacle avoidance: double integrator + CBF safety filter
  4. CLF-CBF-QP: combined stability + safety for ACC
"""

import numpy as np
import matplotlib.pyplot as plt
from cbf import (
    CBFConstraint,
    ECBFConstraint,
    CBFQPSolver,
    CLFCBFQPSolver,
    acc_dynamics_f,
    acc_dynamics_g,
    acc_safety_h,
    acc_safety_h_grad,
    obstacle_dynamics,
    obstacle_safety_h,
    obstacle_safety_h_grad,
    obstacle_h_dot_f,
    obstacle_h_dot_grad,
)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})


# ===================================================================
# Demo 1: ACC with CBF safety filter
# ===================================================================

def demo_acc_cbf():
    """Adaptive Cruise Control — CBF enforces D ≥ D_safe + T*v_ego."""
    print("=" * 55)
    print("Demo 1: ACC with CBF Safety Filter")
    print("=" * 55)

    # Setup
    D_safe, T = 10.0, 1.8
    v0 = 20.0  # lead car speed [m/s]
    m = 1650.0
    dt = 0.05
    n_steps = 600  # 30 seconds

    # CBF constraint
    cbf_h = CBFConstraint(
        h=lambda x: acc_safety_h(x, D_safe, T),
        h_grad=lambda x: acc_safety_h_grad(x, D_safe, T),
        alpha=1.5
    )

    # Nominal controller: PI speed control (track v_des=24, lead car at v0=20)
    v_des = 24.0  # desired speed higher than lead → nominal will push too close
    Kp, Ki = 200.0, 50.0

    solver = CBFQPSolver([cbf_h], u_min=np.array([-m * 9.0]), u_max=np.array([m * 9.0]))

    # Simulate with CBF — start safe (h > 0), but nominal would violate
    x_cbf = np.array([55.0, 18.0])  # h = 55 - 10 - 1.8*18 = 55 - 42.4 = 12.6 > 0 (SAFE)
    x_cbf_hist = np.zeros((n_steps + 1, 2))
    u_cbf_hist = np.zeros(n_steps)
    h_cbf_hist = np.zeros(n_steps + 1)
    x_cbf_hist[0] = x_cbf
    h_cbf_hist[0] = acc_safety_h(x_cbf, D_safe, T)

    integral = 0.0
    x = x_cbf.copy()
    for k in range(n_steps):
        speed_err = v_des - x[1]
        integral += speed_err * dt
        u_nom = np.array([Kp * speed_err + Ki * integral])

        f_x = acc_dynamics_f(x, v0)
        g_x = acc_dynamics_g(x, m)
        u_safe, info = solver.solve(x, lambda s: acc_dynamics_f(s, v0),
                                    lambda s: acc_dynamics_g(s, m), u_nom)

        x_dot = f_x + g_x @ u_safe
        x = x + dt * x_dot
        x_cbf_hist[k + 1] = x
        u_cbf_hist[k] = u_safe[0]
        h_cbf_hist[k + 1] = acc_safety_h(x, D_safe, T)

    # Simulate WITHOUT CBF (same nominal controller, no safety filter)
    x_nocbf = np.array([55.0, 18.0])
    x_nocbf_hist = np.zeros((n_steps + 1, 2))
    h_nocbf_hist = np.zeros(n_steps + 1)
    x_nocbf_hist[0] = x_nocbf
    h_nocbf_hist[0] = acc_safety_h(x_nocbf, D_safe, T)

    integral = 0.0
    x = x_nocbf.copy()
    for k in range(n_steps):
        speed_err = v_des - x[1]
        integral += speed_err * dt
        u_nom = np.array([Kp * speed_err + Ki * integral])
        x_dot = acc_dynamics_f(x, v0) + acc_dynamics_g(x, m) @ u_nom
        x = x + dt * x_dot
        x_nocbf_hist[k + 1] = x
        h_nocbf_hist[k + 1] = acc_safety_h(x, D_safe, T)

    t = np.arange(0, n_steps + 1) * dt

    msg = "[SAFE]" if np.all(h_cbf_hist >= -1e-6) else "[UNSAFE]"
    print(f"With CBF:    h(x) always >= 0? {msg}  (min h = {np.min(h_cbf_hist):.3f})")
    msg2 = "[SAFE]" if np.all(h_nocbf_hist >= -1e-6) else "[UNSAFE]"
    print(f"Without CBF: h(x) always >= 0? {msg2}  (min h = {np.min(h_nocbf_hist):.3f})")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Adaptive Cruise Control: CBF Safety Filter", fontsize=14)

    axes[0, 0].plot(t, x_cbf_hist[:, 0], 'b-', label='with CBF', linewidth=1.2)
    axes[0, 0].plot(t, x_nocbf_hist[:, 0], 'r--', label='no CBF', linewidth=1.2)
    axes[0, 0].axhline(y=D_safe + T * v0, color='gray', linestyle=':',
                       alpha=0.5, label=f'min safe D@v_lead = {D_safe + T*v0:.0f}m')
    axes[0, 0].set_ylabel("Distance [m]"); axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_title("Inter-vehicle Distance $D$"); axes[0, 0].legend()

    axes[0, 1].plot(t, x_cbf_hist[:, 1], 'b-', linewidth=1.2)
    axes[0, 1].plot(t, x_nocbf_hist[:, 1], 'r--', linewidth=1.2)
    axes[0, 1].axhline(y=v_des, color='orange', linestyle=':', alpha=0.5, label=f'v_des={v_des}')
    axes[0, 1].axhline(y=v0, color='gray', linestyle=':', alpha=0.3, label=f'v_lead={v0}')
    axes[0, 1].set_ylabel("Speed [m/s]"); axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_title("Ego Speed $v$"); axes[0, 1].legend()

    axes[1, 0].plot(t, h_cbf_hist, 'b-', linewidth=1.2)
    axes[1, 0].plot(t, h_nocbf_hist, 'r--', linewidth=1.2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_ylabel("$h(x)$"); axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_title("CBF Safety Function $h(x) = D - D_{safe} - T v$")
    axes[1, 0].legend(['with CBF', 'no CBF'])

    axes[1, 1].plot(t[:-1], u_cbf_hist, 'g-', linewidth=0.8)
    axes[1, 1].set_ylabel("Force [N]"); axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_title("Control Input (CBF-filtered)")

    plt.tight_layout()
    plt.savefig('control/cbf/fig_demo1_acc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved fig_demo1_acc.png")


# ===================================================================
# Demo 2: Obstacle avoidance — double integrator + CBF
# ===================================================================

def demo_obstacle_avoidance():
    """Double integrator with CBF enforcing obstacle avoidance."""
    print("\n" + "=" * 55)
    print("Demo 2: Obstacle Avoidance via CBF")
    print("=" * 55)

    dt = 0.02
    n_steps = 500

    obs = np.array([3.0, 2.0])
    R = 1.5

    cbf_h = ECBFConstraint(
        h=lambda x: obstacle_safety_h(x, obs, R),
        h_grad=lambda x: obstacle_safety_h_grad(x, obs),
        h_dot_fun=lambda x: obstacle_h_dot_f(x, obs),
        grad_h_dot=lambda x: obstacle_h_dot_grad(x, obs),
        k_vec=[2.0],
        alpha=2.0
    )

    # Nominal: go-to-goal  P-controller
    goal = np.array([8.0, 6.0])

    solver = CBFQPSolver([cbf_h], u_min=np.array([-10.0, -10.0]), u_max=np.array([10.0, 10.0]))

    # Simulate multiple start positions
    starts = [
        np.array([0.0, 0.0, 0.0, 0.0]),     # direct path through obstacle
        np.array([0.0, 5.0, 0.0, 0.0]),     # from above
        np.array([6.0, 0.0, 0.0, -1.0]),    # from right, heading toward obstacle
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Obstacle Avoidance: CBF Safety Filter on Double Integrator", fontsize=14)

    colors = ['b', 'g', 'm']

    for idx, x0 in enumerate(starts):
        x = x0.copy()
        x_hist = np.zeros((n_steps + 1, 4))
        h_hist = np.zeros(n_steps + 1)
        x_hist[0] = x
        h_hist[0] = obstacle_safety_h(x, obs, R)

        for k in range(n_steps):
            # Nominal: PD to goal
            u_nom = 2.0 * (goal - x[:2]) - 1.5 * x[2:]
            f_x, g_x = obstacle_dynamics(x, use_cbf=True)
            u_safe, info = solver.solve(x, obstacle_dynamics,
                                        lambda s: obstacle_dynamics(s, use_cbf=True)[1],
                                        u_nom)
            x_dot = f_x + g_x @ u_safe
            x = x + dt * x_dot
            x_hist[k + 1] = x
            h_hist[k + 1] = obstacle_safety_h(x, obs, R)

        ax1.plot(x_hist[:, 0], x_hist[:, 1], color=colors[idx], linewidth=1.0,
                 label=f'start ({x0[0]},{x0[1]})')
        ax1.plot(x0[0], x0[1], 'ko', markersize=6)
        ax1.plot(x_hist[-1, 0], x_hist[-1, 1], 'r*', markersize=8)

        ax2.plot(np.arange(n_steps + 1) * dt, h_hist, color=colors[idx],
                 linewidth=1.0)

        min_h = np.min(h_hist)
        print(f"  Path {idx+1}: min h(x) = {min_h:.4f} [{'SAFE' if min_h >= 0 else 'VIOLATION'}]")

    # Draw obstacle
    theta_c = np.linspace(0, 2 * np.pi, 100)
    ax1.fill(obs[0] + R * np.cos(theta_c), obs[1] + R * np.sin(theta_c),
             color='red', alpha=0.3, label='obstacle')
    ax1.plot(goal[0], goal[1], 'g*', markersize=15, label='goal')
    ax1.set_xlabel("$p_x$"); ax1.set_ylabel("$p_y$")
    ax1.set_title("2D Trajectories"); ax1.legend(fontsize=8)
    ax1.set_aspect('equal')

    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("$h(x)$")
    ax2.set_title("Safety Function Over Time")

    plt.tight_layout()
    plt.savefig('control/cbf/fig_demo2_obstacle.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved fig_demo2_obstacle.png")


# ===================================================================
# Demo 3: CLF-CBF-QP — combined stability + safety for ACC
# ===================================================================

def demo_clf_cbf_qp():
    """CLF-CBF-QP for ACC: track desired speed while maintaining safe distance."""
    print("\n" + "=" * 55)
    print("Demo 3: CLF-CBF-QP — Stability + Safety")
    print("=" * 55)

    D_safe, T_hw = 10.0, 1.8
    v0, v_des = 20.0, 24.0  # lead speed, desired ego speed
    m = 1650.0
    dt = 0.05
    n_steps = 800  # 40 seconds

    # CLF: V = ½ (v - v_des)²  (track desired speed)
    def V(x):
        return 0.5 * (x[1] - v_des) ** 2

    def V_grad(x):
        return np.array([0.0, x[1] - v_des])

    # CBF for safety
    cbf = CBFConstraint(
        h=lambda x: acc_safety_h(x, D_safe, T_hw),
        h_grad=lambda x: acc_safety_h_grad(x, D_safe, T_hw),
        alpha=2.0
    )

    solver = CLFCBFQPSolver(
        V, V_grad, [cbf],
        gamma=1.5, slack_penalty=1e5,
        u_min=np.array([-m * 9.0]), u_max=np.array([m * 9.0])
    )

    # Scenario: start far away, speed up, then lead car appears "closer"
    x0 = np.array([80.0, 22.0])  # far, near desired speed
    x = x0.copy()
    x_hist = np.zeros((n_steps + 1, 2))
    u_hist = np.zeros(n_steps)
    h_hist = np.zeros(n_steps + 1)
    V_hist = np.zeros(n_steps + 1)
    delta_hist = np.zeros(n_steps)
    x_hist[0] = x
    h_hist[0] = acc_safety_h(x, D_safe, T_hw)
    V_hist[0] = V(x)

    for k in range(n_steps):
        f_x = acc_dynamics_f(x, v0)
        g_x = acc_dynamics_g(x, m)

        u_opt, info = solver.solve(x,
                                   lambda s: acc_dynamics_f(s, v0),
                                   lambda s: acc_dynamics_g(s, m),
                                   u_nom=np.array([0.0]))
        u_val = float(u_opt.item() if hasattr(u_opt, 'item') else u_opt)
        x_dot = f_x + g_x.flatten() * u_val
        x = x + dt * x_dot
        x_hist[k + 1] = x
        u_hist[k] = u_val
        h_hist[k + 1] = acc_safety_h(x, D_safe, T_hw)
        V_hist[k + 1] = V(x)
        delta_hist[k] = info['delta']

    t = np.arange(0, n_steps + 1) * dt

    # Compare with pure CLF (no CBF)
    x_clf = x0.copy()
    x_clf_hist = np.zeros((n_steps + 1, 2))
    h_clf_hist = np.zeros(n_steps + 1)
    x_clf_hist[0] = x_clf
    h_clf_hist[0] = acc_safety_h(x_clf, D_safe, T_hw)

    x = x_clf.copy()
    for k in range(n_steps):
        f_x = acc_dynamics_f(x, v0)
        g_x = acc_dynamics_g(x, m)
        Lf_V = float(np.dot(V_grad(x), f_x))
        Lg_V = V_grad(x) @ g_x  # [1,]
        gamma = 1.5
        u = -(Lf_V + gamma * V(x)) / (np.dot(Lg_V, Lg_V) + 1e-10) * Lg_V[0]
        u = np.clip(u, -m * 9.0, m * 9.0)
        x_dot = f_x + g_x.flatten() * float(u)
        x = x + dt * x_dot
        x_clf_hist[k + 1] = x
        h_clf_hist[k + 1] = acc_safety_h(x, D_safe, T_hw)

    safe_cbf = np.all(h_hist >= -1e-6)
    safe_clf = np.all(h_clf_hist >= -1e-6)
    print(f"CLF-CBF-QP: h(x) >= 0? [{ {True: 'SAFE', False: 'UNSAFE'}[safe_cbf]}], "
          f"min h = {np.min(h_hist):.4f}")
    print(f"CLF only:   h(x) >= 0? [{ {True: 'SAFE', False: 'UNSAFE'}[safe_clf]}], "
          f"min h = {np.min(h_clf_hist):.4f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("CLF-CBF-QP: Stability + Safety for ACC", fontsize=14)

    axes[0, 0].plot(t, x_hist[:, 0], 'b-', linewidth=1.2)
    axes[0, 0].plot(t, x_clf_hist[:, 0], 'r--', linewidth=1.0)
    axes[0, 0].axhline(y=D_safe + T_hw * v0, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_ylabel("Distance [m]"); axes[0, 0].set_title("$D$ (CLF vs CLF-CBF)")

    axes[0, 1].plot(t, x_hist[:, 1], 'b-', linewidth=1.2, label='CLF-CBF')
    axes[0, 1].plot(t, x_clf_hist[:, 1], 'r--', linewidth=1.0, label='CLF only')
    axes[0, 1].axhline(y=v_des, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_ylabel("Speed [m/s]"); axes[0, 1].set_title("$v_{ego}$")
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(t[:-1], u_hist, 'g-', linewidth=0.8)
    axes[0, 2].set_ylabel("Force [N]"); axes[0, 2].set_title("Control $u$")

    axes[1, 0].plot(t, h_hist, 'b-', linewidth=1.2)
    axes[1, 0].plot(t, h_clf_hist, 'r--', linewidth=1.0)
    axes[1, 0].axhline(y=0, color='k', linestyle='--')
    axes[1, 0].set_ylabel("$h(x)$"); axes[1, 0].set_title("CBF (Safety)")

    axes[1, 1].plot(t, V_hist, 'm-', linewidth=1.2)
    axes[1, 1].set_ylabel("$V(x)$"); axes[1, 1].set_title("CLF (Stability)")

    axes[1, 2].plot(t[:-1], delta_hist, 'orange', linewidth=0.8)
    axes[1, 2].set_ylabel("$\\delta$"); axes[1, 2].set_title("CLF Slack (Relaxation)")
    axes[1, 2].set_yscale('log')

    plt.tight_layout()
    plt.savefig('control/cbf/fig_demo3_clf_cbf_qp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved fig_demo3_clf_cbf_qp.png")


# ===================================================================
# Run all
# ===================================================================

if __name__ == '__main__':
    demo_acc_cbf()
    demo_obstacle_avoidance()
    demo_clf_cbf_qp()
    print("\nAll CBF demos complete.")
