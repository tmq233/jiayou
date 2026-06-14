"""
Lyapunov stability analysis demos.

Coverage:
  1. Linear system: Lyapunov equation, V(x) contours, phase portrait
  2. Pendulum: energy-based Lyapunov function, region of attraction
  3. Van der Pol oscillator: limit cycle, LaSalle's invariant set
  4. Inverted pendulum: CLF-based stabilization via Sontag's formula
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from lyapunov import (
    solve_lyapunov_eq,
    check_definiteness,
    quad_lyap_func,
    quad_lyap_deriv,
    simulate_system,
    sontag_formula,
    pendulum_dynamics,
    van_der_pol_dynamics,
    inverted_pendulum_f,
    inverted_pendulum_g,
)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})


# ===================================================================
# Demo 1: Linear system — Lyapunov equation & phase portrait
# ===================================================================

def demo_linear_system():
    """Lyapunov equation for ẋ = Ax. Show V(x)=xᵀPx and V̇(x)."""
    print("=" * 55)
    print("Demo 1: Linear System Lyapunov Stability")
    print("=" * 55)

    # Stable linear system (Hurwitz matrix)
    A = np.array([[-2.0, 1.0],
                  [-0.5, -1.0]])
    print(f"\nA =\n{A}")
    print(f"Eigenvalues: {np.linalg.eigvals(A)}")

    P = solve_lyapunov_eq(A)
    definiteness, lam_min = check_definiteness(P)
    print(f"\nP =\n{P}")
    print(f"P is {definiteness} (λ_min = {lam_min:.6f})")

    # Grid evaluation
    grid = np.linspace(-3, 3, 40)
    X, Y = np.meshgrid(grid, grid)
    V_grid = np.zeros_like(X)
    V_dot_grid = np.zeros_like(X)
    for i in range(len(grid)):
        for j in range(len(grid)):
            s = np.array([X[i, j], Y[i, j]])
            V_grid[i, j] = quad_lyap_func(s, P)
            V_dot_grid[i, j] = quad_lyap_deriv(s, A, P)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Linear System: $\\dot{x} = Ax$ — Lyapunov Analysis", fontsize=14)

    # V(x) contour
    c1 = axes[0].contourf(X, Y, V_grid, levels=20, cmap='Blues')
    axes[0].contour(X, Y, V_grid, levels=8, colors='navy', linewidths=0.5)
    plt.colorbar(c1, ax=axes[0], shrink=0.75)
    axes[0].set_title("$V(x) = x^T P x$")
    axes[0].set_xlabel("$x_1$"); axes[0].set_ylabel("$x_2$")

    # V̇(x) contour
    c2 = axes[1].contourf(X, Y, V_dot_grid, levels=20, cmap='Reds_r')
    axes[1].contour(X, Y, V_dot_grid, levels=8, colors='darkred', linewidths=0.5)
    plt.colorbar(c2, ax=axes[1], shrink=0.75)
    axes[1].set_title("$\\dot{V}(x) = x^T (A^T P + PA) x$ (≤ 0)")
    axes[1].set_xlabel("$x_1$"); axes[1].set_ylabel("$x_2$")

    # Phase portrait
    for s0_x in [-2, -1, 0, 1, 2]:
        for s0_y in [-2, -1, 0, 1, 2]:
            if s0_x == 0 and s0_y == 0:
                continue
            s0 = np.array([s0_x, s0_y])
            t_sim, x_sim = simulate_system(lambda x: A @ x, s0, (0, 5), 0.05)
            axes[2].plot(x_sim[:, 0], x_sim[:, 1], 'gray', alpha=0.5, linewidth=0.7)
    axes[2].plot(0, 0, 'ro', markersize=8)
    axes[2].set_title("Phase Portrait (origin = asymptotically stable)")
    axes[2].set_xlabel("$x_1$"); axes[2].set_ylabel("$x_2$")
    axes[2].set_xlim(-3, 3); axes[2].set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig('control/lyapunov/fig_demo1_linear.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved fig_demo1_linear.png")


# ===================================================================
# Demo 2: Pendulum — energy-based Lyapunov function
# ===================================================================

def demo_pendulum():
    """Pendulum: E(θ,θ̇) as natural Lyapunov function. Region of attraction."""
    print("\n" + "=" * 55)
    print("Demo 2: Pendulum — Energy as Lyapunov Function")
    print("=" * 55)

    g, L = 9.81, 1.0
    m = 1.0

    def energy(x):
        t, td = x[0], x[1]
        pe = m * g * L * (1.0 - np.cos(t))
        ke = 0.5 * m * L ** 2 * td ** 2
        return float(pe + ke)

    def energy_dot(x):
        _, td = x[0], x[1]
        return 0.0  # conservative system: Ė = 0

    # Phase portrait
    grid_theta = np.linspace(-np.pi, np.pi, 50)
    grid_omega = np.linspace(-8, 8, 50)
    Th, Om = np.meshgrid(grid_theta, grid_omega)
    E_grid = np.zeros_like(Th)
    for i in range(len(grid_theta)):
        for j in range(len(grid_omega)):
            E_grid[j, i] = energy(np.array([Th[j, i], Om[j, i]]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Pendulum: Energy-Based Lyapunov Stability", fontsize=14)

    # Energy contours + trajectories
    c = axes[0].contourf(Th, Om, E_grid, levels=30, cmap='viridis')
    axes[0].contour(Th, Om, E_grid, levels=15, colors='white', linewidths=0.3)
    plt.colorbar(c, ax=axes[0], shrink=0.75)
    axes[0].set_title("$E(\\theta, \\dot{\\theta}) = \\frac{1}{2}mL^2\\dot{\\theta}^2 + mgL(1-\\cos\\theta)$")

    # Trajectories
    for s0_theta in np.linspace(-2.5, 2.5, 8):
        for s0_omega in np.linspace(-4, 4, 4):
            if abs(s0_theta) < 0.1 and abs(s0_omega) < 0.1:
                continue
            t_sim, x_sim = simulate_system(lambda x: pendulum_dynamics(x, g, L, b=0.05),
                                           np.array([s0_theta, s0_omega]),
                                           (0, 8), 0.05)
            axes[0].plot(x_sim[:, 0], x_sim[:, 1], 'white', alpha=0.6, linewidth=0.6)
    axes[0].plot(0, 0, 'r*', markersize=10)
    axes[0].set_xlabel("$\\theta$"); axes[0].set_ylabel("$\\dot{\\theta}$")
    axes[0].set_xlim(-np.pi, np.pi); axes[0].set_ylim(-8, 8)

    # Region of attraction analysis
    E_levels = np.linspace(0, 25, 8)
    for level in E_levels:
        axes[1].cla()
        axes[1].contourf(Th, Om, E_grid, levels=30, cmap='viridis', alpha=0.6)
        axes[1].contour(Th, Om, E_grid, levels=[level], colors='red', linewidths=2)
        in_region = E_grid <= level
        axes[1].contourf(Th, Om, in_region, levels=[0.5, 1.0], colors=['none'],
                         hatches=['...'], alpha=0.3)
        axes[1].set_title(f"Sublevel Set: $E \\leq {level:.1f}$ "
                          f"— Region of Attraction")
        axes[1].set_xlabel("$\\theta$"); axes[1].set_ylabel("$\\dot{\\theta}$")
        axes[1].set_xlim(-np.pi, np.pi); axes[1].set_ylim(-8, 8)

    plt.tight_layout()
    plt.savefig('control/lyapunov/fig_demo2_pendulum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved fig_demo2_pendulum.png")


# ===================================================================
# Demo 3: Van der Pol — LaSalle's invariance principle
# ===================================================================

def demo_van_der_pol():
    """Van der Pol oscillator: limit cycle as invariant set."""
    print("\n" + "=" * 55)
    print("Demo 3: Van der Pol — LaSalle's Invariance Principle")
    print("=" * 55)

    mu = 1.5

    # Candidate Lyapunov: V(x) = ½(x₁² + x₂²)
    def V_ls(x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2)

    def V_dot_ls(x):
        return x[0] * x[1] + x[1] * (mu * (1 - x[0] ** 2) * x[1] - x[0])

    # Phase portrait
    grid = np.linspace(-4, 4, 50)
    X1, X2 = np.meshgrid(grid, grid)
    Vd_grid = np.zeros_like(X1)
    for i in range(len(grid)):
        for j in range(len(grid)):
            Vd_grid[i, j] = V_dot_ls(np.array([X1[i, j], X2[i, j]]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Van der Pol: LaSalle's Invariant Set = Limit Cycle", fontsize=14)

    # V̇ heatmap
    c = axes[0].contourf(X1, X2, Vd_grid, levels=30, cmap='RdBu_r')
    axes[0].contour(X1, X2, Vd_grid, levels=[0.0], colors='black', linewidths=2)
    plt.colorbar(c, ax=axes[0], shrink=0.75)
    axes[0].set_title("$\\dot{V}(x)$ — sign indefinite near origin")

    # Trajectories showing convergence to limit cycle
    initial_conditions = [
        (0.5, 0.0), (3.0, 3.0), (-3.0, -3.0), (-0.5, 5.0), (4.0, -2.0)
    ]
    for x0 in initial_conditions:
        t_sim, x_sim = simulate_system(lambda x: van_der_pol_dynamics(x, mu),
                                       np.array(x0), (0, 20), 0.05)
        axes[1].plot(x_sim[:, 0], x_sim[:, 1], alpha=0.8, linewidth=1.0,
                     label=f"$x_0=({x0[0]},{x0[1]})$")
        axes[1].plot(x_sim[0, 0], x_sim[0, 1], 'ko', markersize=4)
        axes[1].plot(x_sim[-1, 0], x_sim[-1, 1], 'r*', markersize=8)

    axes[1].set_title("All trajectories → unique limit cycle")
    axes[1].legend(fontsize=7, loc='upper right')
    axes[1].set_xlabel("$x_1$"); axes[1].set_ylabel("$x_2$")

    plt.tight_layout()
    plt.savefig('control/lyapunov/fig_demo3_van_der_pol.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved fig_demo3_van_der_pol.png")


# ===================================================================
# Demo 4: Inverted pendulum — CLF + Sontag's formula
# ===================================================================

def demo_clf_control():
    """Stabilize inverted pendulum at upright using CLF + Sontag's formula."""
    print("\n" + "=" * 55)
    print("Demo 4: CLF Stabilization — Sontag's Universal Formula")
    print("=" * 55)

    g_val, L, m, b = 9.81, 1.0, 1.0, 0.05

    # CLF: V = ½ kp (θ-π)² + ½ kd θ̇²  (error relative to upright)
    kp, kd = 15.0, 2.0
    theta_eq = np.pi  # upright equilibrium

    def V(x):
        e = x[0] - theta_eq
        return 0.5 * kp * e ** 2 + 0.5 * kd * x[1] ** 2

    def V_grad(x):
        e = x[0] - theta_eq
        return np.array([kp * e, kd * x[1]])

    # Sontag controller
    def clf_controller(x):
        u, info = sontag_formula(
            x,
            lambda s: inverted_pendulum_f(s, m, L, g_val, b),
            lambda s: inverted_pendulum_g(s, m, L),
            V, V_grad
        )
        return u[0], info

    # Simulate closed-loop
    x0 = np.array([np.pi - 0.8, 0.5])  # 0.8 rad from upright
    dt = 0.01
    n_steps = 800
    x_hist = np.zeros((n_steps + 1, 2))
    u_hist = np.zeros(n_steps)
    V_hist = np.zeros(n_steps + 1)
    x_hist[0] = x0
    V_hist[0] = V(x0)

    x = x0.copy()
    for k in range(n_steps):
        u_k, info = clf_controller(x)
        x_dot = inverted_pendulum_f(x, m, L, g_val, b) + \
                inverted_pendulum_g(x, m, L) @ np.array([u_k])
        x = x + dt * x_dot  # Euler
        x_hist[k + 1] = x
        u_hist[k] = u_k
        V_hist[k + 1] = V(x)

    t = np.arange(0, n_steps + 1) * dt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Inverted Pendulum: CLF + Sontag Universal Formula", fontsize=14)

    # State
    axes[0, 0].plot(t, x_hist[:, 0], 'b-', label='$\\theta$ (angle)')
    axes[0, 0].plot(t, x_hist[:, 1], 'r-', label='$\\dot{\\theta}$ (ang. vel)')
    axes[0, 0].axhline(y=np.pi, color='gray', linestyle='--', alpha=0.5,
                       label='upright ($\\pi$)')
    axes[0, 0].set_ylabel("State"); axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].legend(); axes[0, 0].set_title("State Trajectory")

    # Control
    axes[0, 1].plot(t[:-1], u_hist, 'g-')
    axes[0, 1].set_ylabel("Torque [Nm]"); axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_title("Control Input $u$")

    # V(x) over time
    axes[1, 0].plot(t, V_hist, 'm-', linewidth=1.5)
    axes[1, 0].set_ylabel("$V(x)$"); axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_title("Lyapunov Function $V(x)$ — monotonic decrease")

    # Phase portrait
    axes[1, 1].plot(x_hist[:, 0], x_hist[:, 1], 'k-', linewidth=0.8, alpha=0.7)
    axes[1, 1].plot(x_hist[0, 0], x_hist[0, 1], 'ko', markersize=8, label='start')
    axes[1, 1].plot(x_hist[-1, 0], x_hist[-1, 1], 'r*', markersize=12, label='end')
    axes[1, 1].axhline(y=0, color='gray', linewidth=0.5)
    axes[1, 1].axvline(x=np.pi, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel("$\\theta$"); axes[1, 1].set_ylabel("$\\dot{\\theta}$")
    axes[1, 1].set_title("Phase Portrait"); axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('control/lyapunov/fig_demo4_clf.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Check asymptotic stability
    V_decrease = np.diff(V_hist)
    monotonic = np.all(V_decrease <= 1e-8)
    print(f"  V(x0) = {V_hist[0]:.3f}, V(x_T) = {V_hist[-1]:.4f}")
    print(f"  V monotonic decrease: {monotonic}")
    print("[OK] Saved fig_demo4_clf.png")


# ===================================================================
# Run all
# ===================================================================

if __name__ == '__main__':
    demo_linear_system()
    demo_pendulum()
    demo_van_der_pol()
    demo_clf_control()
    print("\nAll Lyapunov demos complete.")
