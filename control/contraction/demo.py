"""
Contraction Analysis demos.

Coverage:
  1. Linear system: contraction metric via Lyapunov equation
  2. Simple nonlinear: exponential convergence to unique equilibrium
  3. FitzHugh-Nagumo: contraction with time-scale separation
  4. Coupled oscillators: partial contraction (synchronization)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from contraction import (
    jacobian_numerical,
    generalized_jacobian,
    check_contraction_metric,
    contraction_metric_linear,
    simulate_system,
    fitzhugh_nagumo,
    coupled_oscillators,
    simple_nonlinear,
)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})


# ===================================================================
# Demo 1: Linear contraction metric
# ===================================================================

def demo_linear_contraction():
    """Find contraction metric M for ẋ = Ax, verify F = -I."""
    print("=" * 55)
    print("Demo 1: Linear Contraction Metric")
    print("=" * 55)

    A = np.array([[-3.0, 2.0],
                  [-1.0, -1.5]])
    M, F = contraction_metric_linear(A)
    print(f"A =\n{A}")
    print(f"M =\n{M}")
    print(f"F = AᵀM + MA =\n{F}")

    # Verify contraction
    is_ctr, lam = check_contraction_metric(F, M)
    print(f"Contraction check: {is_ctr}, λ_max(F, M) = {lam:.6f}")

    # Simulate 3 initial conditions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Linear Contraction: $\\dot{x} = Ax$", fontsize=14)

    colors = ['b', 'r', 'g']
    for c_idx, x0 in enumerate([(3, 3), (-3, 2), (1, -3)]):
        t, x = simulate_system(lambda s: A @ s, np.array(x0), (0, 3), 0.01)
        ax1.plot(t, x[:, 0], color=colors[c_idx], alpha=0.7,
                 label=f'$x_1(0)={x0[0]}$')
        ax1.plot(t, x[:, 1], color=colors[c_idx], alpha=0.3, linestyle='--',
                 label=f'$x_2(0)={x0[1]}$')
        ax2.plot(x[:, 0], x[:, 1], color=colors[c_idx], alpha=0.7)

    ax1.set_xlabel("t"); ax1.set_ylabel("State")
    ax1.set_title("State Trajectories"); ax1.legend(fontsize=7, ncol=2)
    ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
    ax2.set_title("Phase Portrait"); ax2.plot(0, 0, 'k*', markersize=10)

    plt.tight_layout()
    plt.savefig('control/contraction/fig_demo1_linear.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved fig_demo1_linear.png")


# ===================================================================
# Demo 2: Nonlinear contraction
# ===================================================================

def demo_nonlinear_contraction():
    """ẋ₁ = -x₁ + x₂², ẋ₂ = -2x₂. Verify contraction with I metric on a grid."""
    print("\n" + "=" * 55)
    print("Demo 2: Nonlinear Contraction — Exponential Convergence")
    print("=" * 55)

    f = simple_nonlinear

    # Check contraction on a grid
    grid = np.linspace(-3, 3, 20)
    all_contracting = True
    max_lam = -np.inf
    for x1 in grid:
        for x2 in grid:
            x = np.array([x1, x2])
            J = jacobian_numerical(f, x)
            M = np.eye(2)
            M_dot = np.zeros((2, 2))
            F = generalized_jacobian(M, M_dot, J)
            is_ctr, lam = check_contraction_metric(F, M)
            if not is_ctr:
                all_contracting = False
            max_lam = max(max_lam, lam)

    print(f"Grid check: all contracting = {all_contracting}")
    print(f"Worst λ_max(F+2λM) over grid = {max_lam:.6f}")

    # J(x) = [[-1, 2x₂], [0, -2]]
    # F = J + Jᵀ = [[-2, 2x₂], [2x₂, -4]]
    # det(F - λI) = (-2-λ)(-4-λ) - 4x₂² = λ² + 6λ + 8 - 4x₂²
    # λ = -3 ± √(1 + 4x₂²)  < 0  for  √(1 + 4x₂²) < 3  →  |x₂| < √2 ≈ 1.414
    print("Analytic: F ≺ 0 on strip |x₂| < sqrt(2) ≈ 1.414")

    # Simulate convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Nonlinear Contraction: $\\dot{x}_1 = -x_1 + x_2^2,\\; \\dot{x}_2 = -2x_2$",
                 fontsize=13)

    for x0 in [(3, 3), (-3, 2), (1, -3), (-2.5, -2.5)]:
        t, x = simulate_system(f, np.array(x0), (0, 4), 0.02)
        axes[0].plot(t, np.sqrt(x[:, 0]**2 + x[:, 1]**2),
                     alpha=0.8, label=f'$x_0=({x0[0]},{x0[1]})$')
        axes[1].plot(x[:, 0], x[:, 1], alpha=0.7)

    axes[0].set_yscale('log')
    axes[0].set_xlabel("t"); axes[0].set_ylabel("$\\|x\\|_2$")
    axes[0].set_title("Exponential Convergence (log scale)")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("$x_1$"); axes[1].set_ylabel("$x_2$")
    axes[1].set_title("Phase Portrait"); axes[1].plot(0, 0, 'k*', markersize=10)

    plt.tight_layout()
    plt.savefig('control/contraction/fig_demo2_nonlinear.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved fig_demo2_nonlinear.png")


# ===================================================================
# Demo 3: FitzHugh-Nagumo — contraction via time-scale separation
# ===================================================================

def demo_fitzhugh_nagumo():
    """FitzHugh-Nagumo neuron: verify contraction numerically."""
    print("\n" + "=" * 55)
    print("Demo 3: FitzHugh-Nagumo — Contraction via Timescale Separation")
    print("=" * 55)

    # Check Jacobian eigenvalues near limit cycle
    t, x_lc = simulate_system(fitzhugh_nagumo, np.array([0.0, 0.0]),
                               (0, 100), 0.05)
    x_traj = x_lc[-500:, :]  # last points (near limit cycle)

    # Check symmetric part of Jacobian along trajectory
    lam_maxes = []
    for x in x_traj[::20]:
        J = jacobian_numerical(fitzhugh_nagumo, x)
        J_sym = 0.5 * (J + J.T)
        lam_maxes.append(float(np.max(eigvals(J_sym).real)))

    avg_lam = np.mean(lam_maxes)
    print(f"Mean λ_max(J_sym) along limit cycle: {avg_lam:.6f}")
    print(f"System: { 'contracting' if avg_lam < 0 else 'NOT uniformly contracting' }")

    # Phase portrait with multiple initial conditions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("FitzHugh-Nagumo Neuron", fontsize=14)

    for x0 in [(2, 0), (-2, 2), (0, -2), (1.5, 1.5)]:
        t, x = simulate_system(fitzhugh_nagumo, np.array(x0), (0, 60), 0.05)
        ax1.plot(x[:, 0], x[:, 1], alpha=0.6, linewidth=0.8)
        ax1.plot(x[0, 0], x[0, 1], 'ko', markersize=4)

    # Highlight limit cycle
    ax1.plot(x_traj[:, 0], x_traj[:, 1], 'r-', linewidth=2.0, label='limit cycle')
    ax1.set_xlabel("$v$ (membrane potential)"); ax1.set_ylabel("$w$ (recovery)")
    ax1.set_title("Phase Portrait");
    ax1.legend(fontsize=9)

    # lambda_max along trajectory
    ax2.plot(lam_maxes, 'b-', alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Sample index"); ax2.set_ylabel("$\\lambda_{\\max}(J_{sym})$")
    ax2.set_title("Uniform Negative Definiteness Check")

    plt.tight_layout()
    plt.savefig('control/contraction/fig_demo3_fhn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved fig_demo3_fhn.png")


# ===================================================================
# Demo 4: Coupled oscillators — partial contraction (synchronization)
# ===================================================================

def demo_coupled_oscillators():
    """Two coupled phase oscillators: contraction in the difference variable."""
    print("\n" + "=" * 55)
    print("Demo 4: Coupled Oscillators — Partial Contraction")
    print("=" * 55)

    omega = np.array([1.0, 1.2])
    K_vals = [0.05, 0.3, 2.0]
    K_crit = abs(omega[0] - omega[1]) / 2.0
    print(f"Critical coupling K_crit = |ω₁-ω₂|/2 = {K_crit:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Coupled Phase Oscillators: $\\dot{\\theta}_i = \\omega_i + K \\sum \\sin(\\theta_j-\\theta_i)$",
                 fontsize=13)

    for idx, K in enumerate(K_vals):
        ax = axes[idx]
        x0_list = [(0.0, 1.5), (2.0, 0.5), (4.0, 2.0)]
        for x0 in x0_list:
            t, x = simulate_system(
                lambda s: coupled_oscillators(s, omega=omega, K=K),
                np.array(x0), (0, 30), 0.02)
            delta = x[:, 0] - x[:, 1]
            delta_mod = np.mod(delta + np.pi, 2 * np.pi) - np.pi
            ax.plot(t, delta_mod, alpha=0.7, linewidth=0.8,
                    label=f'$\\Delta\\theta(0)={x0[0]-x0[1]:.1f}$')

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel("t")
        ax.set_ylabel("$\\Delta\\theta$ (mod $\\pi$)")
        status = "sync" if K > K_crit else "no sync"
        ax.set_title(f"K = {K} ({status})" + (" [contracting]" if K > K_crit else ""))
        if idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('control/contraction/fig_demo4_coupled.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved fig_demo4_coupled.png")


# ===================================================================
# Run all
# ===================================================================

if __name__ == '__main__':
    demo_linear_contraction()
    demo_nonlinear_contraction()
    demo_fitzhugh_nagumo()
    demo_coupled_oscillators()
    print("\nAll Contraction demos complete.")
