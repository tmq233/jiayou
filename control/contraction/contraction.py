"""
Contraction Analysis & Incremental Stability.

Key result (Lohmiller & Slotine, 1998):
  A nonlinear system  ẋ = f(x, t)  is contracting with rate λ > 0 if
  there exists a metric  M(x, t) ≻ 0  such that the generalized Jacobian

      F = Ṁ + M·J + Jᵀ·M  ⪯  -2λ M

  where J = ∂f/∂x.  Then  d/dt ‖δx‖_M ≤ -λ ‖δx‖_M  (exponential convergence).
"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, eigvals


# ---------------------------------------------------------------------------
# Jacobian & generalized Jacobian
# ---------------------------------------------------------------------------

def jacobian_numerical(f, x, eps=1e-6):
    """Finite-difference Jacobian of vector field f at x.

    Args:
        f: callable f(x) -> [n,]
        x: [n,] state
        eps: perturbation size

    Returns:
        J: [n, n] Jacobian matrix
    """
    n = len(x)
    f0 = f(x)
    J = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        J[:, i] = (f(x + e) - f0) / eps
    return J


def generalized_jacobian(M, M_dot, J):
    """Compute generalized Jacobian: F = Ṁ + M·J + Jᵀ·M.

    Args:
        M: [n, n] metric at current state
        M_dot: [n, n] time derivative of metric
        J: [n, n] Jacobian ∂f/∂x

    Returns:
        F: [n, n] generalized Jacobian
    """
    return M_dot + M @ J + J.T @ M


def check_contraction_metric(F, M, lam=0.0, tol=1e-8):
    """Check if metric M makes the system contracting: F + 2λ·M ⪯ 0.

    Returns:
        is_contracting: bool
        lam_max: maximum generalized eigenvalue (should be ≤ 0)
    """
    # Solve generalized eigenvalue problem: F v = μ M v
    M_sym = (M + M.T) / 2.0
    eigvals_gen = eigvals(F, M_sym)
    lam_max = float(np.max(eigvals_gen.real))
    is_contracting = lam_max <= tol
    return is_contracting, lam_max


def contraction_metric_linear(A, Q=None):
    """For linear ẋ = Ax, solve AᵀM + MA = -Q for contraction metric.

    A stable A → M ≻ 0, F = -Q (constant).

    Args:
        A: [n, n] Hurwitz system matrix
        Q: [n, n] positive definite (default: I)

    Returns:
        M: [n, n] contraction metric
        F: [n, n] generalized Jacobian (constant)
    """
    n = A.shape[0]
    if Q is None:
        Q = np.eye(n)
    M = solve_continuous_lyapunov(A.T, -Q)
    F = A.T @ M + M @ A
    return M, F


# ---------------------------------------------------------------------------
# Simulation (shared utility)
# ---------------------------------------------------------------------------

def simulate_system(f, x0, t_span, dt):
    """RK4 integration. Returns (t, x_hist) of shape ([T], [T, n])."""
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / dt))
    t = np.linspace(t0, tf, n_steps + 1)
    n = len(x0)
    x_hist = np.zeros((n_steps + 1, n))
    x_hist[0] = x0
    x = np.array(x0, dtype=float)
    for i in range(n_steps):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_hist[i + 1] = x
    return t, x_hist


# ---------------------------------------------------------------------------
# Example dynamical systems
# ---------------------------------------------------------------------------

def fitzhugh_nagumo(x, a=0.7, b=0.8, tau=12.5):
    """FitzHugh-Nagumo neuron model.

    ẋ₁ = x₁ - x₁³/3 - x₂ + I_ext
    ẋ₂ = (1/τ)(x₁ + a - b·x₂)

    Contracting for τ sufficiently large (time-scale separation).
    """
    I_ext = 0.5
    x1, x2 = x[0], x[1]
    dx1 = x1 - x1 ** 3 / 3.0 - x2 + I_ext
    dx2 = (x1 + a - b * x2) / tau
    return np.array([dx1, dx2])


def coupled_oscillators(x, omega=np.array([1.0, 1.2]), K=2.0):
    """Two diffusively coupled phase oscillators.

    θ̇ᵢ = ωᵢ + K ∑ⱼ sin(θⱼ - θᵢ)

    The phase difference δ = θ₁ - θ₂ is contracting for K > |ω₁-ω₂|/2.
    """
    d1 = omega[0] + K * np.sin(x[1] - x[0])
    d2 = omega[1] + K * np.sin(x[0] - x[1])
    return np.array([d1, d2])


def simple_nonlinear(x):
    """ẋ₁ = -x₁ + x₂², ẋ₂ = -2x₂.  Contracting with identity metric."""
    return np.array([-x[0] + x[1] ** 2, -2.0 * x[1]])
