"""
Lyapunov Stability Analysis for nonlinear affine systems.

Covers:
  - Lyapunov direct method (linear & nonlinear)
  - Lyapunov equation for linear systems
  - Control Lyapunov Functions (CLF) & Sontag's universal formula
  - LaSalle's invariance principle
"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, eigvals


# ---------------------------------------------------------------------------
# Linear system Lyapunov analysis
# ---------------------------------------------------------------------------

def solve_lyapunov_eq(A, Q=None):
    """Solve Aᵀ P + P A = -Q for P.

    Args:
        A: [n, n] system matrix (must be Hurwitz for P ≻ 0)
        Q: [n, n] positive definite matrix (default: I)

    Returns:
        P: [n, n] solution
    """
    n = A.shape[0]
    if Q is None:
        Q = np.eye(n)
    return solve_continuous_lyapunov(A.T, -Q)


def check_definiteness(P, tol=1e-10):
    """Check definiteness of symmetric matrix P via eigenvalues.

    Returns:
        str: 'positive definite', 'positive semi-definite', 'negative definite',
             'negative semi-definite', 'indefinite'
        float: minimum eigenvalue
    """
    P_sym = (P + P.T) / 2.0
    eigs = np.sort(eigvals(P_sym).real)
    lam_min, lam_max = eigs[0], eigs[-1]

    if lam_min > tol:
        return 'positive definite', lam_min
    elif lam_min >= -tol:
        return 'positive semi-definite', lam_min
    elif lam_max < -tol:
        return 'negative definite', lam_max
    elif lam_max <= tol:
        return 'negative semi-definite', lam_max
    return 'indefinite', lam_min if abs(lam_min) > abs(lam_max) else lam_max


def quad_lyap_func(x, P):
    """V(x) = xᵀ P x  (quadratic Lyapunov candidate).

    Args:
        x: [n,] state vector
        P: [n, n] positive definite matrix

    Returns:
        V: scalar
    """
    return float(x @ P @ x)


def quad_lyap_deriv(x, A, P):
    """V̇(x) = xᵀ (AᵀP + PA) x for ẋ = Ax.

    Args:
        x: [n,] state
        A: [n, n] system matrix
        P: [n, n] Lyapunov matrix

    Returns:
        V_dot: scalar
    """
    AP = A.T @ P + P @ A
    return float(x @ AP @ x)


# ---------------------------------------------------------------------------
# Nonlinear system utilities
# ---------------------------------------------------------------------------

def lie_derivative(V_grad, f, x):
    """Lie derivative L_f V = ∇V(x) · f(x).

    Args:
        V_grad: callable ∇V(x) -> [n,]
        f: callable f(x) -> [n,]
        x: [n,] state

    Returns:
        L_f_V: scalar
    """
    return float(np.dot(V_grad(x), f(x)))


def simulate_system(f, x0, t_span, dt, method='rk4'):
    """Simulate ODE dx/dt = f(x) using fixed-step integrator.

    Args:
        f: callable f(x) -> [n,]
        x0: [n,] initial state
        t_span: (t_start, t_end)
        dt: time step
        method: 'euler' or 'rk4'

    Returns:
        t: [T,] time points
        x_hist: [T, n] state trajectory
    """
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / dt))
    t = np.linspace(t0, tf, n_steps + 1)
    n = len(x0)
    x_hist = np.zeros((n_steps + 1, n))
    x_hist[0] = x0
    x = np.array(x0, dtype=float)

    if method == 'euler':
        for i in range(n_steps):
            x = x + dt * f(x)
            x_hist[i + 1] = x
    elif method == 'rk4':
        for i in range(n_steps):
            k1 = f(x)
            k2 = f(x + 0.5 * dt * k1)
            k3 = f(x + 0.5 * dt * k2)
            k4 = f(x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            x_hist[i + 1] = x
    else:
        raise ValueError(f"Unknown method: {method}")

    return t, x_hist


# ---------------------------------------------------------------------------
# Control Lyapunov Function (CLF)
# ---------------------------------------------------------------------------

def sontag_formula(x, f, g, V, V_grad):
    """Sontag's universal formula for CLF-based stabilization.

    For affine system  ẋ = f(x) + g(x) u,  given a CLF V(x):

        a(x) = L_f V(x)    (∇V · f)
        b(x) = L_g V(x)    (∇V · g)

        u(x) =                          0                         if b(x) = 0
               - (a(x) + √(a(x)² + ‖b(x)‖⁴)) / ‖b(x)‖²  * b(x)ᵀ  if b(x) ≠ 0

    Guarantees: V̇ ≤ -√(a² + ‖b‖⁴) < 0  (asymptotic stability)

    Args:
        x: [n,] state
        f: callable f(x) -> [n,]  (drift vector field)
        g: callable g(x) -> [n, m]  (control matrix)
        V: callable V(x) -> float  (CLF)
        V_grad: callable ∇V(x) -> [n,]  (gradient of V)

    Returns:
        u: [m,] control input
        info: dict with a, b, denominator
    """
    a = float(np.dot(V_grad(x), f(x)))
    b = V_grad(x) @ g(x)  # [m,]
    b_norm2 = float(np.dot(b, b))  # ‖b‖²

    if b_norm2 < 1e-12:
        return np.zeros_like(b), {'a': a, 'b': b, 'method': 'zero_b'}

    # Stabilizing term: ϕ(x) when a > 0 for aggressive stabilization
    phi = a + np.sqrt(a ** 2 + b_norm2 ** 2)
    u = -(phi / b_norm2) * b
    return u, {'a': a, 'b': b, 'phi': phi, 'b_norm2': b_norm2, 'method': 'sontag'}


def clf_qp(x, f, g, V, V_grad, u_nominal=None, u_min=None, u_max=None,
           slack_penalty=1e4):
    """CLF constraint enforced via Quadratic Program (min-norm).

    min ½‖u - u_nom‖² + penalty * δ²
    s.t. L_f V + L_g V · u ≤ -γ V(x) + δ
         u_min ≤ u ≤ u_max
         δ ≥ 0  (slack)

    Uses closed-form solution when no bounds (simplified, no QP solver needed).

    Args:
        x: [n,] state
        f: callable f(x) -> [n,]
        g: callable g(x) -> [n, m]
        V: callable V(x) -> float
        V_grad: callable ∇V(x) -> [n,]
        u_nominal: [m,] nominal control (default: zero)
        u_min, u_max: [m,] control bounds
        slack_penalty: penalty for violating CLF condition

    Returns:
        u: [m,] control
        info: dict
    """
    a = float(np.dot(V_grad(x), f(x)))  # L_f V
    b = V_grad(x) @ g(x)  # L_g V  [m,]
    gamma = 1.0
    V_val = V(x)

    m = len(b)
    if u_nominal is None:
        u_nominal = np.zeros(m)

    # CLF condition: a + b·u ≤ -γ V  →  b·u ≤ -γ V - a
    # Project u_nom onto half-space defined by CLF constraint
    # Using formula for projection onto a half-space
    rhs = -gamma * V_val - a
    violation = float(np.dot(b, u_nominal)) - rhs  # >0 means constraint violated

    if violation <= 0:
        u = u_nominal
        slack = 0.0
    else:
        # Closed-form projection: u = u_nom - (b·u_nom - rhs)/‖b‖² * b
        b_norm2 = float(np.dot(b, b))
        if b_norm2 > 1e-12:
            u = u_nominal - (violation / b_norm2) * b
        else:
            u = u_nominal
        slack = 0.0

    # Clip to bounds
    if u_min is not None:
        u = np.clip(u, u_min, u_max)
    if u_max is not None:
        u = np.clip(u, u_min if u_min is not None else -np.inf, u_max)

    V_dot = a + float(np.dot(b, u))
    return u, {
        'a': a, 'b': b, 'V': V_val, 'V_dot': V_dot, 'slack': slack
    }


# ---------------------------------------------------------------------------
# Common nonlinear system dynamics
# ---------------------------------------------------------------------------

def pendulum_dynamics(x, g=9.81, L=1.0, b=0.0):
    """Undamped pendulum: θ̈ = -(g/L) sin(θ) - b θ̇.
    State x = [θ, θ̇].
    """
    theta, theta_dot = x[0], x[1]
    theta_ddot = -(g / L) * np.sin(theta) - b * theta_dot
    return np.array([theta_dot, theta_ddot])


def van_der_pol_dynamics(x, mu=1.0):
    """Van der Pol oscillator: ẋ₁ = x₂, ẋ₂ = μ(1 - x₁²)x₂ - x₁."""
    return np.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


def inverted_pendulum_f(x, m=1.0, L=1.0, g=9.81, b=0.1):
    """Inverted pendulum drift dynamics f(x). State x = [θ, θ̇]."""
    theta, theta_dot = x[0], x[1]
    theta_ddot = (g / L) * np.sin(theta) - (b / (m * L ** 2)) * theta_dot
    return np.array([theta_dot, theta_ddot])


def inverted_pendulum_g(x, m=1.0, L=1.0):
    """Inverted pendulum control matrix g(x)."""
    return np.array([[0.0], [1.0 / (m * L ** 2)]])
