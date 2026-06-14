"""
Control Barrier Functions (CBF) for safety-critical control.

For affine system  ẋ = f(x) + g(x) u  and safe set  C = {x | h(x) ≥ 0}:

    h is a CBF if  ∃ u :  L_f h(x) + L_g h(x)·u + α(h(x)) ≥ 0

where α(·) is an extended class-K function.

CBF-QP:  min ‖u - u_nom‖²  s.t. CBF constraint + input bounds.
CLF-CBF-QP: combine stability (CLF, relaxed) + safety (CBF, hard).
"""

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# CBF constraint
# ---------------------------------------------------------------------------

class CBFConstraint:
    """Single CBF constraint for h(x) ≥ 0.

    Constraint:  ḣ(x, u) + α h(x) ≥ 0
      where ḣ = L_f h + L_g h · u

    Args:
        h:       callable h(x) -> float
        h_grad:  callable ∇h(x) -> [n,]
        alpha:   class-K function or scalar multiplier
    """

    def __init__(self, h, h_grad, alpha=1.0):
        self._h = h
        self._h_grad = h_grad
        if callable(alpha):
            self._alpha = alpha
        else:
            self._alpha = lambda r: float(alpha) * r

    def evaluate(self, x):
        """Return h(x), ∇h(x)."""
        return self._h(x), self._h_grad(x)

    def constraint_coeffs(self, x, f, g):
        """Return A_cbf, b_cbf such that A_cbf·u ≤ b_cbf encodes CBF.

        CBF:  L_f h + L_g h·u ≥ -α(h)  →  -(L_g h)·u ≤ L_f h + α(h)
        """
        hx = self._h(x)
        grad_h = self._h_grad(x)  # [n,]
        alpha_h = self._alpha(hx)

        Lf_h = float(np.dot(grad_h, f(x)))        # scalar
        Lg_h = grad_h @ g(x)                       # [m,]

        # Encode as:  -Lg_h · u  ≤ Lf_h + alpha_h
        A_cbf = -Lg_h.reshape(1, -1)               # [1, m]
        b_cbf = np.array([Lf_h + alpha_h])         # [1,]
        return A_cbf, b_cbf


# ---------------------------------------------------------------------------
# CBF-QP solver (min-norm, closed-form when unconstrained)
# ---------------------------------------------------------------------------

class CBFQPSolver:
    """Solves: min_u  ½‖u - u_nom‖²
               s.t.  A_cbf u ≤ b_cbf
                     u_min ≤ u ≤ u_max

    Uses closed-form KKT when single CBF and no active bounds.
    Falls back to SLSQP for general case.
    """

    def __init__(self, cbf_constraints, u_min=None, u_max=None):
        self.constraints = cbf_constraints
        self.u_min = u_min
        self.u_max = u_max

    def solve(self, x, f, g, u_nom=None):
        """Solve CBF-QP at state x.

        Args:
            x: [n,] current state
            f: callable f(x) -> [n,]
            g: callable g(x) -> [n, m]
            u_nom: [m,] nominal control

        Returns:
            u: [m,] safe control
            info: dict
        """
        m = g(x).shape[1]
        if u_nom is None:
            u_nom = np.zeros(m)

        # Gather all CBF constraints
        A_list, b_list = [], []
        for cbf in self.constraints:
            Ai, bi = cbf.constraint_coeffs(x, f, g)
            A_list.append(Ai)
            b_list.append(bi)

        A_cbf = np.vstack(A_list)  # [K, m]
        b_cbf = np.concatenate(b_list)  # [K,]

        # Check if u_nom is already safe
        violations = A_cbf @ u_nom - b_cbf
        if np.all(violations <= 1e-10):
            u_opt = self._clip(u_nom)
            return u_opt, {'status': 'nominal_safe', 'violations': violations}

        # Closed-form for single constraint (projection onto half-space)
        if A_cbf.shape[0] == 1:
            a = A_cbf[0]   # [m,]
            b_val = b_cbf[0]
            # u = u_nom - λ aᵀ  where λ = max(0, (a·u_nom - b) / ‖a‖²)
            a_dot_u = float(np.dot(a, u_nom))
            a_norm2 = float(np.dot(a, a))
            if a_norm2 > 1e-12 and a_dot_u > b_val:
                lam = (a_dot_u - b_val) / a_norm2
                u_opt = u_nom - lam * a
            else:
                u_opt = u_nom
            u_opt = self._clip(u_opt)
            return u_opt, {'status': 'projected', 'lambda': lam if 'lam' in dir() else 0.0}

        # General QP via SLSQP
        result = minimize(
            lambda u: 0.5 * np.sum((u - u_nom) ** 2),
            u_nom,
            method='SLSQP',
            constraints={'type': 'ineq', 'fun': lambda u: b_cbf - A_cbf @ u},
            bounds=[(self.u_min[i] if self.u_min is not None else -np.inf,
                     self.u_max[i] if self.u_max is not None else np.inf)
                    for i in range(m)],
            options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
        )
        return result.x, {'status': 'sqp', 'success': result.success}

    def _clip(self, u):
        u_c = np.copy(u)
        if self.u_min is not None:
            u_c = np.maximum(u_c, self.u_min)
        if self.u_max is not None:
            u_c = np.minimum(u_c, self.u_max)
        return u_c


# ---------------------------------------------------------------------------
# CLF-CBF-QP: Stability + Safety
# ---------------------------------------------------------------------------

class CLFCBFQPSolver:
    """Combined stability (CLF, relaxed) and safety (CBF, hard) QP.

    min  ½‖u - u_nom‖² + p_δ δ²
    s.t. L_f V + L_g V·u + γ V ≤ δ        (CLF, relaxed by slack δ)
         L_f h_i + L_g h_i·u + α_i h_i ≥ 0  (CBF i, hard)
         u_min ≤ u ≤ u_max
         δ ≥ 0
    """

    def __init__(self, V, V_grad, cbf_constraints,
                 gamma=1.0, slack_penalty=1e4,
                 u_min=None, u_max=None):
        self._V = V
        self._V_grad = V_grad
        self.cbf_constraints = cbf_constraints
        self.gamma = gamma
        self.slack_penalty = slack_penalty
        self.u_min = u_min
        self.u_max = u_max

    def solve(self, x, f, g, u_nom=None):
        m = g(x).shape[1]
        if u_nom is None:
            u_nom = np.zeros(m)

        # CLF coefficients
        Lf_V = float(np.dot(self._V_grad(x), f(x)))
        Lg_V = self._V_grad(x) @ g(x)  # [m,]
        gamma_V = self.gamma * self._V(x)

        # CBF coefficients
        A_cbf_list, b_cbf_list = [], []
        for cbf in self.cbf_constraints:
            Ai, bi = cbf.constraint_coeffs(x, f, g)
            A_cbf_list.append(Ai)
            b_cbf_list.append(bi)
        A_cbf = np.vstack(A_cbf_list)  # [K, m]
        b_cbf = np.concatenate(b_cbf_list)  # [K,]

        n_var = m + 1  # u (m dims) + delta (1 dim)
        x0 = np.concatenate([u_nom, [0.0]])

        def objective(z):
            u = z[:m]
            delta = z[m]
            return 0.5 * np.sum((u - u_nom) ** 2) + self.slack_penalty * delta ** 2

        constraints = []

        # CLF: Lf_V + Lg_V·u + gamma_V ≤ δ  →  δ - Lg_V·u ≥ Lf_V + gamma_V
        def clf_con(z):
            u = z[:m]
            delta = z[m]
            return delta - float(np.dot(Lg_V, u)) - (Lf_V + gamma_V)
        constraints.append({'type': 'ineq', 'fun': clf_con})

        # CBF: A_cbf u ≤ b_cbf  →  b_cbf - A_cbf u ≥ 0
        def cbf_con(z):
            u = z[:m]
            return b_cbf - A_cbf @ u
        constraints.append({'type': 'ineq', 'fun': cbf_con})

        # delta ≥ 0
        constraints.append({'type': 'ineq', 'fun': lambda z: z[m]})

        bounds = []
        for i in range(m):
            lo = self.u_min[i] if self.u_min is not None else -np.inf
            hi = self.u_max[i] if self.u_max is not None else np.inf
            bounds.append((lo, hi))
        bounds.append((0.0, None))  # delta ≥ 0

        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )

        u_opt = result.x[:m]
        delta_opt = result.x[m]
        return u_opt, {
            'success': result.success,
            'delta': delta_opt,
            'Lf_V': Lf_V,
            'Lg_V': Lg_V,
        }


# ---------------------------------------------------------------------------
# Exponential CBF (ECBF) for higher relative degree
# ---------------------------------------------------------------------------

class ECBFConstraint:
    """Exponential CBF for systems where h has relative degree r > 1.

    Constructs a new CBF ψ(x) of relative degree 1 via:
        ψ₀ = h
        ψᵢ = ψ̇ᵢ₋₁ + kᵢ ψᵢ₋₁   (i = 1, ..., r-1)

    Then enforces:  ψ̇_{r-1} + α ψ_{r-1} ≥ 0  (relative degree 1 in u).

    For obstacle avoidance (r=2):
        h(x) = ‖p - p_obs‖² - R²
        ψ(x) = ḣ + k₁ h = 2(p-p_obs)·v + k₁(‖p-p_obs‖² - R²)
    """

    def __init__(self, h, h_grad, h_dot_fun, grad_h_dot, k_vec=None, alpha=1.0):
        """
        Args:
            h:           callable h(x) -> float
            h_grad:      callable ∇h(x) -> [n,]
            h_dot_fun:   callable ḣ(x) = ∇h·f(x)  (drift part only)
            grad_h_dot:  callable ∇(ḣ)(x) -> [n,]  (full gradient, from f+g·u)
                         For r=2: ∂/∂x [∇h·(f+g·u)]  evaluated symbolically
            k_vec:       list of k values [k₁, ..., k_{r-1}]
            alpha:       class-K multiplier for the final barrier
        """
        self._h = h
        self._h_grad = h_grad
        self._h_dot_fun = h_dot_fun
        self._grad_h_dot = grad_h_dot
        self._k = k_vec if k_vec is not None else [1.0]
        if callable(alpha):
            self._alpha = alpha
        else:
            self._alpha = lambda r: float(alpha) * r

    def ecbf_value(self, x):
        """Compute ψ(x) (the reduced-relative-degree barrier)."""
        h_val = self._h(x)
        h_dot_val = self._h_dot_fun(x)
        return h_dot_val + self._k[0] * h_val

    def ecbf_grad(self, x):
        """Compute ∇ψ(x)."""
        grad_h = self._h_grad(x)
        grad_h_dot_val = self._grad_h_dot(x)
        return grad_h_dot_val + self._k[0] * grad_h

    def constraint_coeffs(self, x, f, g):
        """CBF constraint for ψ: L_f ψ + L_g ψ·u + α ψ ≥ 0."""
        psi_val = self.ecbf_value(x)
        grad_psi = self.ecbf_grad(x)
        alpha_psi = self._alpha(psi_val)

        Lf_psi = float(np.dot(grad_psi, f(x)))
        Lg_psi = grad_psi @ g(x)  # [m,]

        A_cbf = -Lg_psi.reshape(1, -1)
        b_cbf = np.array([Lf_psi + alpha_psi])
        return A_cbf, b_cbf


# ---------------------------------------------------------------------------
# Pre-built systems for demos
# ---------------------------------------------------------------------------

def acc_dynamics_f(x, v0=20.0):
    """Adaptive Cruise Control drift: ego car following lead car.

    State x = [D, v_ego]
      D = distance to lead car
      v_ego = ego speed

    Ẋ:  Ḋ = v_lead - v_ego  (assuming constant lead speed v0)
        v̇_ego = -F_r / m   (zero-throttle drift: rolling resistance)
    """
    m = 1650.0
    f0, f1, f2 = 0.1, 5.0, 0.25
    Fr = f0 + f1 * x[1] + f2 * x[1] ** 2
    return np.array([v0 - x[1], -Fr / m])


def acc_dynamics_g(x, m=1650.0):
    """ACC control matrix: u = wheel force, applied to v_ego."""
    return np.array([[0.0], [1.0 / m]])


def acc_safety_h(x, D_safe=10.0, T=1.8):
    """Safety function for ACC: h(x) = D - D_safe - T * v_ego ≥ 0.
    (safe distance = D_safe + time-headway * v_ego)
    """
    return float(x[0] - D_safe - T * x[1])


def acc_safety_h_grad(x, D_safe=10.0, T=1.8):
    return np.array([1.0, -T])


def obstacle_dynamics(x, use_cbf=False):
    """Double integrator 2D point mass — obstacle avoidance.
    State x = [px, py, vx, vy],  control u = [ax, ay].

    Returns f(x) (4,) without CBF, or (f,g) when use_cbf=True.
    """
    f_vec = np.array([x[2], x[3], 0.0, 0.0])
    if use_cbf:
        g_mat = np.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
        return f_vec, g_mat
    return f_vec


def obstacle_safety_h(x, obs_center=np.array([3.0, 2.0]), R=2.0):
    """Safety function: h(x) = ‖p - p_obs‖² - R² ≥ 0."""
    dp = x[:2] - obs_center
    return float(np.dot(dp, dp) - R ** 2)


def obstacle_safety_h_grad(x, obs_center=np.array([3.0, 2.0])):
    """∇h = [2(px - cx), 2(py - cy), 0, 0]."""
    grad = np.zeros(4)
    grad[:2] = 2.0 * (x[:2] - obs_center)
    return grad


def obstacle_h_dot_f(x, obs_center=np.array([3.0, 2.0])):
    """ḣ(x) = 2(p - p_obs)·v  (drift part: f contributes v, no acceleration)."""
    return float(2.0 * np.dot(x[:2] - obs_center, x[2:4]))


def obstacle_h_dot_grad(x, obs_center=np.array([3.0, 2.0])):
    """∇(ḣ)(x) = ∂/∂x [2(p-p_obs)·v].
    = [2vx, 2vy, 2(px-cx), 2(py-cy)]  (∂/∂p, ∂/∂v)
    """
    grad = np.zeros(4)
    grad[:2] = 2.0 * x[2:4]          # ∂/∂p: 2v
    grad[2:4] = 2.0 * (x[:2] - obs_center)  # ∂/∂v: 2(p-p_obs)
    return grad
