import numpy as np
from scipy.optimize import minimize


class MPCController:
    """Nonlinear MPC for kinematic bicycle model.

    State  x = [px, py, theta, v]
    Control u = [a, delta]

    Discrete dynamics (Euler, step dt):
        px'    = px + v * cos(theta) * dt
        py'    = py + v * sin(theta) * dt
        theta' = theta + (v / L) * tan(delta) * dt
        v'     = v + a * dt

    Cost:
        J = sum_{k=0}^{N-1} [ (x_k - r_k)^T Q (x_k - r_k) + u_k^T R u_k ]
            + (x_N - r_N)^T Q_f (x_N - r_N)
    """

    def __init__(self, dt=0.1, N=15, Q_diag=None, R_diag=None, L=2.5,
                 u_min=None, u_max=None, v_min=0.0, v_max=20.0,
                 delta_max=np.pi / 4):
        self.dt = dt
        self.N = N
        self.L = L

        if Q_diag is None:
            Q_diag = np.array([10.0, 10.0, 5.0, 1.0])
        if R_diag is None:
            R_diag = np.array([0.1, 0.1])
        self.Q = np.diag(Q_diag) if Q_diag.ndim == 1 else Q_diag
        self.R = np.diag(R_diag) if R_diag.ndim == 1 else R_diag
        self.Q_f = self.Q * 2.0

        if u_min is None:
            u_min = np.array([-3.0, -delta_max])
        if u_max is None:
            u_max = np.array([3.0, delta_max])
        self.u_min = np.asarray(u_min, dtype=float)
        self.u_max = np.asarray(u_max, dtype=float)
        self.v_min = v_min
        self.v_max = v_max

        self._prev_u = None
        self._solve_count = 0

    def _dynamics_step(self, x, u):
        """Single Euler step.  x: (4,), u: (2,) -> (4,)"""
        px, py, theta, v = x
        a, delta = u
        px_next = px + v * np.cos(theta) * self.dt
        py_next = py + v * np.sin(theta) * self.dt
        theta_next = theta + (v / self.L) * np.tan(delta) * self.dt
        v_next = np.clip(v + a * self.dt, self.v_min, self.v_max)
        return np.array([px_next, py_next, theta_next, v_next])

    def _rollout(self, x0, u_seq):
        """Roll out N steps.  x0: (4,), u_seq: (N,2) -> (N+1,4)"""
        N = u_seq.shape[0]
        x_seq = np.zeros((N + 1, 4))
        x_seq[0] = x0
        for k in range(N):
            x_seq[k + 1] = self._dynamics_step(x_seq[k], u_seq[k])
        return x_seq

    def _cost(self, u_flat, x0, x_ref):
        N = self.N
        u_seq = u_flat.reshape(N, 2)
        u_seq = np.clip(u_seq, self.u_min, self.u_max)
        x_seq = self._rollout(x0, u_seq)

        state_err = x_seq[:-1] - x_ref[:-1]  # (N, 4)
        stage_cost = np.sum(state_err * (state_err @ self.Q), axis=1)  # (N,)
        ctrl_cost = np.sum(u_seq * (u_seq @ self.R), axis=1)  # (N,)

        terminal_err = x_seq[-1] - x_ref[-1]
        terminal_cost = terminal_err @ self.Q_f @ terminal_err

        return float(np.sum(stage_cost) + np.sum(ctrl_cost) + terminal_cost)

    def solve(self, x0, x_ref):
        """Solve MPC for current state and reference horizon.

        Args:
            x0:    current state (4,)
            x_ref: reference states over horizon (N+1, 4)

        Returns:
            u_opt:  first optimal control (2,)
            u_seq:  full control sequence (N, 2)
            x_pred: predicted state trajectory (N+1, 4)
            info:   dict with solve status
        """
        N = self.N

        if self._prev_u is not None:
            u_init = np.vstack([self._prev_u[1:], self._prev_u[-1]])
        else:
            u_init = np.zeros((N, 2))

        bounds = []
        for _ in range(N):
            bounds.append((self.u_min[0], self.u_max[0]))
            bounds.append((self.u_min[1], self.u_max[1]))

        self._solve_count += 1
        result = minimize(
            self._cost, u_init.flatten(), args=(x0, x_ref),
            method='SLSQP', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-6, 'disp': False}
        )

        u_seq = result.x.reshape(N, 2)
        u_seq = np.clip(u_seq, self.u_min, self.u_max)
        x_pred = self._rollout(x0, u_seq)
        self._prev_u = u_seq

        info = {
            'success': result.success,
            'cost': float(result.fun),
            'nit': result.nit,
            'n_evals': self._solve_count,
        }
        return u_seq[0].copy(), u_seq, x_pred, info

    def reset(self):
        self._prev_u = None
        self._solve_count = 0

    def update_params(self, dt=None, N=None, Q_diag=None, R_diag=None,
                      L=None, u_min=None, u_max=None):
        """Update controller parameters at runtime. Resets warm-start."""
        if dt is not None:
            self.dt = dt
        if N is not None:
            self.N = N
        if Q_diag is not None:
            self.Q = np.diag(Q_diag) if Q_diag.ndim == 1 else Q_diag
            self.Q_f = self.Q * 2.0
        if R_diag is not None:
            self.R = np.diag(R_diag) if R_diag.ndim == 1 else R_diag
        if L is not None:
            self.L = L
        if u_min is not None:
            self.u_min = np.asarray(u_min, dtype=float)
        if u_max is not None:
            self.u_max = np.asarray(u_max, dtype=float)
        self.reset()
