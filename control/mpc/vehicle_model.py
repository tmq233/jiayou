import numpy as np


class VehicleModel:
    """Kinematic bicycle model with optional process noise.

    State  x = [px, py, theta, v]
    Control u = [a, delta]
    """

    def __init__(self, dt=0.1, L=2.5, v_min=0.0, v_max=20.0):
        self.dt = dt
        self.L = L
        self.v_min = v_min
        self.v_max = v_max
        self.state = np.zeros(4)
        self.noise_std = np.array([0.0, 0.0, 0.0, 0.0])

    def _dynamics(self, x, u):
        px, py, theta, v = x
        a, delta = u
        px_next = px + v * np.cos(theta) * self.dt
        py_next = py + v * np.sin(theta) * self.dt
        theta_next = theta + (v / self.L) * np.tan(delta) * self.dt
        v_next = np.clip(v + a * self.dt, self.v_min, self.v_max)
        return np.array([px_next, py_next, theta_next, v_next])

    def step(self, u):
        self.state = self._dynamics(self.state, np.asarray(u))
        noise = np.random.randn(4) * self.noise_std
        self.state += noise
        return self.state.copy()

    def reset(self, state=None):
        if state is not None:
            self.state = np.asarray(state, dtype=float).copy()
        else:
            self.state = np.zeros(4)

    def set_noise(self, std_px=0.0, std_py=0.0, std_theta=0.0, std_v=0.0):
        self.noise_std = np.array([std_px, std_py, std_theta, std_v])
