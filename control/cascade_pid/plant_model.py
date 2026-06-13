import numpy as np


class DCMotorPlant:
    """Second-order plant modelling a DC motor + inertial load.

    States:  x = [theta (rad), omega (rad/s)]
    Dynamics:
        dtheta/dt = omega
        domega/dt = (K * u - B * omega - tau_dist) / J

    Parameters
    ----------
    J : float  — moment of inertia          (kg.m^2)
    B : float  — viscous friction coeff     (Nm.s/rad)
    K : float  — torque constant            (Nm per control unit)
    max_torque : float  — input saturation  (Nm)
    """

    def __init__(self, J=0.1, B=0.1, K=1.0, max_torque=5.0,
                 pos0=0.0, vel0=0.0):
        self.J = J
        self.B = B
        self.K = K
        self.max_torque = max_torque
        self.position = pos0
        self.velocity = vel0

    @property
    def state(self):
        return np.array([self.position, self.velocity])

    def set_params(self, J=None, B=None, K=None):
        """Online parameter change (simulates load/inertia variation)."""
        if J is not None:
            self.J = J
        if B is not None:
            self.B = B
        if K is not None:
            self.K = K

    def dynamics(self, state, u, tau_dist=0.0):
        """Returns [dtheta/dt, domega/dt]."""
        pos, vel = state
        u_clamped = np.clip(u, -self.max_torque, self.max_torque)
        dpos = vel
        dvel = (self.K * u_clamped - self.B * vel - tau_dist) / self.J
        return np.array([dpos, dvel])

    def step(self, u, tau_dist, dt):
        """RK4 integration of one time step."""
        x = self.state
        k1 = self.dynamics(x, u, tau_dist)
        k2 = self.dynamics(x + 0.5 * dt * k1, u, tau_dist)
        k3 = self.dynamics(x + 0.5 * dt * k2, u, tau_dist)
        k4 = self.dynamics(x + dt * k3, u, tau_dist)
        x_new = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.position, self.velocity = x_new
