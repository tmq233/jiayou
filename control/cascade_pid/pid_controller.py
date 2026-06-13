import numpy as np


class PIDController:
    """PID with derivative low-pass filter, output clamping, conditional-
    integration anti-windup."""

    def __init__(self, kp, ki, kd, dt, output_limits=(-np.inf, np.inf),
                 tau_f=0.005, deriv_on_meas=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_min, self.output_max = output_limits
        self.tau_f = tau_f
        self.deriv_on_meas = deriv_on_meas
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_meas = 0.0
        self.prev_deriv = 0.0

    def update(self, setpoint, measurement):
        error = setpoint - measurement

        p = self.kp * error

        # Derivative: on error (default) or on measurement (avoids kick)
        if self.deriv_on_meas:
            raw_deriv = -(measurement - self.prev_meas) / self.dt
        else:
            raw_deriv = (error - self.prev_error) / self.dt

        alpha = self.tau_f / (self.tau_f + self.dt)
        self.prev_deriv = alpha * self.prev_deriv + (1.0 - alpha) * raw_deriv
        d = self.kd * self.prev_deriv

        u_raw = p + self.integral + d
        u = np.clip(u_raw, self.output_min, self.output_max)

        # Conditional integration: freeze when saturated
        if u == u_raw:
            self.integral += self.ki * error * self.dt
        self.integral = np.clip(self.integral, self.output_min, self.output_max)

        self.prev_error = error
        self.prev_meas = measurement
        return u


class CascadePID:
    """Cascade PID: outer position → velocity ref → inner velocity → torque.

    Parameters
    ----------
    vel_ref_alpha : float
        EMA smoothing factor for velocity reference (1.0 = no filter).
        Lower values reduce noise propagation from outer to inner loop
        without compromising inner-loop disturbance rejection.
    """

    def __init__(self, outer_pid, inner_pid, vel_limit=np.inf,
                 vel_ref_alpha=1.0):
        self.outer = outer_pid
        self.inner = inner_pid
        self.vel_limit = vel_limit
        self.vel_ref_alpha = vel_ref_alpha
        self.vel_ref = 0.0

    def reset(self):
        self.outer.reset()
        self.inner.reset()
        self.vel_ref = 0.0

    def update_outer(self, pos_setpoint, pos_measurement):
        """Position loop: outputs EMA-filtered, clamped velocity reference."""
        vel_raw = self.outer.update(pos_setpoint, pos_measurement)
        vel_raw = np.clip(vel_raw, -self.vel_limit, self.vel_limit)
        self.vel_ref = (self.vel_ref_alpha * vel_raw +
                        (1.0 - self.vel_ref_alpha) * self.vel_ref)
        return self.vel_ref

    def update_inner(self, vel_measurement):
        """Velocity loop: outputs torque using the last vel_ref."""
        return self.inner.update(self.vel_ref, vel_measurement)
