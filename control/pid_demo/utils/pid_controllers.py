"""
PID Controller Variants — 位置式 / 增量式 / 抗积分饱和 / 死区

All forms use the standard parallel PID structure:
    u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de/dt
"""

import numpy as np


class PIDPositional:
    """Positional (absolute) PID controller.

    u(k) = Kp·e(k) + Ki·Ts·Σe(i) + Kd·[e(k) − e(k−1)] / Ts

    Integral accumulates ALL past errors → windup risk when actuator saturates.
    """

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, Ts=0.01,
                 u_min=-np.inf, u_max=np.inf):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.Ts = Ts
        self.u_min, self.u_max = u_min, u_max
        self.reset()

    def reset(self):
        self.e_prev = 0.0
        self.integral = 0.0       # Σ e(i)·Ts
        self.u_prev = 0.0
        self.setpoint = 0.0

    def update(self, y: float, r: float = None) -> float:
        if r is not None:
            self.setpoint = r
        e = self.setpoint - y

        P = self.Kp * e
        self.integral += e * self.Ts
        I = self.Ki * self.integral
        D = self.Kd * (e - self.e_prev) / self.Ts if self.Ts > 0 else 0.0

        u_raw = P + I + D
        u = np.clip(u_raw, self.u_min, self.u_max)

        self.e_prev = e
        self.u_prev = u
        return u

    def set_setpoint(self, r: float):
        self.setpoint = r


class PIDIncremental:
    """Incremental (velocity) PID controller.

    Δu(k) = Kp·[e(k)−e(k−1)] + Ki·Ts·e(k)
            + Kd·[e(k)−2e(k−1)+e(k−2)] / Ts
    u(k)  = u(k−1) + Δu(k)

    Natural anti-windup: no explicit Σe accumulator. Clamping u(k) automatically
    stops windup. Bumpless transfer: only increments are computed each step.
    """

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, Ts=0.01,
                 u_min=-np.inf, u_max=np.inf):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.Ts = Ts
        self.u_min, self.u_max = u_min, u_max
        self.reset()

    def reset(self):
        self.e_prev = 0.0
        self.e_prev2 = 0.0
        self.u_prev = 0.0
        self.setpoint = 0.0

    def update(self, y: float, r: float = None) -> float:
        if r is not None:
            self.setpoint = r
        e = self.setpoint - y

        dP = self.Kp * (e - self.e_prev)
        dI = self.Ki * self.Ts * e
        dD = self.Kd * (e - 2 * self.e_prev + self.e_prev2) / self.Ts if self.Ts > 0 else 0.0

        u_raw = self.u_prev + dP + dI + dD
        u = np.clip(u_raw, self.u_min, self.u_max)

        self.e_prev2 = self.e_prev
        self.e_prev = e
        self.u_prev = u
        return u

    def set_setpoint(self, r: float):
        self.setpoint = r


class PIDAntiWindupClamping:
    """Positional PID with conditional integration (clamping anti-windup).

    Rule: Stop integrating when actuator is saturated in the SAME direction:
      - u at u_max AND e > 0  →  STOP (would push further up)
      - u at u_min AND e < 0  →  STOP (would push further down)
      - Otherwise: integrate normally
    """

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, Ts=0.01,
                 u_min=-np.inf, u_max=np.inf):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.Ts = Ts
        self.u_min, self.u_max = u_min, u_max
        self.reset()

    def reset(self):
        self.e_prev = 0.0
        self.integral = 0.0
        self.u_prev = 0.0
        self.setpoint = 0.0

    def update(self, y: float, r: float = None) -> float:
        if r is not None:
            self.setpoint = r
        e = self.setpoint - y

        P = self.Kp * e
        D = self.Kd * (e - self.e_prev) / self.Ts if self.Ts > 0 else 0.0

        u_pre = P + self.Ki * self.integral + D

        # Conditional integration
        integrate = True
        if u_pre >= self.u_max and e > 0:
            integrate = False
        elif u_pre <= self.u_min and e < 0:
            integrate = False

        if integrate:
            self.integral += e * self.Ts

        I = self.Ki * self.integral
        u = np.clip(P + I + D, self.u_min, self.u_max)

        self.e_prev = e
        self.u_prev = u
        return u

    def set_setpoint(self, r: float):
        self.setpoint = r


class PIDDeadzone:
    """PID with dead-zone on error input.

    if |e| < dead_zone:  e_eff = 0     (ignore tiny errors)
    else:                e_eff = e − sign(e) * dead_zone

    Suppresses limit-cycle oscillations caused by mechanical backlash,
    sensor quantization, or Coulomb friction.
    """

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, Ts=0.01,
                 dead_zone=0.05, u_min=-np.inf, u_max=np.inf):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.Ts = Ts
        self.dead_zone = dead_zone
        self.u_min, self.u_max = u_min, u_max
        self.reset()

    def reset(self):
        self.e_prev = 0.0
        self.integral = 0.0
        self.u_prev = 0.0
        self.setpoint = 0.0

    def update(self, y: float, r: float = None) -> float:
        if r is not None:
            self.setpoint = r
        e_raw = self.setpoint - y

        if abs(e_raw) < self.dead_zone:
            e = 0.0
        else:
            e = e_raw - np.sign(e_raw) * self.dead_zone

        P = self.Kp * e
        self.integral += e * self.Ts
        I = self.Ki * self.integral
        D = self.Kd * (e - self.e_prev) / self.Ts if self.Ts > 0 else 0.0

        u = np.clip(P + I + D, self.u_min, self.u_max)

        self.e_prev = e
        self.u_prev = u
        return u

    def set_setpoint(self, r: float):
        self.setpoint = r
