"""Four comparison scenarios demonstrating cascade PID advantages.

Plant:  G(s) = K / [s (Js + B)],   J=0.05, B=0.05, K=1.0
Open-loop:  ω_n ≈ 4.47 rad/s,  ζ ≈ 0.11  (severely under-damped)

Tuning rationale
----------------
Single PID — binomial pole placement, three poles at ω₀ = 5 rad/s:
    J s³ + (B + Kd) s² + Kp s + Ki = 0
    →  Kp=3.75, Ki=6.25, Kd=0.70

Cascade inner velocity PI — binomial, ω_v = 25 rad/s:
    J s² + (B + Kpv) s + Kiv = 0
    →  Kpv=2.45, Kiv=31.25

Cascade outer position PI — bandwidth ≈ 5 rad/s:
    →  Kpp=4.0, Kip=10.0, Kdp=0.3
"""

import numpy as np
from pid_controller import PIDController, CascadePID
from plant_model import DCMotorPlant
from simulation import run_comparison


# ── Controller factory ──────────────────────────────────────────────────────

def _make_controllers(dt_single=0.01, dt_outer=0.01, dt_inner=0.001,
                      vel_limit=8.0, torque_limit=5.0):
    """Create matched controllers with systematic tuning."""
    # Single PID — binomial pole placement, omega_0 = 4 rad/s
    # J s^3 + (B+Kd) s^2 + Kp s + Ki = 0
    # ->  Kp=2.4, Ki=3.2, Kd=0.55
    single = PIDController(kp=2.4, ki=3.2, kd=0.55, dt=dt_single,
                           output_limits=(-torque_limit, torque_limit),
                           tau_f=0.01)

    # Inner velocity PI — zeta=1, omega_v = 20 rad/s (fast, well-damped)
    # B+Kpv = 2J omega_v = 2.0  -> Kpv=1.95
    # Kiv = J omega_v^2 = 20.0  -> Kiv=20.0
    inner = PIDController(kp=2.0, ki=20.0, kd=0.05, dt=dt_inner,
                          output_limits=(-torque_limit, torque_limit),
                          tau_f=0.002)

    # Outer position — P-dominant (small I avoids windup).
    # Inner loop tracks vel_ref; P maps position error to velocity linearly.
    outer = PIDController(kp=2.0, ki=0.3, kd=0.0, dt=dt_outer,
                          output_limits=(-vel_limit, vel_limit),
                          tau_f=0.01)

    cascade = CascadePID(outer, inner, vel_limit=vel_limit)
    return single, cascade


def _make_plants():
    return (DCMotorPlant(J=0.05, B=0.05, K=1.0, max_torque=5.0),
            DCMotorPlant(J=0.05, B=0.05, K=1.0, max_torque=5.0))


# ── Scenario 1: Step response ───────────────────────────────────────────────

def scenario_step_response():
    """Step from 0 -> 1 rad at t = 1.0 s."""
    single_pid, cascade_pid = _make_controllers()
    p_s, p_c = _make_plants()

    def ref_func(t):
        return 1.0 if t >= 1.0 else 0.0

    (res_s, res_c) = run_comparison(p_s, p_c, single_pid, cascade_pid,
                                    ref_func, t_final=5.0)
    return res_s, res_c, "Scenario 1: Step Response"


# ── Scenario 2: Disturbance rejection ───────────────────────────────────────

def scenario_disturbance():
    """Constant ref = 1 rad, step disturbance tau_dist = 1.5 Nm at t = 2 s.

    Key insight: the disturbance enters at the velocity level.
    The cascade inner loop counteracts it immediately (within ~1/ω_v = 40 ms).
    The single PID must wait for position error to build up.
    """
    single_pid, cascade_pid = _make_controllers()
    p_s, p_c = _make_plants()

    def ref_func(t):
        return 1.0 if t >= 1.0 else 0.0

    def dist_func(t):
        return 1.5 if t >= 3.0 else 0.0

    (res_s, res_c) = run_comparison(p_s, p_c, single_pid, cascade_pid,
                                    ref_func, dist_func=dist_func, t_final=6.0)
    return res_s, res_c, "Scenario 2: Disturbance Rejection (1.5 Nm at 3s)"


# ── Scenario 3: Parameter robustness ────────────────────────────────────────

def scenario_parameter_change():
    """Step response, then J triples 0.05 -> 0.15 at t = 3 s.

    The inner velocity loop masks the inertia change from the outer loop.
    Single PID's derivative term was tuned for the original J — it loses
    damping when J increases (effective ω_n drops).
    """
    single_pid, cascade_pid = _make_controllers()
    p_s, p_c = _make_plants()

    def ref_func(t):
        return 1.0 if t >= 1.0 else 0.0

    def param_change(plant, t):
        if t >= 3.0:
            plant.set_params(J=0.15)

    (res_s, res_c) = run_comparison(p_s, p_c, single_pid, cascade_pid,
                                    ref_func, param_change_func=param_change,
                                    t_final=8.0)
    return res_s, res_c, "Scenario 3: Parameter Robustness (J 0.05->0.15 at 3s)"


# ── Scenario 4: Measurement noise ───────────────────────────────────────────

def scenario_noise():
    """Constant ref = 1 rad, position noise sigma = 0.005 rad.

    Cascade uses EMA-filtered velocity reference (alpha=0.3) to suppress
    noise propagation from outer to inner loop, without compromising
    the inner loop's disturbance rejection bandwidth.
    """
    single_pid = PIDController(kp=2.4, ki=3.2, kd=0.55, dt=0.01,
                               output_limits=(-5.0, 5.0), tau_f=0.01)

    inner = PIDController(kp=2.0, ki=20.0, kd=0.05, dt=0.001,
                          output_limits=(-5.0, 5.0), tau_f=0.002)
    outer = PIDController(kp=2.0, ki=0.3, kd=0.0, dt=0.01,
                          output_limits=(-8.0, 8.0), tau_f=0.01)
    cascade_pid = CascadePID(outer, inner, vel_limit=8.0,
                             vel_ref_alpha=0.3)   # <-- key: smooth vel_ref

    p_s, p_c = _make_plants()
    rng = np.random.RandomState(42)

    def ref_func(t):
        return 1.0 if t >= 1.0 else 0.0

    def noise_func(t):
        return rng.normal(0.0, 0.005)

    (res_s, res_c) = run_comparison(p_s, p_c, single_pid, cascade_pid,
                                    ref_func, noise_func=noise_func,
                                    t_final=5.0)
    return res_s, res_c, "Scenario 4: Measurement Noise (sigma=0.005 rad)"


# ── Scenario registry ───────────────────────────────────────────────────────

SCENARIOS = [
    scenario_step_response,
    scenario_disturbance,
    scenario_parameter_change,
    scenario_noise,
]
