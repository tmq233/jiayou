import numpy as np


class SimulationResult:
    """Stores logged data for a single controller run."""

    def __init__(self):
        self.time = []
        self.pos = []
        self.vel = []
        self.u = []
        self.ref = []
        self.vel_ref = []  # only populated for cascade (inner-loop setpoint)

    def record(self, t, pos, vel, u, ref, vel_ref=None):
        self.time.append(t)
        self.pos.append(pos)
        self.vel.append(vel)
        self.u.append(u)
        self.ref.append(ref)
        self.vel_ref.append(vel_ref)


def run_comparison(plant_single, plant_cascade,
                   single_pid, cascade_pid,
                   ref_func, dist_func=None, noise_func=None,
                   param_change_func=None,
                   t_final=5.0, sim_dt=1e-4,
                   single_ratio=100,
                   cascade_outer_ratio=100,
                   cascade_inner_ratio=10):
    """Side-by-side simulation: single-loop vs cascade PID on identical plants.

    Multi-rate control:
        - Simulation integration runs at sim_dt          (10 kHz default)
        - Single PID updates every single_ratio steps    (100 Hz default)
        - Cascade outer loop every cascade_outer_ratio   (100 Hz default)
        - Cascade inner loop every cascade_inner_ratio   (1 kHz default)

    Parameters
    ----------
    plant_single, plant_cascade : DCMotorPlant
        Two independent plant instances (same initial state).
    single_pid : PIDController
    cascade_pid : CascadePID
    ref_func(t) -> float       : reference trajectory
    dist_func(t) -> float      : disturbance torque (optional)
    noise_func(t) -> float     : measurement noise (optional)
    param_change_func(plant, t): called each step for online parameter changes
    t_final : float            : simulation duration (s)
    sim_dt  : float            : integration time step (s)
    single_ratio : int         : single PID update every N sim steps
    cascade_outer_ratio : int  : cascade outer loop every N sim steps
    cascade_inner_ratio : int  : cascade inner loop every N sim steps
    """
    num_steps = int(t_final / sim_dt)
    results = SimulationResult(), SimulationResult()
    res_s, res_c = results

    u_single = 0.0
    u_cascade = 0.0

    for step in range(num_steps):
        t = step * sim_dt
        ref_val = ref_func(t)
        tau_dist = dist_func(t) if dist_func else 0.0
        noise = noise_func(t) if noise_func else 0.0

        if param_change_func:
            param_change_func(plant_single, t)
            param_change_func(plant_cascade, t)

        # --- Single PID ---
        if step % single_ratio == 0:
            pos_meas = plant_single.position + noise
            u_single = single_pid.update(ref_val, pos_meas)
        plant_single.step(u_single, tau_dist, sim_dt)

        # --- Cascade PID ---
        if step % cascade_outer_ratio == 0:
            pos_meas = plant_cascade.position + noise
            cascade_pid.update_outer(ref_val, pos_meas)
        if step % cascade_inner_ratio == 0:
            vel_meas = plant_cascade.velocity
            u_cascade = cascade_pid.update_inner(vel_meas)
        plant_cascade.step(u_cascade, tau_dist, sim_dt)

        # Log at 1 kHz
        if step % 10 == 0:
            res_s.record(t, plant_single.position, plant_single.velocity,
                         u_single, ref_val, None)
            res_c.record(t, plant_cascade.position, plant_cascade.velocity,
                         u_cascade, ref_val, cascade_pid.vel_ref)

    return results
