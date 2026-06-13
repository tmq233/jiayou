"""Plotting and performance-metric utilities."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ── Metrics ─────────────────────────────────────────────────────────────────

def _rms(x):
    return float(np.sqrt(np.mean(np.array(x) ** 2)))


def _settling_time(t, pos, step_t, band=0.02):
    """Settling time measured from `step_t` to when |pos - r_ss| < band."""
    t_arr = np.array(t)
    p_arr = np.array(pos)
    # steady-state = mean of last 10 %
    n_ss = max(1, len(p_arr) // 10)
    r_ss = float(np.mean(p_arr[-n_ss:]))
    if abs(r_ss) < 1e-4:
        return 0.0
    tol = band * abs(r_ss) + 0.001
    below = np.abs(p_arr - r_ss) <= tol
    # find last violation after step_t
    idx_start = np.searchsorted(t_arr, step_t)
    for i in range(len(t_arr) - 1, idx_start, -1):
        if not below[i]:
            return float(t_arr[min(i + 1, len(t_arr) - 1)])
    return float(t_arr[idx_start])


def _overshoot(pos, step_t):
    """Overshoot (%) relative to steady state after step_t."""
    p = np.array(pos)
    r_ss = float(np.mean(p[-max(1, len(p) // 10):]))
    if abs(r_ss) < 1e-4:
        return 0.0
    idx = np.searchsorted(np.array(pos), r_ss)  # rough
    peak = float(np.max(p))
    if peak > r_ss:
        return (peak - r_ss) / abs(r_ss) * 100.0
    return 0.0


def _iae_from(t, pos, ref, t_start):
    """IAE computed from t_start to end."""
    t_arr = np.array(t)
    p_arr = np.array(pos)
    r_arr = np.array(ref)
    idx = np.searchsorted(t_arr, t_start)
    if idx >= len(t_arr) - 1:
        return 0.0
    dt = t_arr[1] - t_arr[0]
    return float(np.sum(np.abs(p_arr[idx:] - r_arr[idx:])) * dt)


def _max_deviation_after(t, pos, ref, t_event):
    """Maximum absolute deviation from reference after t_event."""
    t_arr = np.array(t)
    p_arr = np.array(pos)
    r_arr = np.array(ref)
    idx = np.searchsorted(t_arr, t_event)
    if idx >= len(t_arr):
        return 0.0
    return float(np.max(np.abs(p_arr[idx:] - r_arr[idx:])))


def compute_metrics(time, pos, ref, u, step_t=None):
    """Standard performance indices.

    If step_t is None, measures over the full trace.
    """
    t = np.array(time)
    p = np.array(pos)
    r = np.array(ref)
    ctrl = np.array(u)
    e = p - r
    dt = t[1] - t[0]

    r_ss = float(np.mean(r[-max(1, len(r) // 10):]))

    # Overshoot
    if abs(r_ss) > 1e-4 and step_t is not None:
        overshoot = _overshoot(pos, step_t)
    elif abs(r_ss) > 1e-4:
        peak = float(np.max(p))
        overshoot = (peak - r_ss) / abs(r_ss) * 100.0 if peak > r_ss else 0.0
    else:
        overshoot = 0.0

    # Settling time
    if step_t is not None:
        st = _settling_time(time, pos, step_t)
    else:
        st = _settling_time(time, pos, 0.0)

    # ISE / IAE
    ise = float(np.sum(e ** 2) * dt)
    iae = float(np.sum(np.abs(e)) * dt)

    # Control RMS
    u_rms = float(np.sqrt(np.mean(ctrl ** 2)))

    return {
        "overshoot_%": overshoot,
        "settling_time_s": st,
        "ISE": ise,
        "IAE": iae,
        "u_rms": u_rms,
        "pos_rms": _rms(p),
    }


def compute_disturbance_metrics(time, pos, ref, u, t_dist):
    """Metrics specific to disturbance rejection."""
    return {
        "max_dev_after_dist": _max_deviation_after(time, pos, ref, t_dist),
        "iae_after_dist": _iae_from(time, pos, ref, t_dist),
    }


def compute_noise_metrics(time, pos, ref, u, t_noise_start=0.0):
    """Metrics specific to noise sensitivity."""
    t_arr = np.array(time)
    p_arr = np.array(pos)
    ctrl = np.array(u)
    idx = np.searchsorted(t_arr, t_noise_start)
    if idx >= len(t_arr) - 1:
        return {"pos_std": 0.0, "u_std": 0.0}
    return {
        "pos_std": float(np.std(p_arr[idx:])),
        "u_std": float(np.std(ctrl[idx:])),
    }


# ── Table printing ──────────────────────────────────────────────────────────

def _print_row(label, sv, cv, lower_is_better=True):
    if abs(sv) > 1e-8:
        imp = (sv - cv) / abs(sv) * 100.0
    else:
        imp = 0.0
    arrow = "↑" if (imp > 0 and not lower_is_better) or (imp < 0 and lower_is_better) else ""
    print(f"  {label:<28} {sv:10.4f}   {cv:10.4f}   {imp:+7.1f}%  {arrow}")


def print_metrics_table(scenario_name, m_single, m_cascade):
    print(f"\n{'='*75}")
    print(f"  {scenario_name}")
    print(f"{'='*75}")
    print(f"{'Metric':<28} {'Single PID':>10} {'Cascade PID':>10} {'  Δ%':>8}")
    print(f"{'-'*28} {'-'*10} {'-'*10} {'-'*8}")

    for key, label, lower_better in [
        ("overshoot_%", "Overshoot (%)", True),
        ("settling_time_s", "Settling time (s)", True),
        ("ISE", "ISE", True),
        ("IAE", "IAE", True),
        ("u_rms", "Control RMS", True),
    ]:
        _print_row(label, m_single[key], m_cascade[key], lower_better)

    # Extra metrics if present
    for key, label, lower_better in [
        ("max_dev_after_dist", "Max dev after disturb", True),
        ("iae_after_dist", "IAE after disturb", True),
        ("pos_std", "Position std", True),
        ("u_std", "Control std", True),
    ]:
        if key in m_single and key in m_cascade:
            _print_row(label, m_single[key], m_cascade[key], lower_better)


# ── Plotting ────────────────────────────────────────────────────────────────

def _style_ax(ax, xlabel="Time (s)"):
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.yaxis.set_major_locator(MaxNLocator(5))


def plot_scenario(res_s, res_c, title, filepath):
    """2×2 comparison figure for one scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ts, tc = np.array(res_s.time), np.array(res_c.time)

    # Position tracking
    ax = axes[0, 0]
    ax.plot(ts, res_s.ref, "k--", linewidth=0.8, label="reference")
    ax.plot(ts, res_s.pos, "C0", linewidth=1.2, label="Single PID")
    ax.plot(tc, res_c.pos, "C1", linewidth=1.2, label="Cascade PID")
    ax.set_ylabel("Position (rad)")
    ax.set_title("Position Tracking")
    ax.legend(fontsize=8, loc="lower right")
    _style_ax(ax)

    # Velocity
    ax = axes[0, 1]
    ax.plot(ts, res_s.vel, "C0", linewidth=1.2, label="Single PID")
    ax.plot(tc, res_c.vel, "C1", linewidth=1.2, label="Cascade PID")
    vref = np.array([v if v is not None else np.nan for v in res_c.vel_ref])
    ax.plot(tc, vref, "C2", linestyle="--", linewidth=0.8,
            label="Cascade $\\omega_{ref}$")
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Velocity")
    ax.legend(fontsize=8)
    _style_ax(ax)

    # Control effort
    ax = axes[1, 0]
    ax.plot(ts, res_s.u, "C0", linewidth=0.8, label="Single PID")
    ax.plot(tc, res_c.u, "C1", linewidth=0.8, label="Cascade PID")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Control Effort")
    ax.legend(fontsize=8)
    _style_ax(ax)

    # Tracking error
    ax = axes[1, 1]
    e_s = np.array(res_s.pos) - np.array(res_s.ref)
    e_c = np.array(res_c.pos) - np.array(res_c.ref)
    ax.plot(ts, e_s, "C0", linewidth=1.0, label="Single PID")
    ax.plot(tc, e_c, "C1", linewidth=1.0, label="Cascade PID")
    ax.set_ylabel("Position Error (rad)")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=8)
    _style_ax(ax)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {filepath}")


def plot_metrics_summary(all_metrics, filepath):
    """Bar chart: single vs cascade across all scenarios."""
    labels = list(all_metrics.keys())
    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Performance Metrics: Single vs Cascade PID", fontsize=13,
                 fontweight="bold")

    metric_keys = ["overshoot_%", "settling_time_s", "ISE", "IAE"]
    metric_names = ["Overshoot (%)", "Settling Time (s)", "ISE", "IAE (rad.s)"]

    for ax, mkey, mname in zip(axes.flat, metric_keys, metric_names):
        sv = [all_metrics[l]["single"][mkey] for l in labels]
        cv = [all_metrics[l]["cascade"][mkey] for l in labels]
        ax.bar(x - width / 2, sv, width, label="Single PID", color="C0",
               alpha=0.85)
        ax.bar(x + width / 2, cv, width, label="Cascade PID", color="C1",
               alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i+1}" for i in range(n)], fontsize=9)
        ax.set_ylabel(mname)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {filepath}")
