"""MPC 模型预测控制 — 车辆轨迹跟踪交互演示

用法:
    python demo.py          # 启动 GUI
    python demo.py --help   # 查看帮助

交互方式:
    滑条  — 实时调节 MPC 参数
    按钮  — 复位 / 暂停 / 切换轨迹 / 噪声
    键盘  — Space=暂停, R=复位, 1=圆形, 2=8字形, N=切换噪声
"""

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Polygon
from mpc_core import MPCController
from vehicle_model import VehicleModel

# ── 参考轨迹生成 ─────────────────────────────────────────────────────────

def ref_circle(t, R=25.0, omega=None, v_des=8.0):
    """圆形轨迹: (R cos(ωt), R sin(ωt)), 恒定速度"""
    if omega is None:
        omega = v_des / R
    px = R * np.cos(omega * t)
    py = R * np.sin(omega * t)
    theta = omega * t + np.pi / 2
    v = v_des
    return np.array([px, py, theta, v])


def ref_figure8(t, A=25.0, B=18.0, omega=0.12):
    """8字形轨迹 (Lemniscate of Gerono)"""
    px = A * np.sin(omega * t)
    py = B * np.sin(2 * omega * t)
    dpx = A * omega * np.cos(omega * t)
    dpy = 2 * B * omega * np.cos(2 * omega * t)
    theta = np.arctan2(dpy, dpx)
    v = np.sqrt(dpx ** 2 + dpy ** 2)
    return np.array([px, py, theta, v])


# ── 默认参数 ─────────────────────────────────────────────────────────────

DEFAULT = {
    'path_type': 'circle',
    'dt': 0.1,
    'N': 15,
    'Q_xy': 10.0,
    'Q_theta': 5.0,
    'Q_v': 1.0,
    'R_a': 0.1,
    'R_delta': 0.1,
    'L': 2.5,
    'v_des': 8.0,
    'circle_R': 25.0,
    'fig8_A': 25.0,
    'fig8_B': 18.0,
    'fig8_omega': 0.12,
}


# ── 车辆绘制 ─────────────────────────────────────────────────────────────

def get_car_poly(x, y, theta, L_car=4.0, W_car=2.0):
    """返回车辆三角形 Polygon"""
    local = np.array([
        [L_car / 2, 0],
        [-L_car / 2, -W_car / 2],
        [-L_car / 2, W_car / 2],
    ])
    c, s = np.cos(theta), np.sin(theta)
    R_mat = np.array([[c, -s], [s, c]])
    global_pts = local @ R_mat.T + np.array([x, y])
    return Polygon(global_pts, closed=True, facecolor='#2196F3',
                   edgecolor='#1565C0', linewidth=1.5, zorder=5, alpha=0.9)


# ── 主 Demo 类 ───────────────────────────────────────────────────────────

class MPCDemo:
    def __init__(self):
        self.p = DEFAULT.copy()
        self._build()
        self._setup_plot()
        self._setup_widgets()

    # ── 构建 MPC + 车辆 ──────────────────────────────────────────────────

    def _build(self):
        self.mpc = MPCController(
            dt=self.p['dt'], N=self.p['N'],
            Q_diag=np.array([self.p['Q_xy'], self.p['Q_xy'],
                             self.p['Q_theta'], self.p['Q_v']]),
            R_diag=np.array([self.p['R_a'], self.p['R_delta']]),
            L=self.p['L'],
        )
        self.vehicle = VehicleModel(dt=self.p['dt'], L=self.p['L'])
        self.t = 0.0
        self.history = []
        self.u_history = []
        self.info = {}
        self._reset_state()

    def _reset_state(self):
        s0 = self._ref_state(0.0)
        self.vehicle.reset(s0)
        self.mpc.reset()
        self.t = 0.0
        self.history = [s0.copy()]
        self.u_history = []
        self.info = {}
        self.x_pred = None
        self._params_dirty = False

    # ── 参考轨迹辅助 ─────────────────────────────────────────────────────

    def _ref_fn(self):
        return ref_circle if self.p['path_type'] == 'circle' else ref_figure8

    def _ref_state(self, t):
        fn = self._ref_fn()
        if self.p['path_type'] == 'circle':
            return fn(t, R=self.p['circle_R'], v_des=self.p['v_des'])
        else:
            return fn(t, A=self.p['fig8_A'], B=self.p['fig8_B'],
                      omega=self.p['fig8_omega'])

    def _ref_horizon(self, t):
        ts = t + np.arange(self.mpc.N + 1) * self.mpc.dt
        return np.array([self._ref_state(ti) for ti in ts])

    def _full_ref_path(self, num=500):
        if self.p['path_type'] == 'circle':
            T = 2 * np.pi * self.p['circle_R'] / self.p['v_des']
        else:
            T = 2 * np.pi / self.p['fig8_omega']
        ts = np.linspace(0, T, num)
        return np.array([self._ref_state(ti)[:2] for ti in ts])

    # ── 绘图区域 ─────────────────────────────────────────────────────────

    def _setup_plot(self):
        self.fig = plt.figure('MPC 车辆轨迹跟踪演示', figsize=(17, 11.5))
        self.fig.patch.set_facecolor('#FAFAFA')

        # 主视图 — 占上方 62%
        self.ax = self.fig.add_axes([0.04, 0.35, 0.68, 0.62])
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#F5F5F5')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_xlabel('X [米]', fontsize=11)
        self.ax.set_ylabel('Y [米]', fontsize=11)
        self.ax.set_title('MPC 车辆轨迹跟踪 — 拖动下方滑条实时调参',
                          fontsize=14, fontweight='bold')

        # 参考轨迹
        ref_xy = self._full_ref_path()
        self.line_ref, = self.ax.plot(ref_xy[:, 0], ref_xy[:, 1],
                                       '--', color='#FF5722', linewidth=1.8,
                                       alpha=0.7, label='参考轨迹')

        # 实际轨迹
        self.line_traj, = self.ax.plot([], [], '-', color='#4CAF50',
                                        linewidth=2.2, label='实际轨迹', alpha=0.9)

        # MPC 预测轨迹
        self.line_pred, = self.ax.plot([], [], 'o-', color='#FFC107',
                                        markersize=5, linewidth=1.3, alpha=0.65,
                                        label='MPC 预测 horizon')

        self.car_patch = None

        # 状态文字
        self.text_info = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontfamily='sans-serif', fontsize=9.5,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.88))

        self.ax.legend(loc='lower right', fontsize=10, framealpha=0.85)
        self._autoscale_pending = True

    def _autoscale(self):
        ref_xy = self._full_ref_path()
        mx, my = ref_xy[:, 0], ref_xy[:, 1]
        pad = 10
        self.ax.set_xlim(mx.min() - pad, mx.max() + pad)
        self.ax.set_ylim(my.min() - pad, my.max() + pad)

    # ── 控制组件 ─────────────────────────────────────────────────────────

    def _setup_widgets(self):
        """
        布局 (figure 17×11.5 英寸):
          主视图:  [0.04, 0.35, 0.68, 0.62]   (顶部 62%)
          状态面板: [0.74, 0.35, 0.24, 0.62]   (右侧)
          滑条区:  两列四行, y 中心在 0.29 / 0.23 / 0.17 / 0.11
          按钮区:  y=0.025
        滑条之间留足间隙，不会重叠。
        """
        color = '#2196F3'
        slider_h = 0.028   # 滑条高度
        slider_w = 0.41    # 滑条宽度

        def _sl_left(row, label, vmin, vmax, vinit, fmt='%.1f'):
            """左列滑条"""
            y_centers = [0.29, 0.23, 0.17, 0.11]
            y = y_centers[row]
            ax_s = self.fig.add_axes([0.065, y, slider_w, slider_h])
            ax_s.set_facecolor('#E8E8E8')
            s = Slider(ax_s, label, vmin, vmax, valinit=vinit, valfmt=fmt,
                       color=color)
            s.label.set_fontsize(10)
            s.valtext.set_fontsize(8.5)
            return s

        def _sl_right(row, label, vmin, vmax, vinit, fmt='%.1f'):
            """右列滑条"""
            y_centers = [0.29, 0.23, 0.17, 0.11]
            y = y_centers[row]
            ax_s = self.fig.add_axes([0.52, y, slider_w, slider_h])
            ax_s.set_facecolor('#E8E8E8')
            s = Slider(ax_s, label, vmin, vmax, valinit=vinit, valfmt=fmt,
                       color=color)
            s.label.set_fontsize(10)
            s.valtext.set_fontsize(8.5)
            return s

        # 左列 (4 个)
        self.slider_N      = _sl_left(0, '预测步数 N', 3, 30,
                                       self.p['N'], fmt='%d')
        self.slider_Qxy    = _sl_left(1, '位置权重 Q_xy', 0.1, 100,
                                       self.p['Q_xy'], fmt='%.1f')
        self.slider_Qtheta = _sl_left(2, '航向权重 Q_θ', 0.1, 50,
                                       self.p['Q_theta'], fmt='%.1f')
        self.slider_Ra     = _sl_left(3, '加速代价 R_a', 0.01, 10,
                                       self.p['R_a'], fmt='%.3f')

        # 右列 (4 个)
        self.slider_dt   = _sl_right(0, '步长 dt [秒]', 0.05, 0.20,
                                      self.p['dt'], fmt='%.2f')
        self.slider_v    = _sl_right(1, '目标速度 [米/秒]', 2, 20,
                                      self.p['v_des'], fmt='%.0f')
        self.slider_Rd   = _sl_right(2, '转向代价 R_δ', 0.01, 10,
                                      self.p['R_delta'], fmt='%.3f')
        self.slider_L    = _sl_right(3, '轴距 L [米]', 1.5, 4.0,
                                      self.p['L'], fmt='%.1f')

        # 注册回调
        all_sliders = [self.slider_N, self.slider_dt, self.slider_v,
                       self.slider_Qxy, self.slider_Qtheta,
                       self.slider_Ra, self.slider_Rd, self.slider_L]
        for s in all_sliders:
            s.on_changed(self._on_slider)

        # ── 按钮（必须保持引用防止 GC 回收）──
        self._buttons = []
        btn_w, btn_h = 0.078, 0.032
        btn_y = 0.025
        self._buttons.append(
            self._make_btn(0.065, btn_y, btn_w, btn_h, '重置', self._on_reset))
        self._buttons.append(
            self._make_btn(0.155, btn_y, btn_w, btn_h, '暂停', self._on_pause))
        self._buttons.append(
            self._make_btn(0.245, btn_y, btn_w + 0.015, btn_h, '圆形轨迹',
                           self._on_circle))
        self._buttons.append(
            self._make_btn(0.350, btn_y, btn_w + 0.015, btn_h, '8字轨迹',
                           self._on_figure8))
        self._buttons.append(
            self._make_btn(0.455, btn_y, btn_w + 0.015, btn_h, '噪 声',
                           self._on_noise))

        # ── 右侧状态面板 ──
        ax_status = self.fig.add_axes([0.74, 0.04, 0.24, 0.28])
        ax_status.axis('off')
        self.status_text = ax_status.text(
            0.02, 0.98, '', transform=ax_status.transAxes,
            fontfamily='sans-serif', fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.92))
        self._update_status_text()

        self.paused = False
        self.key_map = self.fig.canvas.mpl_connect('key_press_event',
                                                    self._on_key)

    def _make_btn(self, x, y, w, h, label, callback):
        ax_b = self.fig.add_axes([x, y, w, h])
        btn = Button(ax_b, label, color='#E0E0E0', hovercolor='#BBDEFB')
        btn.label.set_fontsize(10)
        btn.on_clicked(callback)
        return btn

    # ── 回调 ─────────────────────────────────────────────────────────────

    def _on_slider(self, val):
        self._read_params()
        self._params_dirty = True

    def _read_params(self):
        self.p['N']       = int(self.slider_N.val)
        self.p['dt']      = float(self.slider_dt.val)
        self.p['v_des']   = float(self.slider_v.val)
        self.p['Q_xy']    = float(self.slider_Qxy.val)
        self.p['Q_theta'] = float(self.slider_Qtheta.val)
        self.p['R_a']     = float(self.slider_Ra.val)
        self.p['R_delta'] = float(self.slider_Rd.val)
        self.p['L']       = float(self.slider_L.val)

    def _apply_params(self):
        self.mpc.update_params(
            dt=self.p['dt'], N=self.p['N'],
            Q_diag=np.array([self.p['Q_xy'], self.p['Q_xy'],
                             self.p['Q_theta'], self.p['Q_v']]),
            R_diag=np.array([self.p['R_a'], self.p['R_delta']]),
            L=self.p['L'],
        )
        self.vehicle.dt = self.p['dt']
        self.vehicle.L = self.p['L']

    def _on_reset(self, event=None):
        self._read_params()
        self._apply_params()
        self._reset_state()
        self._rebuild_ref_path()

    def _on_pause(self, event=None):
        self.paused = not self.paused

    def _on_circle(self, event=None):
        if self.p['path_type'] == 'circle':
            return
        self.p['path_type'] = 'circle'
        self._apply_params()
        self._reset_state()
        self._rebuild_ref_path()

    def _on_figure8(self, event=None):
        if self.p['path_type'] == 'figure8':
            return
        self.p['path_type'] = 'figure8'
        self._apply_params()
        self._reset_state()
        self._rebuild_ref_path()

    def _on_noise(self, event=None):
        std = self.vehicle.noise_std
        if np.allclose(std, 0):
            self.vehicle.set_noise(0.03, 0.03, 0.01, 0.02)
        else:
            self.vehicle.set_noise(0, 0, 0, 0)

    def _on_key(self, event):
        if event.key == ' ':
            self._on_pause()
        elif event.key == 'r':
            self._on_reset()
        elif event.key == '1':
            self._on_circle()
        elif event.key == '2':
            self._on_figure8()
        elif event.key == 'n':
            self._on_noise()

    def _rebuild_ref_path(self):
        ref_xy = self._full_ref_path()
        self.line_ref.set_data(ref_xy[:, 0], ref_xy[:, 1])

    def _update_status_text(self):
        p = self.p
        mpc = self.mpc
        path_name = '圆形' if p['path_type'] == 'circle' else '8字形'
        noise_on = np.any(self.vehicle.noise_std > 0)
        lines = [
            '════ MPC 状态 ════',
            '',
            f'轨迹:      {path_name}',
            f'预测步数:  {mpc.N}',
            f'步长:      {mpc.dt:.2f} 秒',
            f'轴距:      {mpc.L:.1f} 米',
            '',
            '── 代价权重 ──',
            f'Q_xy:      {p["Q_xy"]:.1f}',
            f'Q_θ:       {p["Q_theta"]:.1f}',
            f'Q_v:       {p["Q_v"]:.2f}',
            f'R_a:       {p["R_a"]:.3f}',
            f'R_δ:       {p["R_delta"]:.3f}',
            '',
            '── 运行状态 ──',
        ]
        if self.info:
            ok = '✓' if self.info.get('success') else '✗'
            lines += [
                f'求解:      {ok}',
                f'代价:      {self.info.get("cost", 0):.2f}',
                f'迭代:      {self.info.get("nit", 0)}',
                f'累计求解:  {self.info.get("n_evals", 0)}',
            ]
        if noise_on:
            lines += ['', '⚠ 噪声: 开启']
        lines += [
            '',
            '── 键盘快捷键 ──',
            'Space  暂停',
            'R      复位',
            '1      圆形轨迹',
            '2      8字轨迹',
            'N      切换噪声',
        ]
        self.status_text.set_text('\n'.join(lines))

    # ── 仿真步进 ─────────────────────────────────────────────────────────

    def step(self):
        if self._params_dirty:
            self._apply_params()
            self._params_dirty = False

        x0 = self.vehicle.state.copy()
        x_ref = self._ref_horizon(self.t)

        try:
            u_opt, u_seq, self.x_pred, self.info = self.mpc.solve(x0, x_ref)
        except Exception:
            u_opt = np.zeros(2)
            self.x_pred = np.tile(x0, (self.mpc.N + 1, 1))
            self.info = {'success': False, 'cost': 0, 'nit': 0, 'n_evals': 0}

        self.vehicle.step(u_opt)
        self.t += self.mpc.dt
        self.history.append(self.vehicle.state.copy())
        self.u_history.append(u_opt)

        if len(self.history) > 2000:
            self.history = self.history[-1500:]
            self.u_history = self.u_history[-1500:]

    def animate(self, frame):
        if not self.paused:
            self.step()

        hist = np.array(self.history)
        if len(hist) > 1:
            self.line_traj.set_data(hist[:, 0], hist[:, 1])

        if self.x_pred is not None:
            self.line_pred.set_data(self.x_pred[:, 0], self.x_pred[:, 1])

        if self.car_patch is not None:
            self.car_patch.remove()
        s = self.vehicle.state
        self.car_patch = get_car_poly(s[0], s[1], s[2])
        self.ax.add_patch(self.car_patch)

        steering_deg = (self.u_history[-1][1] * 180 / np.pi) if self.u_history else 0
        accel = self.u_history[-1][0] if self.u_history else 0
        paused_str = ' [已暂停]' if self.paused else ''
        self.text_info.set_text(
            f'时间={self.t:.1f}秒  |  车速={s[3]:.2f} 米/秒  |  '
            f'转向角={steering_deg:+.1f}°  |  '
            f'加速度={accel:+.2f} 米/秒²  |  '
            f'求解次数={self.info.get("n_evals", 0)}'
            + paused_str
        )

        self._update_status_text()

        if self._autoscale_pending and len(self.history) > 5:
            self._autoscale()
            self._autoscale_pending = False

        return [self.line_traj, self.line_pred, self.car_patch]

    def run(self):
        self._rebuild_ref_path()
        self._autoscale()
        self._autoscale_pending = False
        self.ani = animation.FuncAnimation(
            self.fig, self.animate,
            interval=max(20, int(self.mpc.dt * 1000)),
            blit=False, cache_frame_data=False
        )
        plt.show()


# ── 入口 ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)
    demo = MPCDemo()
    demo.run()
