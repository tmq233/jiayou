"""
DDPG算法 —— 工具模块
包含：Actor网络、Critic网络、策略评估函数、参数解析工具
"""

import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch


# ============================================================
# Actor（策略网络 / 演员网络）
# ============================================================
# 功能：根据当前状态 s，输出一个确定性的动作 a = μ(s)
# DDPG 使用"确定性策略梯度"（Deterministic Policy Gradient, DPG），
# 而非随机策略（如 PPO 输出均值和方差，再从中采样）。
# 确定性策略的优点是：不需要对动作空间进行高方差采样，
# 更适合高维连续动作空间。
#
# 网络结构：state_dim → 400 → 300 → action_dim
# 激活函数：前两层用 ReLU，输出层用 tanh 并乘以 max_action
# tanh 将输出限制在 [-1, +1]，乘以 max_action 后映射到实际动作范围
# ============================================================
#定义 Actor 网络类来输出动作
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        """
        参数说明：
            state_dim  : 状态空间的维度（观测向量长度）
            action_dim : 动作空间的维度（动作向量长度）
            net_width  : 隐藏层第一层的宽度（默认400）
            maxaction  : 动作的最大绝对值，如 Pendulum 为 2.0
                         用于将 tanh 的 [-1,+1] 缩放到实际动作范围
        """
        super(Actor, self).__init__()

        # 第一层：状态 → 400维隐藏层
        self.l1 = nn.Linear(state_dim, net_width)
        # 第二层：400维 → 300维隐藏层
        self.l2 = nn.Linear(net_width, 300)
        # 第三层（输出层）：300维 → 动作维度
        self.l3 = nn.Linear(300, action_dim)

        # 保存动作上限，forward 中使用
        self.maxaction = maxaction

    def forward(self, state):
        """
        前向传播：给定状态 s，返回确定性动作 a = μ(s)

        输入 shape：state: [batch_size, state_dim]
        输出 shape：a:     [batch_size, action_dim]

        流程：
            s → Linear+ReLU → Linear+ReLU → Linear+tanh → *maxaction → a
        """
        # 第一层：线性变换 + ReLU 激活
        a = torch.relu(self.l1(state))
        # 第二层：线性变换 + ReLU 激活
        a = torch.relu(self.l2(a))
        # 输出层：线性变换 + tanh 激活 + 缩放到实际动作范围
        # tanh 保证输出在 (-1, 1) 之间，乘以 maxaction 映射到 [-max, +max]
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a


# ============================================================
# Q_Critic（Q值网络 / 评论家网络）
# ============================================================
# 功能：评估在状态 s 下执行动作 a 的 Q 值 Q(s, a)
# Q 值 = 从状态 s 开始，执行动作 a，之后一直遵循 Actor 策略的累积折扣奖励
#
# 网络结构：(state_dim + action_dim) → 400 → 300 → 1
# 将状态和动作拼接在一起作为输入（这是 DDPG 的标准做法），
# 输出一个标量 Q 值。
#
# 注意：输入不包含目标动作（不同于 DQN 输出每个动作的 Q 值），
# 因为 DDPG 处理的是连续动作空间，无法枚举所有动作。
# ============================================================
###定义 Q_Critic 网络类来输出 Q 值
class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        """
        参数说明：
            state_dim  : 状态空间维度
            action_dim : 动作空间维度
            net_width  : 隐藏层第一层宽度（默认400）
        """
        super(Q_Critic, self).__init__()

        # 第一层：状态+动作拼接 → 400维
        # 输入维度 = state_dim + action_dim
        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        # 第二层：400维 → 300维
        self.l2 = nn.Linear(net_width, 300)
        # 输出层：300维 → 1（单个Q值）
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        """
        前向传播：计算 Q(s, a)

        输入 shape：
            state:  [batch_size, state_dim]
            action: [batch_size, action_dim]
        输出 shape：
            q: [batch_size, 1] — 每个样本的 Q 值

        流程：
            (s, a) 拼接 → Linear+ReLU → Linear+ReLU → Linear → Q值
        """
        # 沿第1维（列方向）拼接状态和动作：dim=1 表示按列拼接
        sa = torch.cat([state, action], 1)
        # 第一隐藏层 + ReLU
        q = F.relu(self.l1(sa))
        # 第二隐藏层 + ReLU
        q = F.relu(self.l2(q))
        # 输出层：不加激活函数，Q 值范围不受限（理论上可为任意实数）
        q = self.l3(q)
        return q


# ============================================================
# evaluate_policy：评估当前策略的表现
# ============================================================
# 【为什么要评估？】
#   训练过程中，我们不仅要让 Agent 与环境交互，还需要定期"考试"——
#   暂停训练，关掉噪声，看看 Actor 到底学到了多少。
#   这就像学完一章就做一次小测验，不测验你永远不知道是真学会了还是假学会了。
#
# 【评估和训练的关键区别】
#   ┌──────────┬──────────────────┬──────────────────┐
#   │          │      训练模式     │      评估模式     │
#   ├──────────┼──────────────────┼──────────────────┤
#   │  deterministic │ False           │ True             │
#   │ 动作      │ μ(s) + 噪声      │ μ(s)（纯确定性）  │
#   │ 目标      │ 探索+学习         │ 测量真实水平      │
#   │ 反向传播  │ 是（更新网络）    │ 否（只推断）      │
#   └──────────┴──────────────────┴──────────────────┘
#
# 【为什么用独立环境 eval_env？】
#   如果评估和训练共用同一个环境，评估时走过的状态序列会干扰
#   训练环境的内部随机数状态。用独立环境让"训练"和"考试"
#   完全隔离，评估结果才有可比性。
#
# 【为什么跑多轮取平均？】
#   单轮评估可能碰上"好运气"（初始状态简单）或"坏运气"（初始状态刁钻），
#   多轮取平均可以平滑掉随机波动，给出更稳定的分数。
#   默认 turns=3，既不至于太耗时，又足够消除极端情况。
# ============================================================

def evaluate_policy(env, agent, turns=3):
    """
    评估策略的平均表现（只看不训，纯推断模式）

    参数：
        env   : 独立评估环境（Gym environment）
                必须与训练环境隔离，避免干扰训练环境的内部状态
        agent : DDPG_agent 实例
                调用其 select_action 方法获取动作
        turns : 评估的 episode 数量（默认 3 轮）
                轮数越多，结果越准确，但耗时越长

    返回：
        int — turns 轮累积奖励的平均值
              越高 = 策略越好，越接近 0 = 倒立摆竖直

    示例：
        >>> score = evaluate_policy(eval_env, agent, turns=5)
        >>> print(score)  # -150 (Pendulum 初始状态三分的累积奖励)
    """
    total_scores = 0  # 多轮累积奖励的总和（用于最后求平均）

    # ---- 跑 turns 轮，每轮是一个完整的 episode ----
    for j in range(turns):
        # ----- ① 重置环境，获取该轮初始状态 -----
        # env.reset() 让环境"重新发牌"：
        #   - s:  该轮第一个状态（观测），如 Pendulum 的 [cosθ, sinθ, θ̇]
        #         初始摆杆角度是随机的，所以每轮的初始 s 都不同
        #   - info: 附加信息字典，绝大多数环境返回空字典 {}
        # 注意：这里没有传 seed，意味着每轮的随机初始状态完全由
        #       环境内部的随机数生成器决定，不受训练 seed 的影响
        s, info = env.reset()
        done = False  # 该轮的终止标志，初始化为 False

        # ----- ② 在该 episode 内一步步走，直到结束 -----
        while not done:
            # ③ 动作选择：确定性模式（deterministic=True）
            #    关键：这里不要探索！不要加噪声！
            #    训练时加噪声是为了"试试新花样"，评估时我们要看的是
            #    "Actor 目前认为的最优解"到底能不能拿高分。
            #    如果评估也加噪声，分数会被随机波动淹没，看不出训练的真实进展。
            a = agent.select_action(s, deterministic=True)

            # ④ 环境执行一步，返回 5 个值：
            #    s_next : 执行 a 后环境进入的新状态
            #    r      : 即时奖励（float），Pendulum 为 -(θ² + 0.1θ̇² + 0.001u²)
            #             越接近竖直，|θ| 越小，|r| 越小（越接近 0）
            #    dw     : bool — 环境自然终止（如 Pendulum 永不自然终止，恒为 False；
            #                      但 BipedalWalker 摔倒 / LunarLander 着陆 则为 True）
            #    tr     : bool — 达到最大步数后被截断（truncated）
            #                     如 Pendulum max_episode_steps=200，走满 200 步 tr=True
            #    info   : 额外信息（一般不用）
            s_next, r, dw, tr, info = env.step(a)

            # ⑤ 判断该 episode 是否结束
            #    两种结束条件：
            #      dw=True  → 任务成功或失败（到达目标/摔倒等）
            #      tr=True  → 时间到了（走过 max_episode_steps 步，被强制截断）
            #    任意一个为 True，该轮结束
            done = (dw or tr)

            # ⑥ 累加该轮的即时奖励
            #    注意：这里直接累加 r，不乘折扣因子 γ
            #    因为评估要的是"实际收获了多少"，而不是"TD 估计值"
            total_scores += r

            # ⑦ 状态转移：把下一步状态赋给 s，供下一次动作选择使用
            #    等价于"往前走了一步，现在处在新的位置"
            s = s_next

    # ---- 返回 turns 轮的平均累积奖励（取整） ----
    # int() 向下取整，便于日志输出和对齐
    return int(total_scores / turns)


# ============================================================
# str2bool：字符串转布尔值（用于 argparse 参数解析）
# ============================================================
# 【为什么需要这个函数？】
#   argparse 没有原生的 bool 类型解析。如果直接用 type=bool：
#     parser.add_argument('--flag', type=bool)
#   那么 python main.py --flag False 会把 "False" 当成非空字符串，
#   结果 bool("False") = True！（Python 中任意非空字符串转 bool 都是 True）
#   所以需要手动写一个类型转换函数。
#
# 【在 main.py 中的使用】
#   parser.add_argument('--render', type=str2bool, default=False)
#   命令行: python main.py --render True   → True
#   命令行: python main.py --render False  → False
#   不传:                                 → False（用 default）
#
# 【逻辑流程】
#   ① 如果已经是 bool（可能来自代码内直接调用），直接返回
#   ② 先 .lower() 转小写，然后匹配各种 True/False 的写法
#      ↓ 但下面的元组里既有小写（'true'）又有大写（'TRUE'），
#        这是冗余的吗？
#       → 不是！因为 v.lower() 只转纯 ASCII 字母，
#           某些终端（如中文 Windows 的 cmd）传入的参数可能
#           带有不可见的 Unicode 字符或混合大小写，
#           直接同时匹配大小写形式更稳健（belt-and-suspenders）
#   ③ 什么都匹配不上 → 抛出明确的错误提示
# ============================================================

# ============================================================
# plot_training_curves：训练结束后自动生成完整参数曲线图
# ============================================================
# 【为什么需要这个函数？】
#   TensorBoard 需要在浏览器中查看，有时不方便（如离线环境、
#   写报告/论文需要直接插入图片）。这个函数在训练结束后自动
#   将所有指标画到一张大图上，并保存 CSV 原始数据，方便后续分析。
#
# 【输出】
#   - PNG 图片：3×4 子图布局，每 2000 步一个数据点
#   - CSV 文件：与图片同目录，可用 Excel/Python 打开
# ============================================================

def plot_training_curves(history, save_dir, env_name):
    """
    训练结束后自动生成所有指标曲线图并保存

    参数：
        history  : dict — {'steps': [...], 'ep_r': [...], 'q_loss': [...], ...}
        save_dir : str  — 图片保存目录（如 'plots/PV1/'）
        env_name : str  — 环境简称，用于图表标题（如 'PV1'）

    输出文件：
        {save_dir}/training_curves_{env_name}.png
        {save_dir}/training_curves_{env_name}.csv
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互后端，不弹窗
        import matplotlib.pyplot as plt
        import os, csv
        import numpy as np
    except ImportError:
        print("[Warning] matplotlib not installed. Skip plotting. "
              "Install with: pip install matplotlib")
        return

    os.makedirs(save_dir, exist_ok=True)

    steps = history['steps']
    if len(steps) == 0:
        print("[Warning] No history data to plot.")
        return

    # ================================================================
    # 定义要绘制的指标及其子图位置、标题、Y轴标签
    # ================================================================
    # 格式: (key, row, col, title, ylabel)
    # 3 行 × 4 列 = 12 个子图
    metrics = [
        # ---- 第 1 行：奖励和损失 ----
        ('ep_r',              0, 0, 'Episode Reward (eval)',         'Reward'),
        ('q_loss',            0, 1, 'Critic Loss (MSE)',             'Loss'),
        ('a_loss',            0, 2, 'Actor Loss (-Q mean)',          'Loss'),
        ('reward_batch_mean', 0, 3, 'Batch Reward Mean',            'Reward'),
        # ---- 第 2 行：Q 值 ----
        ('current_Q_mean',    1, 0, 'Current Q(s,a) Mean',          'Q Value'),
        ('target_Q_mean',     1, 1, 'TD Target y Mean',             'Q Value'),
        ('target_Q_raw_mean', 1, 2, "Target Q'(s',a') Mean",        'Q Value'),
        # 第 2 行第 4 列：三条 Q 值合并对比
        ('_q_comparison',     1, 3, 'Q Values Comparison',          'Q Value'),
        # ---- 第 3 行：动作统计 ----
        ('actor_a_mean',      2, 0, 'Actor μ(s) Mean',              'Action'),
        ('actor_a_std',       2, 1, 'Actor μ(s) Std',               'Action'),
        ('target_a_mean',     2, 2, "Target Actor μ'(s') Mean",     'Action'),
        ('target_a_std',      2, 3, "Target Actor μ'(s') Std",      'Action'),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle(f'DDPG Training Curves — {env_name}', fontsize=16, fontweight='bold')

    for key, row, col, title, ylabel in metrics:
        ax = axes[row][col]

        if key == '_q_comparison':
            # 特殊处理：三条 Q 值画在同一个子图里做对比
            if 'current_Q_mean' in history and len(history['current_Q_mean']) > 0:
                ax.plot(steps, history.get('current_Q_mean', []),
                        color='#2196F3', linewidth=1.2, alpha=0.85, label='Current Q(s,a)')
                ax.plot(steps, history.get('target_Q_mean', []),
                        color='#FF9800', linewidth=1.2, alpha=0.85, label='TD Target y')
                ax.plot(steps, history.get('target_Q_raw_mean', []),
                        color='#4CAF50', linewidth=1.2, alpha=0.85, label="Target Q'(s',a')")
                ax.legend(fontsize=8, loc='best')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
        else:
            values = history.get(key, [])
            if len(values) > 0:
                ax.plot(steps, values, color='#2196F3', linewidth=1.2, alpha=0.85)
                # 添加平滑趋势线（窗口平均）
                if len(values) >= 10:
                    window = max(len(values) // 20, 3)
                    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                    smooth_steps = steps[window-1:]
                    ax.plot(smooth_steps, smoothed, color='#FF5722',
                            linewidth=1.5, alpha=0.7, linestyle='--', label=f'Smooth (w={window})')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Steps (k)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留空间给总标题

    # ---- 保存 PNG ----
    png_path = os.path.join(save_dir, f'training_curves_{env_name}.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n📈 训练曲线图已保存: {png_path}")

    # ---- 保存 CSV ----
    csv_path = os.path.join(save_dir, f'training_curves_{env_name}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        # 只写入实际存在的列
        columns = ['steps', 'ep_r', 'q_loss', 'a_loss', 'reward_batch_mean',
                   'current_Q_mean', 'target_Q_mean', 'target_Q_raw_mean',
                   'actor_a_mean', 'actor_a_std', 'target_a_mean', 'target_a_std']
        available_cols = [c for c in columns if history.get(c) and len(history[c]) > 0]
        writer = csv.writer(f)
        writer.writerow(available_cols)
        n = min(len(history[c]) for c in available_cols if c != 'steps')
        # 确保 steps 列对应
        for i in range(min(len(steps), n)):
            row_data = []
            for c in available_cols:
                if c == 'steps':
                    row_data.append(steps[i])
                elif i < len(history.get(c, [])):
                    row_data.append(history[c][i])
                else:
                    row_data.append('')
            writer.writerow(row_data)
    print(f"📊 训练数据 CSV 已保存: {csv_path}")


def str2bool(v):
    """
    将命令行字符串参数转换为布尔值（argparse type 回调）

    参数：
        v : str 或 bool
            - 命令行传入时是 str（如 'True', 'False', 'yes', 'no'）
            - 代码内直接调用时可能是 bool（直接原样返回）

    返回：
        bool — True 或 False

    异常：
        argparse.ArgumentTypeError — 当传入的字符串无法识别时抛出

    支持的写法（不区分大小写）：
        True 写法： 'yes', 'True', 'true', 'TRUE', 't', 'y', '1'
        False 写法：'no', 'False', 'false', 'FALSE', 'f', 'n', '0'

    示例：
        >>> str2bool('True')
        True
        >>> str2bool('no')
        False
        >>> str2bool(True)    # 已经是 bool，直接返回
        True
        >>> str2bool('abc')   # 无法识别
        argparse.ArgumentTypeError: Boolean value expected.
    """
    # ① 如果调用方传入的本身就是一个 bool（不是命令行字符串），
    #    直接返回，不做任何转换。这是防御性编程：
    #    例如代码内可能写 parser.set_defaults(render=True)，
    #    此时 v 已经是 True，不需要匹配字符串
    if isinstance(v, bool):
        return v

    # ② .lower() 先转小写，实现"不区分大小写"匹配
    #    元组中既包含小写也包含大写形式，是因为：
    #    - Python 的 str.lower() 对纯 ASCII 100% 可靠
    #    - 但某些终端/Shell（PowerShell、中文 cmd）可能在传参时
    #      引入不可见字符或非标准编码，导致 .lower() 不完全生效
    #    - 同时在元组中保留大小写变体，相当于加了双重保险
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        # ③ 无法匹配时抛 argparse 专用异常，而非 ValueError
        #    argparse 会捕获这个异常并用格式化输出提示用户：
        #    "usage: main.py [-h] [--render str2bool] ..."
        #    "main.py: error: argument --render: Boolean value expected."
        raise argparse.ArgumentTypeError('Boolean value expected.')
