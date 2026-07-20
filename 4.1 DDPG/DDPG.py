"""
================================================================================
DDPG (Deep Deterministic Policy Gradient) —— 深度确定性策略梯度
================================================================================

【算法背景】
DDPG 是由 DeepMind 在 2016 年提出的算法，它将 DQN 的成功经验（经验回放、
目标网络）与 DPG（确定性策略梯度）结合，专门解决连续动作空间的强化学习问题。

【为什么需要 DDPG？】
  - DQN 只能处理离散动作（如上下左右），无法应对连续控制（如方向盘角度、油门大小）
  - 随机策略梯度（如 REINFORCE, A2C/PPO）需要从策略分布中采样，方差大
  - DPG 直接输出确定性动作，方差小，更适合高维连续控制

【四个核心网络】
  ┌──────────────┐              ┌──────────────┐
  │  Actor μ(s)  │ ─── 动作 ──→ │ Critic Q(s,a)│
  │  当前策略网络  │              │  当前Q值网络   │
  │  正在被训练   │              │  正在被训练   │
  └──────┬───────┘              └──────┬───────┘
         │ 软更新                      │ 软更新
  ┌──────┴───────┐              ┌──────┴───────┐
  │ Actor μ'(s)  │              │ Critic Q'(s,a)│
  │  目标策略网络  │              │  目标Q值网络   │
  │  (冻结,缓慢跟踪)│              │  (冻结,缓慢跟踪)│
  └──────────────┘              └──────────────┘

【训练五步曲】
  Step 1: Actor 输出 a = μ(s)，加高斯噪声 → 环境执行 → 存入回放池
  Step 2: 从回放池随机采样一个 batch (s, a, r, s', dw)
  Step 3: 计算 TD 目标: y = r + γ·Q'(s', μ'(s'))    ← 用两个目标网络！
  Step 4: 更新 Critic: 最小化 L = MSE(Q(s,a), y)
  Step 5: 更新 Actor: 最大化 J = Q(s, μ(s))         ← 确定性策略梯度
  Step 6: 软更新目标网络: θ' ← τ·θ + (1-τ)·θ'

【关键技巧】
  1. 经验回放（Experience Replay）：打破数据相关性，提高样本效率
  2. 目标网络（Target Network）：固定 TD 目标，避免"追逐移动靶子"
  3. 软更新（Soft Update）：目标网络平滑跟踪，训练更稳定
  4. OU 噪声/Gaussian 噪声：连续动作空间的探索策略
================================================================================
"""

from utils import Actor, Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy


# ============================================================
# DDPG_agent：DDPG 智能体
# ============================================================
# 整合了 DDPG 算法的全部组件：
#   - 四个网络：Actor + Actor_Target + Critic + Critic_Target
#   - 经验回放池
#   - 动作选择（训练=探索，测试=利用）
#   - 完整训练流程（Critic优化 → Actor优化 → 软更新）
# ============================================================

class DDPG_agent():
    def __init__(self, **kwargs):
        """
        初始化 DDPG 智能体

        通过 **kwargs 一次性接收所有超参数（来自 main.py 的 argparse），
        避免了长参数列表，代码更简洁。
        超参数包括：state_dim, action_dim, max_action, net_width,
                   gamma, a_lr, c_lr, batch_size, noise, dvc 等
        """
        # 将 kwargs 字典中的所有键值对自动变为实例属性
        # 例如 kwargs={'gamma':0.99, 'a_lr':1e-3} →
        #            self.gamma = 0.99, self.a_lr = 1e-3
        self.__dict__.update(kwargs)

        # τ (tau)：目标网络软更新的系数，取值范围 (0, 1)
        # τ=0.005 意味着每步目标网络只向当前网络移动 0.5%
        # 这是平衡训练稳定性和收敛速度的关键超参数
        self.tau = 0.005  # 设置目标网络软更新的参数

        # ============================================================
        # 1. Actor 网络（策略网络）—— 决定"做什么"
        # ============================================================
        # Actor: 状态 s → 确定性动作 a = μ(s)
        # 为什么要用确定性的？因为连续动作空间中采样方差太大，
        # 直接学一个从状态到动作的映射更高效
        self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
        # Adam 优化器：自适应学习率，DDPG 中 Actor 学习率通常设得较小（如 1e-3~1e-4）
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        # Actor 目标网络：与 Actor 结构相同，但参数通过软更新独立维护
        # deepcopy 确保初始参数完全一致（而非共享引用）
        self.actor_target = copy.deepcopy(self.actor)

        # ============================================================
        # 2. Critic 网络（Q 值网络）—— 评估"做得好不好"
        # ============================================================
        # Critic: (状态, 动作) → Q 值（标量）
        # Q 值 = 在状态 s 执行动作 a 后，未来累积折扣奖励的期望
        self.q_critic = Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
        # Critic 学习率通常较 Actor 稍大，因为需要快速逼近真实 Q 值
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        # Critic 目标网络：用于计算 TD 目标，避免训练目标"追着自己跑"
        self.q_critic_target = copy.deepcopy(self.q_critic)

        # ============================================================
        # 3. 经验回放池 —— 存储和回放历史经验
        # ============================================================
        # 容量 500,000 条，足够覆盖大量交互但不会占用过多显存
        # Off-Policy 的核心：当前策略的数据可以反复使用
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(5e5), dvc=self.dvc)

    # ============================================================
    # select_action：动作选择
    # ============================================================
    # 训练模式（deterministic=False）：
    #   在确定性动作基础上添加高斯噪声来探索新行为
    #   a = μ(s) + ε,  其中 ε ~ N(0, σ)
    #   σ = max_action × noise（如 2.0 × 0.1 = 0.2）
    #
    # 评估模式（deterministic=True）：
    #   直接输出确定性动作，衡量策略的真实水平
    # ============================================================
    def select_action(self, state, deterministic):
        # torch.no_grad()：此上下文内不构建计算图，不记录梯度
        # 动作选择是"推理"而非"训练"，不需要反向传播
        with torch.no_grad():
            # 维度变换：[dim] → [1, dim]，增加 batch 维度使网络接受
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from [x,x,...,x] to [[x,x,...,x]]
            # Actor 前向传播 → 确定性动作 → 从 GPU 移回 CPU → 转 numpy → 去掉 batch 维度
            a = self.actor(state).cpu().numpy()[0]  # from [[x,x,...,x]] to [x,x,...,x]

            if deterministic:
                # 【评估/测试】直接返回确定性动作，不探索
                return a
            else:
                # 【训练】添加高斯噪声进行探索
                # 探索是 RL 的核心：如果始终选最优动作，可能错过更好的未知区域
                # 高斯噪声假设最优动作在确定性动作附近，探索范围由 noise 控制
                noise = np.random.normal(0, self.max_action * self.noise, size=self.action_dim)
                # clip：确保加了噪声的动作仍在合法范围 [-max_action, +max_action]
                return (a + noise).clip(-self.max_action, self.max_action)

    # ============================================================
    # train：单步训练（DDPG 的核心）
    # ============================================================
    # 每与环境交互一步就调用一次，从回放池采样 batch 并更新网络。
    # 这是 Off-Policy 的优势：可以在环境交互的同时进行训练，
    # 不需要等一个 episode 结束。
    #
    # 训练分为四步：
    #   ① 计算 TD 目标（用两个目标网络）
    #   ② 更新 Critic（最小化 TD 误差）
    #   ③ 更新 Actor（确定性策略梯度，最大化 Q 值）
    #   ④ 软更新目标网络
    # ============================================================
    def train(self):
        # ------------------------------------------------------------
        # ① 计算 TD 目标（Temporal Difference Target）
        # ------------------------------------------------------------
        # TD 目标是 Critic 要逼近的"正确答案"：
        #   y = r + γ·Q'(s', μ'(s'))
        # 其中 Q' 和 μ' 都是目标网络，计算时不求梯度（torch.no_grad）
        #
        # 为什么用目标网络而不是当前网络？
        #   如果用当前网络 Q(s', μ(s'))，那么 Q(s,a) 和 Q(s', μ(s'))
        #   用的是同一个 Q 网络，训练目标随自身更新不断变化，
        #   类似"追逐自己的尾巴"，极易发散。目标网络固定了"靶子"。
        with torch.no_grad():
            # 从回放池随机采样 batch_size 条历史经验
            # 随机采样 = 打破时间相关性，满足 SGD 的 i.i.d. 假设
            s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

            # ①-a: 用目标 Actor 计算 s' 对应的最优动作
            #       a' = μ'(s')  ← 注意：这是目标策略网络，不是当前策略
            target_a_next = self.actor_target(s_next)

            # ①-b: 用目标 Critic 评估 (s', a') 的 Q 值
            #       Q'(s', μ'(s')) ← 都用目标网络，双重稳定
            target_Q_raw = self.q_critic_target(s_next, target_a_next)

            # ①-c: TD 目标公式：y = r + γ·Q'(s', μ'(s'))
            #       关键细节 (~dw)：
            #         dw=True  → ~dw=False(0) → y = r        （episode结束，无未来奖励）
            #         dw=False → ~dw=True(1)  → y = r + γ·Q' （正常状态，考虑未来）
            #       这种处理方式也叫"终止状态掩码"
            target_Q = r + (~dw) * self.gamma * target_Q_raw  # dw: die or win

        # ------------------------------------------------------------
        # ② 更新 Critic 网络（Q 值网络）
        # ------------------------------------------------------------
        # 目标：让 Critic 对 Q(s,a) 的估计尽可能接近 TD 目标 y
        # 损失函数：均方误差 MSE = (1/B)·Σ(Q(s_i, a_i) - y_i)²
        #
        # 当前 Critic 对这批 (s,a) 的 Q 值估计
        current_Q = self.q_critic(s, a)

        # Critic 损失：MSE（均方误差），衡量估计值与目标的差距
        q_loss = F.mse_loss(current_Q, target_Q)

        # 梯度下降：更新 Critic 参数使 Q(s,a) 更接近 y
        self.q_critic_optimizer.zero_grad()  # 清空上一步的梯度
        q_loss.backward()                     # 反向传播，计算 ∂L/∂θ_Q
        self.q_critic_optimizer.step()        # 更新 θ_Q ← θ_Q - α·∇L

        # ------------------------------------------------------------
        # ③ 更新 Actor 网络（策略网络）—— 确定性策略梯度（DPG）
        # ------------------------------------------------------------
        # 目标：让 Actor 选择能最大化 Q 值的动作
        #
        # 策略梯度的链式法则（确定性版本）：
        #   ∇_θ J = E[∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ(s)]
        #
        # 直观理解：
        #   Q(s, μ(s)) 是"按当前策略选动作能拿多少分"，
        #   我们要调整 Actor 参数让这个分数尽量高。
        #
        #   第一项 ∇_a Q(s,a)：哪个方向的动作能让 Q 值变大（Critic 指导）
        #   第二项 ∇_θ μ(s)：如何调整 Actor 参数让动作往那个方向偏
        #
        # 因为 PyTorch 默认做梯度下降（最小化），而我们想最大化 Q 值，
        # 所以取负号：a_loss = -Q(s, μ(s))
        # 最小化 -Q 等价于最大化 Q

        # 计算 Actor 当前输出的动作（用于后续日志记录）
        actor_current_a = self.actor(s)
        a_loss = -self.q_critic(s, actor_current_a).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------------------------------
        # ④ 软更新（Soft Update）目标网络
        # ------------------------------------------------------------
        # 公式：θ_target = τ·θ_current + (1-τ)·θ_target
        #
        # 为什么用软更新而不是硬更新（完全拷贝）？
        #   DQN 用硬更新（每 N 步 θ_target ← θ_current）：
        #     简单粗暴，但更新时有突变
        #   DDPG 用软更新（每步都更新一点点）：
        #     目标网络像"影子"一样平滑跟随，训练更稳定
        #
        # τ=0.005 → 目标网络每次只移动 0.5%
        # 大约需要 1/τ = 200 步才能基本跟上当前网络
        with torch.no_grad():
            # 逐一更新 Critic 目标网络的每个参数
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # 逐一更新 Actor 目标网络的每个参数
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # ------------------------------------------------------------
        # ⑤ 汇总本次训练的指标（供外部日志使用）
        # ------------------------------------------------------------
        # 返回一个字典，包含 Critic 损失、Actor 损失、
        # 各网络输出的统计量，便于监控训练状态
        train_info = {
            'q_loss':             q_loss.item(),                       # Critic 的 MSE 损失
            'a_loss':             a_loss.item(),                       # Actor 的确定性策略梯度损失
            'current_Q_mean':     current_Q.mean().item(),             # 当前 Critic Q(s,a) 的均值
            'target_Q_mean':      target_Q.mean().item(),              # TD 目标 y 的均值
            'target_Q_raw_mean':  target_Q_raw.mean().item(),          # 目标 Critic Q'(s', μ'(s')) 的均值
            'actor_a_mean':       actor_current_a.mean().item(),       # 当前 Actor 输出动作的均值
            'actor_a_std':        actor_current_a.std().item(),        # 当前 Actor 输出动作的标准差
            'target_a_mean':      target_a_next.mean().item(),         # 目标 Actor 输出动作的均值
            'target_a_std':       target_a_next.std().item(),          # 目标 Actor 输出动作的标准差
            'reward_batch_mean':  r.mean().item(),                     # 采样 batch 的平均奖励
        }
        return train_info

    # ============================================================
    # save：保存模型到硬盘
    # ============================================================
    # state_dict() 只保存网络权重，不保存网络结构。
    # 文件命名格式：{环境名}_{网络类型}{时间步}，如 PV1_actor100.pth
    def save(self, EnvName, timestep):
        torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName, timestep))
        torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName, timestep))

    # ============================================================
    # load：从硬盘加载预训练模型
    # ============================================================
    # map_location=self.dvc 确保在不同设备（GPU/CPU）之间正确加载
    def load(self, EnvName, timestep):
        self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
        self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))


# ============================================================
# ReplayBuffer：经验回放池
# ============================================================
#
# 【经验回放的三大作用】
#   1. 打破时序相关性：连续交互的样本高度相关（前后帧几乎一样），
#      直接用会导致网络对最近的经验"过拟合"，训练不稳定
#   2. 提高数据利用率：每条经验可以被多次采样学习，而非用一次丢弃
#   3. 满足 i.i.d 假设：随机采样使每个 batch 的样本近似独立同分布，
#      这是 SGD 收敛的基本前提
#
# 【实现方式 — 环形缓冲区（Circular Buffer）】
#   像一个固定长度的队列：新数据写入当前指针位置，
#   存满后回到开头覆盖最旧的数据。
#   采样时随机抽取索引，不关心数据存入的先后顺序。
#
# 【性能优化 — GPU 存储】
#   整个 buffer 存在 GPU 显存中（device=self.dvc），
#   采样和训练都在 GPU 进行，避免了 CPU↔GPU 的传输瓶颈。
#   500,000 条 × (state_dim + action_dim + 1 + state_dim + 1) ≈ 几十 MB 显存。
# ============================================================

class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, dvc):
        """
        参数：
            state_dim  : 状态向量的维度（如 Pendulum-v1 为 3）
            action_dim : 动作向量的维度（如 Pendulum-v1 为 1）
            max_size   : 回放池最大容量（默认 500,000 条经验）
            dvc        : 存储设备（torch.device('cuda') 或 torch.device('cpu')）
        """
        self.max_size = max_size  # 缓冲区最大容量
        self.dvc = dvc            # 存储设备
        self.ptr = 0              # 写入指针（当前要写入/覆盖的位置）
        self.size = 0             # 当前有效数据量（从 0 增长到 max_size）

        # 预分配 tensor 存储所有经验数据
        # 形状由 [max_size, dim] 决定，存满后形状不变
        # 数据类型：状态/动作/奖励用 float32，终止标志用 bool
        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)       # 当前状态
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.dvc)       # 执行的动作
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)                # 即时奖励（保存为 1 维）
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)   # 下一状态
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)                # 终止标志（done/win）

    def add(self, s, a, r, s_next, dw):
        """
        存入一条经验元组 (s, a, r, s_next, dw)

        参数：
            s      : 状态 (numpy array, shape=[state_dim])
            a      : 动作 (numpy array, shape=[action_dim])
            r      : 即时奖励 (Python float)
            s_next : 下一状态 (numpy array, shape=[state_dim])
            dw     : 终止标志 (bool, True=episode结束)
        """
        # 每次只放入一个时刻的数据
        # from_numpy：将 numpy 数组零拷贝转换为 torch tensor（共享内存）
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc)  # Note that a is numpy.array
        self.r[self.ptr] = r  # Python 标量直接赋值，PyTorch 自动转为 tensor
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw  # bool→tensor

        # 环形指针前移：ptr + 1，到达 max_size 后回到 0
        self.ptr = (self.ptr + 1) % self.max_size  # 存满了又从头开始存
        # 有效数据量 +1，但不超过 max_size（存满后保持 max_size）
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        随机采样 batch_size 条经验

        为什么是均匀随机采样？：
          - DDPG 使用均匀采样（非优先经验回放），每条经验权重相同
          - 因为 DDPG 是 Off-Policy，当前策略的数据和历史策略的数据都能用

        返回：
            (s, a, r, s_next, dw) 各一个 batch
            每个形状: s[batch_size, state_dim], a[batch_size, action_dim], ...
        """
        # 在 [0, size) 范围内生成 batch_size 个随机整数索引
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        # 按索引取出对应的经验数据（一行即为一条完整经验）
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
