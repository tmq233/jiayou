"""
================================================================================
DDPG 训练主程序
================================================================================

【文件结构】
  main.py   —— 训练主循环、超参数设置、环境交互
  DDPG.py   —— DDPG 智能体（Actor/Critic 网络、回放池、训练逻辑）
  utils.py  —— Actor/Critic 网络结构、策略评估函数

【支持的 Gym 环境】
  Index 0: Pendulum-v1             —— 倒立摆（入门级连续控制）
  Index 1: LunarLanderContinuous-v3 —— 登月器连续版
  Index 2: Humanoid-v4             —— 人形机器人（高维复杂）
  Index 3: HalfCheetah-v4          —— 半猎豹（MuJoCo 经典基准）
  Index 4: BipedalWalker-v3        —— 双足步行
  Index 5: BipedalWalkerHardcore-v3 —— 双足步行困难版

【运行示例】
  python main.py                           # 默认：Pendulum-v1, CUDA
  python main.py --EnvIdex 1 --dvc cpu     # LunarLander, CPU
  python main.py --EnvIdex 3 --render      # HalfCheetah + 渲染
  python main.py --Loadmodel True          # 加载预训练模型继续训练
"""

from utils import str2bool, evaluate_policy, plot_training_curves
from datetime import datetime
from DDPG import DDPG_agent
import gymnasium as gym
import os, shutil
import argparse
import torch


# ============================================================
# 超参数设置（Hyperparameters）
# ============================================================
# argparse 允许从命令行传入参数，如：
#   python main.py --env Pandulum-v1 --seed 42
# 不传入则使用下面设定的默认值
# ============================================================
parser = argparse.ArgumentParser()

# ---- 运行设备 ----
# 'cuda' = GPU 加速训练（需要 NVIDIA 显卡 + CUDA 版 PyTorch）
# 'cpu'  = CPU 训练（速度较慢，但任何机器都能跑）
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')

# ---- 环境选择 ----
# 0=Pendulum, 1=LunarLander, 2=Humanoid, 3=HalfCheetah, 4=BipedalWalker, 5=BipedalWalkerHardcore
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')

# ---- 日志与可视化 ----
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')  # 是否使用 TensorBoard 记录训练曲线
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')  # 是否渲染环境画面（渲染会大幅降低训练速度）
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')  # 是否加载之前保存的模型
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')  # 加载哪个时间步的模型

# ---- 随机种子 ----
parser.add_argument('--seed', type=int, default=0, help='random seed')  # 固定随机种子以保证可复现性

# ---- 训练步数 ----
parser.add_argument('--Max_train_steps', type=int, default=5e6, help='Max training steps')    # 最大训练步数（500万步）
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')   # 每10万步保存一次模型
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.') # 每2000步评估一次策略

# ---- DDPG 核心超参数 ----
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    # γ（折扣因子）：对未来奖励的衰减系数
    # γ=0.99 → 重视长期奖励（100步后权重≈0.37）
    # γ 越大，策略越"远见"；γ 越小，策略越"短视"
parser.add_argument('--net_width', type=int, default=400, help='Hidden net width, s_dim-400-300-a_dim')
    # 网络宽度：Actor 和 Critic 隐藏层的神经元数
    # 结构：输入 → 400 → 300 → 输出
    # 400 对大多数连续控制任务足够，太大会增加过拟合风险
parser.add_argument('--a_lr', type=float, default=1e-3, help='Learning rate of actor')
    # Actor 学习率：策略网络的优化步长
    # DDPG 中 Actor 学习率通常设得较 Critic 略小（因为策略梯度比 Q 函数梯度噪声更大）
parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic')
    # Critic 学习率：Q 值网络的优化步长
    # Critic 需要快速逼近真实 Q 值，学习率可稍大一些
parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
    # 每步训练从回放池采样的样本数量
    # 128 是常用选择：太大则梯度估计无显著改善但计算变慢，
    #                  太小则梯度估计方差过大，训练不稳定
parser.add_argument('--random_steps', type=int, default=5e4, help='random steps before trianing')
    # 随机探索步数：开始训练前先纯随机采样的步数
    # 这 5 万步用于"预热"回放池，收集足够多样的初始经验再开始学习
    # 如果回放池太小就开始训练，Critic 容易对少量数据过拟合
parser.add_argument('--noise', type=float, default=0.1, help='exploring noise')
    # 探索噪声系数：控制动作探索的幅度
    # 实际噪声标准差 = max_action × noise（如 2.0 × 0.1 = 0.2）
    # noise 越大，探索越多（可能发现更好策略，但也可能乱跑不收敛）
    # noise 越小，越依赖已有知识（可能陷入局部最优）

opt = parser.parse_args()

# 如果请求 CUDA 但 PyTorch 不支持 CUDA，自动回退到 CPU
if opt.dvc == 'cuda' and not torch.cuda.is_available():
    print("CUDA is not available, falling back to CPU.")
    opt.dvc = 'cpu'
opt.dvc = torch.device(opt.dvc)  # 从字符串 'cuda'/'cpu' 转为 PyTorch device 对象
print(opt)  # 打印所有参数，方便检查配置


# ============================================================
# main：训练主函数
# ============================================================
def main():
    # 环境列表（对应 EnvIdex 0~5）
    EnvName = ['Pendulum-v1', 'LunarLanderContinuous-v3', 'Humanoid-v4',
               'HalfCheetah-v4', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3']
    # 环境简称（用于模型保存/加载的文件名）
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4', 'BWv3', 'BWHv3']

    # ============================================================
    # 1. 构建环境
    # ============================================================
    # 训练环境：如果开启 render，会弹出窗口显示环境画面
    env = gym.make(EnvName[opt.EnvIdex], render_mode="human" if opt.render else None)
    # 评估环境：单独创建一个环境用于评估，不渲染
    eval_env = gym.make(EnvName[opt.EnvIdex])

    # 从环境中获取状态和动作空间的维度信息
    opt.state_dim = env.observation_space.shape[0]    # 状态维度（如 Pendulum: 3）
    opt.action_dim = env.action_space.shape[0]         # 动作维度（如 Pendulum: 1）
    opt.max_action = float(env.action_space.high[0])   # 动作上限（如 Pendulum: 2.0）
    # 注意：这里假设动作空间对称，即 action_space ∈ [-max, +max]
    print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{env._max_episode_steps}')

    # ============================================================
    # 2. 设置随机种子（保证可复现性）
    # ============================================================
    env_seed = opt.seed
    torch.manual_seed(opt.seed)           # PyTorch CPU 随机种子
    torch.cuda.manual_seed(opt.seed)      # PyTorch GPU 随机种子
    torch.backends.cudnn.deterministic = True   # cuDNN 确定性模式（牺牲部分性能换取可复现）
    torch.backends.cudnn.benchmark = False      # 关闭 cuDNN 自动优化（保证确定性）
    print("Random Seed: {}".format(opt.seed))

    # ============================================================
    # 3. 构建 TensorBoard 记录器（可选）
    # ============================================================
    # TensorBoard 可以实时可视化训练曲线（奖励、损失等）
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # ============================================================
    # 4. 构建 DDPG 智能体
    # ============================================================
    if not os.path.exists('model'): os.mkdir('model')  # 创建模型保存目录

    # vars(opt) 将 argparse.Namespace 转为 dict，
    # 然后 ** 解包传给 DDPG_agent 的 __init__(**kwargs)
    agent = DDPG_agent(**vars(opt))  # var: transfer argparse to dictionary

    # 如果指定了加载模型，从硬盘加载预训练权重
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    # ============================================================
    # 5. 主循环：交互 或 训练
    # ============================================================
    if opt.render:
        # ---- 渲染模式：只展示，不训练 ----
        while True:
            score = evaluate_policy(env, agent, turns=1)
            print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
    else:
        # ---- 训练模式 ----
        total_steps = 0  # 全局步数计数器
        train_metrics = {}    # 存储累计的训练指标（用于计算区间均值）
        train_steps_count = 0 # 统计区间内实际训练步数

        # ---- 历史数据收集（用于训练结束后的曲线图） ----
        history = {
            'steps': [], 'ep_r': [],
            'q_loss': [], 'a_loss': [], 'reward_batch_mean': [],
            'current_Q_mean': [], 'target_Q_mean': [], 'target_Q_raw_mean': [],
            'actor_a_mean': [], 'actor_a_std': [],
            'target_a_mean': [], 'target_a_std': [],
        }

        while total_steps < opt.Max_train_steps:
            # 每个 episode 使用递增的 seed，避免每次都从同一个初始状态开始
            s, info = env.reset(seed=env_seed)
            env_seed += 1
            done = False

            # ====== 单个 episode 内的交互与训练 ======
            while not done:
                # ---- 动作选择 ----
                # 前 random_steps（5万步）= 纯随机动作（预热阶段）
                # 之后的步数 = Actor 输出确定性动作 + 高斯噪声（训练阶段）
                if total_steps < opt.random_steps:
                    a = env.action_space.sample()  # 从动作空间中均匀随机采样
                else:
                    a = agent.select_action(s, deterministic=False)  # Actor + 探索噪声

                # ---- 环境交互 ----
                # gymnasium 0.26+ 的 step() 返回 5 个值：
                #   s_next: 下一个状态
                #   r:      即时奖励
                #   dw:     终止/成功（done or win）
                #   tr:     截断（达到最大步数被强制结束）
                #   info:   额外信息
                s_next, r, dw, tr, info = env.step(a)
                done = (dw or tr)  # 任一种结束都视为 episode 终止

                # ---- 存入经验回放池 ----
                # 每步都存储，不管是否开始训练
                # 回放池存满后自动覆盖旧数据
                agent.replay_buffer.add(s, a, r, s_next, dw)

                s = s_next          # 状态转移：当前状态变为下一状态
                total_steps += 1    # 全局步数 +1

                # ---- 训练（从第 random_steps 步开始） ----
                # 预热阶段只收集经验不训练，
                # 等回放池中有足够多样的数据后再开始学习
                if total_steps >= opt.random_steps:
                    info_dict = agent.train()
                    train_steps_count += 1

                    # 累计训练指标（用于计算区间均值）
                    for key, val in info_dict.items():
                        if key not in train_metrics:
                            train_metrics[key] = 0.0
                        train_metrics[key] += val

                # ---- 定期评估与日志输出 ----
                # 每 eval_interval（2000）步评估一次当前策略并输出详细训练日志
                if total_steps % opt.eval_interval == 0:
                    # 计算区间均值（避免除以零）
                    actual_train_steps = max(train_steps_count, 1)

                    # ==================== 策略评估 ====================
                    ep_r = evaluate_policy(eval_env, agent, turns=3)  # 跑 3 轮取平均

                    # ==================== 详细日志输出 ====================
                    # 使用分隔线让日志更易读
                    print(f"\n{'='*80}")
                    print(f"📊 Step {int(total_steps/1000):>6d}k | Env: {BrifEnvName[opt.EnvIdex]:>6s} | "
                          f"Episode Reward: {ep_r:>8.1f}")
                    print(f"{'='*80}")

                    # 输出训练指标（区间均值）
                    avg_metrics = {}
                    if train_metrics:
                        # 计算区间均值
                        avg_metrics = {k: v / actual_train_steps for k, v in train_metrics.items()}
                        print(f"  ┌─ 📉 损失 (Loss) ─────────────────────────────────────────────┐")
                        print(f"  │  Critic Loss (MSE)      : {avg_metrics.get('q_loss', 0):>10.4f}           │")
                        print(f"  │  Actor Loss (-Q mean)   : {avg_metrics.get('a_loss', 0):>10.4f}           │")
                        print(f"  ├─ 🎯 Q 值 (Critic) ─────────────────────────────────────────┤")
                        print(f"  │  Current Q(s,a)  mean   : {avg_metrics.get('current_Q_mean', 0):>10.4f}           │")
                        print(f"  │  Target  Q'(s',a') mean : {avg_metrics.get('target_Q_raw_mean', 0):>10.4f}           │")
                        print(f"  │  TD Target y mean       : {avg_metrics.get('target_Q_mean', 0):>10.4f}           │")
                        print(f"  ├─ 🤖 动作 (Actor) ──────────────────────────────────────────┤")
                        print(f"  │  Actor μ(s)       mean  : {avg_metrics.get('actor_a_mean', 0):>10.4f}           │")
                        print(f"  │  Actor μ(s)       std   : {avg_metrics.get('actor_a_std', 0):>10.4f}           │")
                        print(f"  │  Target Actor μ'(s') mean: {avg_metrics.get('target_a_mean', 0):>10.4f}           │")
                        print(f"  │  Target Actor μ'(s') std : {avg_metrics.get('target_a_std', 0):>10.4f}           │")
                        print(f"  ├─ 🎁 奖励 (Reward) ─────────────────────────────────────────┤")
                        print(f"  │  Batch Reward mean      : {avg_metrics.get('reward_batch_mean', 0):>10.4f}           │")
                        print(f"  └────────────────────────────────────────────────────────────┘")

                    if opt.write:
                        # TensorBoard 记录
                        writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                        for key, val in avg_metrics.items():
                            writer.add_scalar(f'train/{key}', val, global_step=total_steps)

                    # 重置累计指标
                    train_metrics = {}
                    train_steps_count = 0

                    # ---- 记录历史数据（用于最终画图） ----
                    history['steps'].append(int(total_steps / 1000))
                    history['ep_r'].append(ep_r)
                    for key in ['q_loss', 'a_loss', 'reward_batch_mean',
                                'current_Q_mean', 'target_Q_mean', 'target_Q_raw_mean',
                                'actor_a_mean', 'actor_a_std', 'target_a_mean', 'target_a_std']:
                        history[key].append(avg_metrics.get(key, 0))

                # ---- 定期保存模型 ----
                # 每 save_interval（10万）步保存一次模型权重
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))

        # ---- 训练结束：自动生成曲线图 ----
        plot_dir = f'plots/{BrifEnvName[opt.EnvIdex]}'
        plot_training_curves(history, plot_dir, BrifEnvName[opt.EnvIdex])

        env.close()
        eval_env.close()


if __name__ == '__main__':
    main()
