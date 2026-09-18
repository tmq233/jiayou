import sys
import os
import numpy as np
import importlib.util

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from 马尔科夫决策过程.Bellman import MDP, Pi_1, Pi_2

# Monte-CarloMethods.py 文件名含连字符，无法直接 import，通过文件路径加载
_mc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Monte-CarloMethods.py")
_mc_spec = importlib.util.spec_from_file_location("MonteCarloMethods", _mc_path)
_mc_module = importlib.util.module_from_spec(_mc_spec)
_mc_spec.loader.exec_module(_mc_module)
sample = _mc_module.sample


def occupancy(episodes, s, a, timestep_max, gamma):
    ''' 计算状态动作对（s,a）出现的频率,以此来估算策略的占用度量 '''
    rho = 0
    total_times = np.zeros(timestep_max)  # 记录每个时间步t各被经历过几次
    occur_times = np.zeros(timestep_max)  # 记录(s_t,a_t)=(s,a)的次数
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


gamma = 0.5
timestep_max = 1000

episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
print(rho_1, rho_2)