import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from 马尔科夫决策过程.Bellman import MDP, S, A, P, R, join

gamma = 0.5  # 折扣因子


def value_iteration(MDP, theta=1e-6):
    """价值迭代算法，求解最优状态价值函数和最优策略"""
    S, A, P, R, gamma = MDP
    # 初始化状态价值为0
    V = {s: 0.0 for s in S}

    while True:
        delta = 0.0  # 记录本轮迭代的最大变化量
        V_new = {}
        for s in S:
            if s == "s5":  # 终止状态，价值恒为0
                V_new[s] = 0.0
                continue
            # 计算当前状态s下所有可能动作的动作价值
            q_values = []
            for a in A:
                # 检查该状态-动作对是否在策略中定义
                key = join(s, a)
                if key not in R:
                    continue
                # 计算 Q(s, a) = r(s, a) + gamma * sum_{s'} P(s'|s, a) * V(s')
                q_sa = R[key]
                for s_next in S:
                    trans_key = join(key, s_next)
                    if trans_key in P:
                        q_sa += gamma * P[trans_key] * V[s_next]
                q_values.append((a, q_sa))
            # 最优价值 = max_a Q(s, a)
            if q_values:
                V_new[s] = max(q for _, q in q_values)
            else:
                V_new[s] = 0.0
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < theta:  # 收敛条件
            break
    return V


def extract_optimal_policy(MDP, V):
    """根据最优状态价值函数提取最优策略（确定性策略）"""
    S, A, P, R, gamma = MDP
    optimal_policy = {}  # 状态 -> 最优动作
    max_action_value = {}  # 状态 -> 最大动作价值

    for s in S:
        if s == "s5":  # 终止状态，无需动作
            optimal_policy[s] = "终止"
            max_action_value[s] = 0.0
            continue

        best_action = None
        best_q_value = float('-inf')
        action_details = {}  # 记录每个动作的Q值用于展示

        for a in A:
            key = join(s, a)
            if key not in R:
                continue
            # 计算 Q(s, a) = r(s, a) + gamma * sum_{s'} P(s'|s, a) * V(s')
            q_sa = R[key]
            for s_next in S:
                trans_key = join(key, s_next)
                if trans_key in P:
                    q_sa += gamma * P[trans_key] * V[s_next]
            action_details[a] = q_sa
            if q_sa > best_q_value:
                best_q_value = q_sa
                best_action = a

        optimal_policy[s] = best_action
        max_action_value[s] = best_q_value

    return optimal_policy, max_action_value


def generate_optimal_sequence(MDP, optimal_policy, start_state):
    """从给定起始状态开始，按照最优策略生成状态-动作序列，直到终止状态"""
    S, A, P, R, gamma = MDP
    sequence = []
    s = start_state
    max_steps = 20  # 防止无限循环

    for _ in range(max_steps):
        if s == "s5" or optimal_policy.get(s) == "终止":
            sequence.append((s, "终止", 0))
            break
        a = optimal_policy[s]
        r = R.get(join(s, a), 0)
        sequence.append((s, a, r))

        # 根据确定性转移找到下一个状态
        s_next = None
        for s_opt in S:
            trans_key = join(join(s, a), s_opt)
            if P.get(trans_key, 0) > 0:
                # 对于确定性转移（概率=1），直接取该下一状态
                if P[trans_key] == 1.0:
                    s_next = s_opt
                    break
        # 如果没有找到确定性转移，取概率最大的
        if s_next is None:
            best_prob = 0
            for s_opt in S:
                trans_key = join(join(s, a), s_opt)
                if P.get(trans_key, 0) > best_prob:
                    best_prob = P[trans_key]
                    s_next = s_opt
        s = s_next
    return sequence


if __name__ == "__main__":
    print("=" * 60)
    print("马尔可夫决策过程 —— 最优策略求解")
    print("=" * 60)

    # 1. 使用价值迭代求解最优状态价值函数
    V_star = value_iteration(MDP, theta=1e-6)
    print("\n1. 最优状态价值函数 V*(s):")
    print("-" * 40)
    for s in S:
        print(f"   V*({s}) = {V_star[s]:.4f}")

    # 2. 提取最优策略和每个状态的最大动作价值
    optimal_policy, max_action_value = extract_optimal_policy(MDP, V_star)
    print("\n2. 最优策略与最大动作价值:")
    print("-" * 40)
    for s in S:
        if s == "s5":
            print(f"   {s}: 终止状态, 最大动作价值 = {max_action_value[s]:.4f}")
        else:
            q_val = max_action_value[s]
            print(f"   {s}: 最优动作 = [{optimal_policy[s]}], "
                  f"max_a Q({s}, a) = {q_val:.4f}")

    # 3. 输出每个状态所有动作的Q值对比
    print("\n3. 各状态动作价值 Q(s, a) 详细对比:")
    print("-" * 40)
    for s in S:
        if s == "s5":
            print(f"   {s}: 终止状态，无可用动作")
            continue
        print(f"   {s}:")
        for a in A:
            key = join(s, a)
            if key not in R:
                continue
            # 计算 Q(s, a)
            q_sa = R[key]
            for s_next in S:
                trans_key = join(key, s_next)
                if trans_key in P:
                    q_sa += gamma * P[trans_key] * V_star[s_next]
            marker = " ★ 最优" if a == optimal_policy[s] else ""
            print(f"      Q({s}, {a}) = {q_sa:.4f}{marker}")

    # 4. 输出从每个非终止状态出发的最优序列
    print("\n4. 从各状态出发的最优策略序列:")
    print("-" * 40)
    for start_s in S:
        if start_s == "s5":
            print(f"   {start_s} → [已是终止状态]")
            continue
        seq = generate_optimal_sequence(MDP, optimal_policy, start_s)
        seq_str = " → ".join(f"{s}[{a}]" for s, a, r in seq)
        total_reward = sum(r for _, _, r in seq)
        print(f"   从 {start_s} 出发: {seq_str}")
        print(f"   累计奖励: {total_reward}")

    print("\n" + "=" * 60)
    print("求解完成！")
    print("=" * 60)
