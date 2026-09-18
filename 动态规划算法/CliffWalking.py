import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        # P[state][action] = [(p, next_state, reward, done)]
        # state = i * ncol + j, 即按行展开成一维编号
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    # 防止越界,如果越界则保持在原位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    # 下个状态的二维坐标转换为一维编号
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P

    def print_reward_grid(self):
        """打印4x12的奖励矩阵，展示每个网格的即时奖励值"""
        # 构建标签网格
        grid = [["" for _ in range(self.ncol)] for _ in range(self.nrow)]

        for i in range(self.nrow):
            for j in range(self.ncol):
                if i == self.nrow - 1 and j == 0:       # 起点 (左下角)
                    grid[i][j] = " S "
                elif i == self.nrow - 1 and j == self.ncol - 1:  # 终点 (右下角)
                    grid[i][j] = " G "
                elif i == self.nrow - 1 and 0 < j < self.ncol - 1:  # 悬崖
                    grid[i][j] = " C "
                else:                                     # 普通格子
                    grid[i][j] = " . "

        print("\n" + "=" * 60)
        print("  CliffWalking Env -- 4x12 Grid Layout")
        print("=" * 60)
        print("  Legend: S=Start  G=Goal  C=Cliff  .=Normal")
        print()

        # 列号
        col_header = "      " + "".join([f" c{j:<2d}" for j in range(self.ncol)])
        print(col_header)
        print("    +" + "---" * self.ncol + "+")

        for i in range(self.nrow):
            row_str = f" r{i} |" + "".join(grid[i]) + " |"
            print(row_str)

        print("    +" + "---" * self.ncol + "+")
        print()
        print("  [奖励规则]")
        # 用符号代替状态价值
        print("  . Normal cell: step reward = -1")
        print("  S Start cell : step reward = -1 (same as normal)")
        print("  C Cliff cell : fall into cliff reward = -100, episode ends")
        print("  G Goal cell  : reach goal reward = 0, episode ends")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    env = CliffWalkingEnv()
    env.print_reward_grid()