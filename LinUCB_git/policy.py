import numpy as np

from arm import linucb_disjoint_arm
from tcppacket import tcp_packet


class linucb_policy:
    def __init__(self, K_arms, d):
        self.K_arms = K_arms
        self.linucb_arms = [linucb_disjoint_arm(arm_index=1, d=d) for i in range(K_arms)]
        self.total_reward = 0

    def select_arm(self, x_array):
        # 初始化 ucb
        highest_ucb = -1

        # 具有最大 UCB 的拉杆索引数组
        candidate_arms = []

        for arm_index in range(self.K_arms):
            # 计算每个句柄的 ucb
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)

            # 如果当前拉杆的 ucb 高于当前的highest_ucb
            if arm_ucb > highest_ucb:
                # 设置其为新的最大的 ucb
                highest_ucb = arm_ucb

                # 重置candidate_arms
                candidate_arms = [arm_index]

            # 如果arm的ucb与当前的highest_ucb相同，则将这个arm添加到candidate_arms中
            if arm_ucb == highest_ucb:
                candidate_arms.append(arm_index)

        # 我们从candidate_arms中随机选择一个arm（最终决定）
        chosen_arm = np.random.choice(candidate_arms)

        return chosen_arm
