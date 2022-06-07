import numpy as np
from matplotlib import pyplot as plt

from MultiArmBandit import Solver, BernoulliBandit, plot_results


class EpsilonGreedy(Solver):
    """epsilon贪心算法，继承Solver类"""

    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估计值（初始化为1还是0？？？）
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆——探索
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 执行，并获取本次动作奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


class DecayingEpsilonGreedy(Solver):
    """epsilon随时间衰减的epsilon-贪婪算法，继承Sovler类"""
    def __init__(self, bandit, init_prob = 1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob]*self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1/self.total_count:     # epsilon的值随时间衰减
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆——探索
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 执行，并获取本次动作奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)

# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# epsilon_greedy_solver_list = [
#     EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
# ]
# epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
# for solver in epsilon_greedy_solver_list:
#     solver.run(5000)
#
# plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])