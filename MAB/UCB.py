import numpy as np

from MultiArmBandit import Solver, BernoulliBandit, plot_results


class UCB(Solver):
    """UCB算法，继承Solver类"""

    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算每个拉杆各自的置信上界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)

coef = 1  # 控制不确定性比重的系数（越大，则考虑的不确定性因素比重越大）
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(8000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])
