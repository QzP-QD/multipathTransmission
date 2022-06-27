import numpy as np


class linucb_disjoint_arm():
    def __init__(self, arm_index, d):
        # 控制“探索-利用”的alpha值，每个arm有自己的alpha（推测）
        self.alpha = 0

        # 初始化拉杆索引
        self.arm_index = arm_index

        # A: (d x d) 矩阵 = D_a.T * D_a + I_d.
        # A的逆矩阵用于 岭回归
        self.A = np.identity(d)

        # b: (d x 1) 向量
        # 等于岭回归语句中 D_a.T * c_a 的值
        self.b = np.zeros([d, 1])

        # 记录选择这个arm的特征值和奖励历史记录
        self.rewards = []
        self.features = []

        self.theta = None

    """
        更新alpha的方法
        对应peekaboo的算法3

        其中的get_score()方法是通过alpha和历史记录计算分数的函数
    """
    def update_alpha(self):
        # gama是算法3中计算alpha用到的参数
        gama = 0.8
        score_alpha = self.get_score(self.alpha)
        for n in range(100):  # 先假设n最大到100，即迭代100次 cand
            alpha_cand = self.alpha + n * gama
            score_cand = self.get_score(alpha_cand)
            if score_cand > score_alpha:
                self.alpha = alpha_cand
                gama = gama / 2
            else:
                break

        while gama > 0.05:
            score1 = self.get_score(self.alpha)  # alpha的分数
            score2 = self.get_score(self.alpha - gama)  # alpha-gama的分数
            score3 = self.get_score(self.alpha + gama)  # alpha+gama的分数

            if score1 <= score2 & score3 <= score2:
                self.alpha = self.alpha - gama
            if score1 <= score3 & score2 <= score3:
                self.alpha = self.alpha + gama

            gama = gama / 2

    def get_score(self, alpha):
        # 通过alpha和历史记录获得score的方法
        # alpha外部传入，各个arm的历史记录自行记录
        return 100

    def calc_UCB(self, x_array):
        # 得到A矩阵的逆以进行岭回归
        A_inv = np.linalg.inv(self.A)

        # 运行岭回归以获得 theta 系数的估计值
        # theta 是一个维度向量 (d x 1)
        self.theta = np.dot(A_inv, self.b)

        # 将 输入特征向量x_array 重塑为维度 (d x 1) 的向量
        x = x_array.reshape([-1, 1])

        # 计算p
        # p 是维度 (1 x 1) 的向量
        p = np.dot(self.theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))

        return p

    def reward_update(self, reward, x_array):
        # 将输入向量维度转为(d x 1)
        x = x_array.reshape([-1, 1])

        # 将新的决策记录到当前arm中
        self.features.append(x)
        self.rewards.append(reward)

        # 更新矩阵A（d*d）的值
        self.A += np.dot(x, x.T)

        # 更新向量b (d x 1)
        # b加上奖励的标量值
        self.b += reward * x
