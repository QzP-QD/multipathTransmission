import numpy as np


class LinUCB:
    def __init__(self):
        self.alpha = 0.25
        self.r1 = 1
        self.r0 = 0
        # 特征维度设为6
        self.d = 6

        # Aa 矩阵集合，用于计算每个臂的不相交部分
        self.Aa = {}
        # AaI：存储所有Aa矩阵的逆
        self.AaI = {}
        # ba：向量集合，保存特征和汇报的乘积
        self.ba = {}
        self.theta = {}

        self.a_max = 0

        self.x = None
        self.xT = None

    def set_articles(self, art):
        for key in art:
            self.Aa[key] = np.identity(self.d)  # 建立单位矩阵
            self.ba[key] = np.zeros((self.d, 1))

            self.AaI[key] = np.identity(self.d)
            self.theta[key] = np.zeros((self.d, 1))

    def update(self,reward):
        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0
            self.Aa[self.a_max] += np.dot(self.x, self.xT)
            self.ba[self.a_max] += r * self.x
            self.AaI[self.a_max] = np.linalg.inv(self.Aa[self.a_max])
            self.theta[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])

        else:
            # error
            pass

    def recommend(self, timestamp, user_features, articles):
        xaT = np.array([user_features]) # d*1
        xa = np.transpose(xaT)

        AaI_tmp = np.array([self.AaI[article] for article in articles])
        theta_tmp = np.array([self.theta[article] for article in articles])
        """代码中有个小trick，及对所有的arm来说，共同使用一个特征，而不是每一个arm单独使用不同的特征"""
        art_max = articles[np.argmax(np.dot(xaT, theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT, AaI_tmp), xa)))]

        self.x = xa
        self.xT = xaT

        self.a_max = art_max

        return self.a_max