import numpy as np

from DoublePath.MultiPath import Solver


class LinUCB(Solver):
    """LinUCB算法的具体实现类，继承自Solver类"""

    def __init__(self, multienv):
        super(LinUCB, self).__init__(multienv)

        # alpha的值后面要改成通过Peekaboo提出的 算法3计算得到
        self.alpha = 0.25
        # 正反馈和负反馈的奖励程度
        self.r1 = 1
        self.r0 = 0
        self.d = 6

        # Aa 矩阵集合，用于计算每个不相交的臂的Aa矩阵(d*d)
        self.Aa = {}
        # AaI：存储所有Aa矩阵的逆
        self.AaI = {}
        # ba：向量集合，存储所有向量的ba向量(d*1)
        self.ba = {}
        # theta：记录每个动作的参数
        self.theta = {}

        self.a_max = 0
        self.x = None
        self.xT = None

    """
    初始化所有矩阵
    
    art 动作集的关键字/序号 集合——暂时定义为 路径编号
    """
    def set_articles(self, art):
        for key in art:
            self.Aa[key] = np.identity(self.d) # Aa初始化为单位矩阵
            self.ba[key] = np.zeros((self.d, 1))\

            self.AaI[key] = np.identity(self.d)
            self.theta[key] = np.zeros((self.d, 1))

    """
    计算推荐结果
    
    user_features 上下文特征
    articles 可选的动作的关键字/序号 的集合
    timestamp 时间戳
    
    这里对于所有的arm，都使用了同一个特征，小trick
    """
    def recommend(self, user_features, articles, timestamp=0):
        xaT = np.array([user_features])  # d*1
        xa = np.transpose(xaT)

        # 计算出当前可用的动作对应的AaI
        AaI_tmp = np.array([self.AaI[article] for article in articles])
        theta_tmp = np.array([self.theta[article] for article in articles])

        # 计算得到置信区间上界最大的动作的关键字/序号
        art_max = articles[np.argmax(
            np.dot(xaT, theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT, AaI_tmp), xa))
        )]

        self.x = xa
        self.xT = xaT
        self.a_max = art_max
        return self.a_max

    def update(self, reward):
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

    """
    进行一次决策的整合流程：
        更新参数、更新奖励、做出实时决策等
    """
    def run_one_step(self):
        user_features = {}  # 通过 self.multienv 获取特征向量
        articles = {}   # 获取行动的字典key
        self.recommend(user_features, articles)

        self.multienv.transmit(self.a_max)

