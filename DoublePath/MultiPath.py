import numpy as np
import math

"""
    下载数据的总大小为2MB，即2*2^20*8 bit
"""


class Message:
    """数据包类，对数据包的状态进行存储"""

    def __init__(self, size=16):
        self.size = size # 单个数据包大小，默认为最大TCP报文长度，16表示是2的16次方Byte
        """
        设定每个数据包所处的状态
            0：未发送
            1：正在发送，未被接收
            2：已被接收，发送方未ack
            3：已接收，发送方ack
            5：丢失
        """
        self.status = 0
        self.telap = 0   # 数据包发送到链路上之后的时间（决策之后经过的时间）
        self.tack = 0   # 数据包发送之后，到其被发送方确认经过的时间


class Path:
    """路径类，一个对象表示一条路径

    初始化路径特征，包括：
    拥塞窗口 CWND
    正在传输过程中的数据包数量，InP，Inflight Packets
    发送窗口 SWND（通过TCP接收方的反馈信息传递给发送端，widow字段）
    往返时延 RTT，单位ms

    带宽      bandwidth-bwidth，单位Mbps
    单向传播时延    OWD，单位ms
    随机丢包率   randomloss-rloss，单位%

    rtt variation：rtt变化率？？？
    """

    def __init__(self, owd,
                 loss, lossmax, lossstep,
                 bwidth=2, ):
        self.cwnd = 1
        self.inp = 0
        self.swnd = 1

        self.bwidth = bwidth
        self.owd = owd

        # 随机丢包率的起始（最低）值，最大值，变化步长
        self.loss = loss
        self.lossmax = lossmax
        self.lossstep = lossstep

        # 往返时延的起始（最低）值，最大值，变化步长
        self.rtt = 1000

        # 正在该路径上传输的数据包列表
        self.packetinflight = []

    def random_change(self, message):
        # 传输时延 单位 ms
        propdelay = math.pow(2, message.size + 3) / (self.bwidth * 1000)
        self.rtt = propdelay + 2 * self.owd

        # 传输数据包的逻辑


class MultiEnv:
    """多路径传输场景"""

    def __init__(self, K=2):
        # 当前仿真环境的路径条数
        self.K = K
        self.paths = {}

        # 初始化两条路径的参数
        self.paths[0] = Path()
        self.paths[1] = Path()

    """
    调度器在 时刻t 选择了 第k条 路径,据选择在其上传输数据并获得反馈奖励
    ——传输数据包，修改路径状态，反馈奖励
    """

    def transmit(self, k):
        mess = Message()
        # 将数据包分发给对应的路径并获取奖励的逻辑


class Solver:
    """解决算法基本框架"""

    def __init__(self, multienv):
        self.multienv = multienv
        # 每条路径的尝试次数
        self.counts = np.zeros(self.multienv.K)
        # 进行到当前步骤的累积奖励
        self.reward = 0.
        # 维护一个列表，记录每一步的选择
        self.actions = []
        # 维护一个列表，记录每一步的奖励
        self.rewards = []

    def update_reward(self, k):
        # 计算累积奖励并返回，k为本次决策选择的拉杆的编号
        self.reward += self.multienv.transmit(k)
        self.rewards.append(self.reward)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆，由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数，num_steps为总运行次数
        for _ in range(num_steps):
            # 新建一些 数据包，模拟在这个时间点上待传输的数据包集合

            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_reward(k)
