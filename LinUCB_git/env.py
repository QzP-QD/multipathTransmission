from policy import linucb_policy
from tcppacket import tcp_packet


class tmp_env:
    c1, c2, c3 = 0.5, 0.7, 0.9

    def __init__(self, K_arms, d):
        self.tref = -1
        self.linucb_policy_object = linucb_policy(K_arms=K_arms, d=d)
        self.gama = 1
        self.total_reward = 0

        # 模拟有两千个数据包待传输
        self.packets = [tcp_packet(200) for i in range(2000)]

    def get_t_ref(self):
        # ！！！临时设定rttf和rtts以及对应的标准差，后续添加具体观测步骤
        rttf = 0.1
        rtts = 0.2

        sigmaf = 0
        sigmas = 0
        self.tref = max(2 * (rttf + sigmaf), rtts + sigmas)

    def get_total_reward(self):
        for packet in self.packets: # 可能无法侦测到全部传输完成的数据包的情况
            self.get_t_ref()
            while packet.get_t_elap() < 3 * self.tref & packet.status == 2:
                r = packet.get_reward()
                self.total_reward += self.gama * r
                if packet.get_t_elap() <= self.tref:
                    self.gama *= tmp_env.c1
                elif packet.get_t_elap() <= 2*self.tref:
                    self.gama *= tmp_env.c2
                else:
                    self.gama *= tmp_env.c3
