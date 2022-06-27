import time


class tcp_packet:
    """
        size = 数据包大小
        status = 数据包状态
                0 未发送
                1 已发送未ack
                2 已ack
    """

    def __init__(self, size):
        self.size = size
        self.status = 0
        self.send_time = -1
        self.ack_time = -1
        self.reward = -1

    def send(self):
        t = time.time()
        self.send_time = int(round(t * 1000))

    def ack(self):
        t = time.time()
        self.ack_time = int(round(t * 1000))

    def get_reward(self):
        self.reward = self.size / (self.ack_time - self.send_time)
        return self.reward

    def get_t_elap(self):
        t = time.time()
        cur_time = int(round(t * 1000))

        return cur_time - self.send_time
