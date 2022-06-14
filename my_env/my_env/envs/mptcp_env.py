import gym
from gym import spaces


class MptcpEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # 动作空间
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,)) # 状态空间
        print("init")

    def step(self, action):
        print("step", action)
        obs, reward, done, info = 1,2,3,4
        return obs, reward, done, info

    def reset(self):
        print("reset")

    def render(self, mode='human'):
        print('render')

    def close(self):
        print('close')