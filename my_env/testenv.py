import gym

env = gym.make('my_env:mptcp-v0')
env.reset()
for _ in range(100):
    env.render()
    action = env.action_space.sample()
    print(action)
    env.step(action)
env.close()
