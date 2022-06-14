from gym.envs.registration import register

register(
    id='mptcp-v0',
    entry_point='my_env.envs:MptcpEnv'
)