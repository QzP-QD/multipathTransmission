import numpy as np

from policy import linucb_policy


def ctr_simulator(K_arms, d, alpha, data_path):
    # 初始化策略
    linucb_policy_object = linucb_policy(K_arms=K_arms, d=d)

    # Instantiate trackers 实例化跟踪器
    aligned_time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []
    unaligned_ctr = []  # for unaligned time steps

    # Open data
    with open(data_path, "r") as f:

        for line_data in f:

            # 1st column: 记录中所选的臂
            # int数据类型
            data_arm = int(line_data.split()[0])

            # 2nd column: 记录中所选的臂所对应的回报奖励
            # Float data type
            data_reward = float(line_data.split()[1])

            # 更新alpha的值
            i = 0.1
            linucb_policy_object.alpha = float(i) / np.sqrt(aligned_time_steps + 1)

            # 3rd columns onwards: 100个协变量，记录的是特征向量
            # 鉴于上下文，即协变量（特征向量），我们找到一个句柄
            covariate_string_list = line_data.split()[2:]
            data_x_array = np.array([float(covariate_elem) for covariate_elem in covariate_string_list])

            # 根据策略进行决策
            arm_index = linucb_policy_object.select_arm(data_x_array)

            # 检查arm_index与data_arm是否匹配，即是否选择了与dataset中相同的操作
            # 注意：arm_index索引为0~9，data_arm索引为1~10
            if arm_index + 1 == data_arm:
                # 使用 奖励信息 更新所选的臂
                linucb_policy_object.linucb_arms[arm_index].reward_update(data_reward, data_x_array)

                # 计算点击率
                aligned_time_steps += 1
                cumulative_rewards += data_reward
                aligned_ctr.append(cumulative_rewards / aligned_time_steps)

    return aligned_time_steps, cumulative_rewards, aligned_ctr, linucb_policy_object
