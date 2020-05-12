import numpy as np
import torch
import gym
import os

from Tools import ReplayBuffer
from RL_Brains import TD3
from Enviroments import CCE_ENV_MODIFIED as E


from Configs.config import Config

from Tools.utils import *

if __name__ == "__main__":

    # 获取配置
    config = Config()

    # 检查文件夹
    if not os.path.exists("../Results"):
        os.makedirs("../Results")

    if config.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # 初始化环境及参数
    envargs = {
        "seed": config.seed,
        "edge_capacity": config.initCapacity,
        "task_size_mean": config.task_size_mean,
        "task_size_std": config.task_size_std,
        "task_length_mean": config.task_length_mean,
        "task_length_std": config.task_length_std,
        "price_mean": config.price_mean,
        "price_std": config.price_std
    }
    env = E.Env(**envargs)

    state_dim = config.state_dim
    action_dim = config.action_dim
    max_action = config.max_action
    print(state_dim, action_dim, max_action)

    if config.seed is not None:
        # env.seed(config.seed)
        torch.manual_seed(config.seed)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    reward_trace = []
    for _ in range(10):

        while True:
            episode_timesteps += 1
            action = np.random.random()
            # 执行动作
            next_state, reward, done, info = env.step(action)

            episode_reward += reward

            if done:
                print(f"Episode Steps: {episode_timesteps} Reward: {episode_reward}")
                reward_trace.append(episode_reward)
                break

    print(reward_trace)
    print(f"avg:{np.mean(reward_trace)}")



























