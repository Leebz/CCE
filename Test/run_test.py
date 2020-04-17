import numpy as np
import torch
import gym
import os

from Tools import utils
from RL_Brains import TD3
from Enviroments import CCE_ENV

from Configs.config import Config

import matplotlib.pyplot as plt
from Tools.plot import plotLearning

if __name__ == "__main__":

    # 获取配置
    config = Config()

    # 检查文件夹
    if not os.path.exists("../Results"):
        os.makedirs("../Results")

    if config.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # 初始化环境及参数
    env = CCE_ENV.Env(config.seed, egde_capacity=config.initCapacity)
    state_dim = config.state_dim
    action_dim = config.action_dim
    max_action = config.max_action
    print(state_dim, action_dim, max_action)

    if config.seed is not None:
        # env.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    kwargs = {
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
        "max_action": config.max_action,
        "discount": config.discount,
        "tau": config.tau,
        "policy_noise": config.policy_noise,
        "noise_clip": config.noise_clip,
        "policy_freq": config.policy_freq,
        "lr": config.lr
    }

    agent = TD3.TD3(**kwargs)
    # agent = DDPG.DDPG(**kwargs)

    # 载入模型
    if config.load_model is not None:
        agent.load(f"./models/{config.load_model}")

    # 初始化经验池
    replay_buffer = utils.ReplayBuffer(state_dim=config.state_dim,
                                       action_dim=config.action_dim,
                                       max_size=config.memory_size)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    reward_trace = []

    for t in range(config.max_timesteps):
        episode_timesteps += 1
        # 选择动作
        if t < config.start_timesteps:
            action = env.action_sample()
        else:
            action = agent.select_action(np.array(state))
            noise = np.random.normal(0, config.max_action * config.expl_noise, size=config.action_dim)
            action = action + noise
            action = action.clip(0, config.max_action)
            # action = (
            #     agent.select_action(np.array(state))
            #     + np.random.normal(0, config.max_action * config.expl_noise, size=config.action_dim)
            # ).clip(0, config.max_action)

        # 执行动作
        next_state, reward, done = env.step(action)
        done_bool = float(done)

        # 存储到经验池中
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # 训练
        if t >= config.start_timesteps:
            agent.train(replay_buffer, config.batch_size)

        if done:
            print(f"Total timesteps: {t+1}, Episode Num: {episode_num+1} Episode Steps: {episode_timesteps} Reward: {episode_reward}")
            print(env.C_TRACE)
            print(env.A_TRACE)
            print("==========================================================================================================")
            reward_trace.append(episode_reward)

            # Reset Env
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            if episode_num % 10 == 0:
                plotLearning(reward_trace, filename="../Results/seed-0-600.png")
























