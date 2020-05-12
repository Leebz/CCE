import numpy as np
import torch
import gym
import os

from Tools import ReplayBuffer
from RL_Brains import TD3
from Enviroments import CCE_ENV_MODIFIED as E

from Configs.config import Config

from Tools.utils import *


def eval_policy(agent, eval_episode=10):
    envargs = {
        "seed": config.seed+100,
        "edge_capacity": config.initCapacity,
        "task_size_mean": config.task_size_mean,
        "task_size_std": config.task_size_std,
        "task_length_mean": config.task_length_mean,
        "task_length_std": config.task_length_std,
        "price_mean": config.price_mean,
        "price_std": config.price_std
    }
    eval_env = E.Env(**envargs)
    avg_reward = 0
    for _ in range(eval_episode):
        state, done = eval_env.reset(), False
        while not done:
            action = agent.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episode
    print("---------------------------------------")
    print(f"Evaluation over {eval_episode} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

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
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim=config.state_dim,
                                              action_dim=config.action_dim,
                                              max_size=config.memory_size)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    reward_trace = []

    record = -np.infty
    eval_record = []

    filename = sim_file_name_generator(config.seed, config.initCapacity, config.task_size_mean)

    for t in range(config.max_episode):
        while True:
            episode_timesteps += 1
            # 选择动作
            if t < config.start_episode:
                action = env.action_sample()
            else:
                action = agent.select_action(np.array(state))
                noise = np.random.normal(0, config.max_action * config.expl_noise, size=config.action_dim)
                action = action + noise
                action = action.clip(0, config.max_action)

            # 执行动作
            next_state, reward, done, info = env.step(action)
            done_bool = float(done) if episode_timesteps < E.TOTAL_TASK_NUM else 0

            # 存储到经验池中
            if info is not None:
                action = info
            replay_buffer.add(state, action, next_state, reward/10, done_bool)

            state = next_state
            episode_reward += reward

            # 训练
            if t >= config.start_episode:
                agent.train(replay_buffer, config.batch_size)

            if done:
                print(f"Episode Num: {t} Episode Steps: {episode_timesteps} Reward: {episode_reward}")
                print(env.C_TRACE)
                print(env.A_TRACE)
                print("==========================================================================================================")
                reward_trace.append(episode_reward)
                if episode_reward > record:
                    record = episode_reward
                # Reset Env
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                if t % config.eval_freq == 0 and t != 0:
                    # eval_record.append(eval_policy(agent, eval_episode=10))
                    # with open(f"{filename}eval.txt", "w") as f:
                    #     f.write(str(eval_record))
                    # agent.save(f"../Models/Simulation/eval_{config.seed}_{t}")

                    plotLearning(reward_trace, filename=f"{filename}train.png", window=10)
                    with open(f"{filename}record.txt", "w") as f:
                        f.write(f"reward: {record}")

                break




























