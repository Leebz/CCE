class Config(object):
    def __init__(self):

        self.state_dim = 2
        self.action_dim = 1
        self.max_action = 1

        self.seed = None                            # 随机种子
        self.start_timesteps = int(1e3)             # 开始学习的步数
        self.eval_freq = int(5e3)                   # 策略评估频率
        self.max_timesteps = int(1e5)               # 总步数
        self.expl_noise = 0.1                       # 高斯噪声标准差
        self.batch_size = 256                       # Batch Size
        self.discount = 0.99                        # 折扣因子
        self.tau = 0.005                            # Target网络更新率

        self.policy_noise = 0.2 * self.max_action   # Critic更新时对target网络添加的噪声
        self.noise_clip = 0.5 * self.max_action     # 策略噪声剪切范围
        self.policy_freq = 2                        # 策略延迟更新频率

        self.show_config()

    def show_config(self):
        print("=================CONFIG:========================")
        print(f"Seed:{self.seed}")
        print(f"Start_timesteps:{self.start_timesteps}")
        print(f"eval_freq:{self.eval_freq}")
        print(f"max_timesteps:{self.max_timesteps}")
        print(f"expl_noise:{self.expl_noise}")
        print(f"batch_size:{self.batch_size}")
        print(f"discount:{self.discount}")
        print(f"tau:{self.tau}")
        print(f"policy_noise:{self.policy_noise}")
        print(f"policy_clip:{self.noise_clip}")
        print(f"policy_freq:{self.policy_freq}")
        print(f"state_dim:{self.state_dim}")
        print(f"action_dim:{self.action_dim}")
        print(f"max_action:{self.max_action}")
        print("==================================================")


if __name__ == "__main__":
    config = Config()





