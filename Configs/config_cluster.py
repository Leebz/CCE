from terminaltables import AsciiTable


class Config(object):
    def __init__(self):
        self.initCapacity = 60

        self.task_length_mean = 5
        self.task_length_std = 5
        self.price_mean = 5
        self.price_std = 1

        self.state_dim = 5
        self.action_dim = 1
        self.max_action = 1.0

        self.seed = 0                            # 随机种子
        self.start_episode = int(5e1)
        self.eval_freq = 50                  # 策略评估频率
        self.max_episode = int(1e6)               # 总步数
        self.expl_noise = 0.1                       # 高斯噪声标准差
        self.batch_size = 128                       # Batch Size
        self.discount = 0.99                        # 折扣因子
        self.tau = 0.005                            # Target网络更新率
        self.lr = 4e-3

        self.policy_noise = 0.2 * self.max_action   # Critic更新时对target网络添加的噪声
        self.noise_clip = 0.5 * self.max_action     # 策略噪声剪切范围
        self.policy_freq = 2                        # 策略延迟更新频率

        self.memory_size = int(1e6)

        self.save_model = False                     # 保存模型
        self.load_model = None

        self.show_config()

    def show_config(self):
        config_data = [['PARAMETER', 'VALUE']]
        var_dict = vars(self)

        for field in var_dict:
            config_data.append([field, var_dict[field]])

        # print(config_data)
        data = AsciiTable(config_data)
        print(data.table)


if __name__ == "__main__":
    config = Config()





