import numpy as np


class Env(object):
    def __init__(self,
                 seed,
                 task_size_mean,
                 task_size_std,
                 task_computation_demand_mean,
                 task_computation_demand_std,
                 task_delay_demand_mean,
                 task_delay_demand_std,
                 bandwidth,
                 user_number,
                 edge_com_capacity,
                 cloud_com_capacity,
                 ):
        self.seed = seed
        np.random.seed(self.seed)

        self.task_size_mean = task_size_mean
        self.task_size_std = task_size_std

        self.task_com_mean = task_computation_demand_mean
        self.task_com_std = task_computation_demand_std

        self.task_delay_demand_mean = task_delay_demand_mean
        self.task_delay_demand_std = task_delay_demand_std

        self.bandwidth = bandwidth
        self.user_number = user_number
        self.edge_com_capacity = edge_com_capacity
        self.cloud_com_capacity = cloud_com_capacity

        self.user_ptr = 0
        self.waiting_buffer = []
        self.ep_r = 0
        self.ep_r_trace = []
        self.done = False

        self.edge_usage_record = []
        self.cloud_usage_record = []


        """
        Edge Computing Capacity
        Cloud Computing Capacity
        Task Size
        Task Com Demand
        Delay Demand
        """
        self.state = []

    def reset(self):
        np.random.seed(self.seed)

        self.user_ptr = 0
        self.waiting_buffer = []
        self.ep_r = 0
        self.ep_r_trace = []
        self.done = False

        task_info = self.generate_task_info()
        self.state = [self.cloud_com_capacity, self.edge_com_capacity, task_info[0], task_info[1], task_info[2]]

    def step(self, action):
        split_action = action[0]
        allocate_action = action[1]

        reward = 0
        if self.user_ptr == self.user_number:
            done = True

        return self.state, reward, done

    def update_waiting_buffer(self):
        pass

    def generate_task_info(self):
        # 生成当前步的数据
        task_size = int(np.random.normal(self.task_size_mean, self.task_size_std))
        task_com = int(np.random.normal(self.task_com_mean, self.task_com_std))
        task_delay = np.round(np.random.normal(self.task_delay_demand_mean, self.task_delay_demand_std), 2)

        return [task_size, task_com, task_delay]

    def update_state(self, action):
        pass

    def update_edge_record(self):
        pass

    def update_cloud_record(self):
        pass


if __name__ == '__main__':
    env = Env(seed=1,
              task_size_mean=1000,
              task_size_std=500,
              task_computation_demand_mean=1000000,
              task_computation_demand_std=1000,
              task_delay_demand_mean=1,
              task_delay_demand_std=0.5,
              bandwidth=50,
              user_number=10,
              edge_com_capacity=1e8,
              cloud_com_capacity=1e9)

    env.reset()

    for i in range(env.user_number):
        env.step()
    print("----------")
    for i in range(env.user_number):
        env.step()

