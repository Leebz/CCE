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
                 backhaul_bandwidth,
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
        self.backhaul_bandwidth = backhaul_bandwidth
        self.user_number = user_number
        self.edge_com_capacity = edge_com_capacity
        self.cloud_com_capacity = cloud_com_capacity

        self.user_ptr = 0
        self.ep_r = 0
        self.ep_r_trace = []
        self.done = False

        self.edge_usage_record = []  # 边缘服务器使用记录
        self.cloud_usage_record = []  # 云服务器 使用记录
        self.user2edge_trans_record = []  # 记录从用户设备到边缘服务器传输数据剩余大小
        self.edge2cloud_trans_record = []  # 记录从边缘结点到云服务器传输数据剩余大小

        """
        Edge Computing Capacity
        Cloud Computing Capacity
        Task Size
        Task Com Demand
        Delay Demand
        """
        self.state = []

    def reset(self):
        self.user_ptr = 0
        # self.waiting_buffer = []
        self.ep_r = 0
        self.ep_r_trace = []
        self.done = False
        # self.state = [self.cloud_com_capacity, self.edge_com_capacity, task_info[0], task_info[1]] # mlgb
        self.state = [self.cloud_com_capacity, self.edge_com_capacity]

    def step(self, action):
        # 传输完成 进行决策操作 （任务分割比例，计算能力分割比例）
        task = self.generate_task_info()
        print(task)

        split_ratio = action[0]
        allocated_resource_ratio = action[1]

        reward = 0

        if self.user_ptr == self.user_number:
            done = True
        else:
            done = False
            self.user_ptr += 1

        return self.state, reward, done

    def transmit_device2edge(self):
        finished_task_index = -1
        one_finished = False
        for task_info in self.user2edge_trans_record:
            if task_info[1] > task_info[3]:
                task_info[3] = task_info[3] + self.bandwidth * 0.1
            elif not one_finished:
                # 返回队列中第一个已完成的序号, -1表示没有完成传输的任务
                finished_task_index = task_info[0]
                one_finished = True

        return finished_task_index

    def delete_trans_record(self, task_index):
        res = []
        for task in self.user2edge_trans_record:
            if task[0] is not task_index:
                res.append(task)

        self.user2edge_trans_record = res

    def transmit_edge2cloud(self):
        pass

    def update_waiting_buffer(self):
        pass

    def generate_task_info(self):
        # 生成当前步的数据
        task_rate = np.random.rand()
        if task_rate > 0.7:
            task_size = int(np.random.normal(self.task_size_mean, self.task_size_std))
            task_com = int(np.random.normal(self.task_com_mean, self.task_com_std))
            # task_delay = np.round(np.random.normal(self.task_delay_demand_mean, self.task_delay_demand_std), 2)
            return [task_size, task_com]
        else:
            return None

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
              backhaul_bandwidth=100,
              user_number=10,
              edge_com_capacity=1e8,
              cloud_com_capacity=1e9)

    env.reset()
    env.user2edge_trans_record = [[0, 120, 100, 0], [1, 500, 400, 0]]

    for i in range(110):
        is_finished = env.transmit_device2edge()
        if is_finished is not -1:
            env.delete_trans_record(is_finished)
        print(is_finished, env.user2edge_trans_record)






