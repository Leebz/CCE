import numpy as np
import copy
import random
EDGE_BASIC_COST = 1.0
EDGE_COEFFICIENT = 1
TOTAL_TASK_NUM = 100


class Env(object):
    def __init__(self, seed=0,
                 edge_capacity=600,
                 task_size_mean=10, task_size_std=3,
                 task_length_mean=5, task_length_std=2,
                 price_mean=5, price_std=1
                 ):
        self.seed = seed
        np.random.seed(self.seed)

        self.edge_capacity = edge_capacity
        self.remain_capacity = self.edge_capacity
        self.task_size_mean = task_size_mean
        self.task_size_std = task_size_std
        self.task_length_mean = task_length_mean
        self.task_length_std = task_length_std
        self.price_mean = price_mean
        self.price_std = price_std

        self.task_counter = 0
        self.ep_r = 0
        self.ep_r_trace = []

        self.done = False

        self.state = []
        self.C_TRACE = []
        self.A_TRACE = []

        # 使用记录
        self.usage_record = []

    def reset(self):
        np.random.seed(self.seed)
        self.task_counter = 0
        self.ep_r = 0
        self.ep_r_trace = []
        self.C_TRACE = []
        self.A_TRACE = []
        self.remain_capacity = self.edge_capacity
        self.usage_record = []

        self.done = False
        # Observe the infos
        task_info = self.task_generator()
        cloud_price = self.cloud_price_generator()
        # State: ①remain capacity in edge, ②released VMs, ③task size, ④task length,
        # ⑤public cloud price
        self.state = [self.remain_capacity, 0, task_info[0], task_info[1], cloud_price]
        return copy.deepcopy(self.state)

    def action_sample(self):
        return random.random()

    def step(self, action):
        info = None
        self.A_TRACE.append(action)
        # 更新历史记录,释放租约到期的VM
        # releasedVM = self.update_record()

        # action是把当前请求加载到云服务器的比例。 0代表申请云服务器。
        # 获取当前服务器状态
        state = copy.deepcopy(self.state)
        remain_capacity = state[0]
        task_size = state[2]
        task_length = state[3]
        cloud_price = state[4]

        cost = 0
        # 判断当前动作合法性, 分配到边缘服务器上的VM大小是否大于当前边缘服务器的剩余资源大小
        cloud_vm = int(np.around(task_size * action, 1))
        edge_vm = task_size - cloud_vm

        if edge_vm > remain_capacity:
            # 边缘服务器承载能力不足时 将边缘服务器装满后，剩余卸载到云服务器
            cloud_vm = task_size - remain_capacity
            edge_vm = remain_capacity
            info = cloud_vm / task_size


        # 计算成本(reward)
        remain_capacity = remain_capacity - edge_vm
        self.C_TRACE.append(remain_capacity)
        edge_workload = 1 - remain_capacity / self.edge_capacity
        edge_cost_coefficient = EDGE_COEFFICIENT * (1 + edge_workload) * EDGE_BASIC_COST # 计算出的VM单位价格
        edge_cost = edge_cost_coefficient     # !!!!!!!!!!!!!!!!!
        cloud_cost = cloud_price * cloud_vm * task_length              # 云服务器最终成本

        cost = edge_cost + cloud_cost  # 最终成本

        # 记录使用数据
        self.usage_record.append([edge_vm, task_length])

        # 更新系统状态
        releasedVM = self.update_record()
        remain_capacity += releasedVM

        self.task_counter += 1
        if self.task_counter == TOTAL_TASK_NUM:
            self.done = True
            self.state = [remain_capacity, releasedVM, 0, 0, 0]
        else:
            task_info = self.task_generator()
            cloud_price = self.cloud_price_generator()

            self.state = [remain_capacity, releasedVM, task_info[0], task_info[1], cloud_price]

        next_state = copy.deepcopy(self.state)

        return next_state, -cost, self.done, info

    def update_record(self):
        delete_index = []
        released_vm = 0
        records = copy.deepcopy(self.usage_record)
        for i in range(len(records)):
            records[i][1] = records[i][1] - 1
            if records[i][1] == 0:
                released_vm += records[i][0]
                delete_index.append(i)

        records = [records[i] for i in range(len(records)) if(i not in delete_index)]
        self.usage_record = records
        return released_vm

    def task_generator(self):
        # 生成任务数据, 四舍五入制
        while True:
            task_size = int(np.around(np.random.normal(self.task_size_mean, self.task_size_std), 1))
            if task_size >= 1:
                break

        while True:
            task_length = int(np.around(np.random.normal(self.task_length_mean, self.task_length_std), 1))
            if task_length >= 1:
                break

        return [task_size, task_length]

    def cloud_price_generator(self):
        # 生成公有云当前价格
        while True:
            price = np.around(np.random.normal(self.price_mean, self.price_std), 4)
            if price > 0.0:
                return price

    def edge_first_step(self):
        info = None
        state = copy.deepcopy(self.state)
        remain_capacity = state[0]
        task_size = state[1]
        task_length = state[2]
        cloud_price = state[3]
        cost = 0

        cloud_vm = 0
        edge_vm = task_size
        if remain_capacity < edge_vm:
            cloud_vm = edge_vm - remain_capacity
            edge_vm = task_size - cloud_vm

        # 计算成本(reward)
        remain_capacity = remain_capacity - edge_vm
        self.C_TRACE.append(remain_capacity)
        edge_workload = 1 - remain_capacity / self.edge_capacity
        edge_cost_coefficient = EDGE_COEFFICIENT * (1 + edge_workload) * EDGE_BASIC_COST  # 计算出的VM单位价格
        edge_cost = edge_cost_coefficient * edge_vm  # 边缘最终成本
        cloud_cost = cloud_price * cloud_vm * task_length  # 云服务器最终成本

        cost = edge_cost + cloud_cost  # 最终成本

        # 记录使用数据
        self.usage_record.append([edge_vm, task_length])

        # 更新系统状态
        releasedVM = self.update_record()
        remain_capacity += releasedVM


        self.task_counter += 1
        if self.task_counter == TOTAL_TASK_NUM:
            self.done = True
            self.state = [remain_capacity, 0, 0, 0]
        else:
            task_info = self.task_generator()
            cloud_price = self.cloud_price_generator()

            self.state = [remain_capacity, task_info[0], task_info[1], cloud_price]

        next_state = copy.deepcopy(self.state)
        return next_state, -cost, self.done, info

    def random_step(self):
        info = None
        # 更新历史记录,释放租约到期的VM

        # action是把当前请求加载到云服务器的比例。 0代表申请云服务器。
        # 获取当前服务器状态
        state = copy.deepcopy(self.state)
        remain_capacity = state[0]
        task_size = state[2]
        task_length = state[3]
        cloud_price = state[4]

        cost = 0
        action = random.random()
        # 判断当前动作合法性, 分配到边缘服务器上的VM大小是否大于当前边缘服务器的剩余资源大小
        cloud_vm = int(np.around(task_size * action, 1))
        edge_vm = task_size - cloud_vm

        if edge_vm > remain_capacity:
            # 边缘服务器承载能力不足时 将边缘服务器装满后，剩余卸载到云服务器
            cloud_vm = task_size - remain_capacity
            edge_vm = remain_capacity
            info = cloud_vm / task_size

        # 计算成本(reward)
        remain_capacity = remain_capacity - edge_vm
        self.C_TRACE.append(remain_capacity)
        edge_workload = 1 - remain_capacity / self.edge_capacity
        edge_cost_coefficient = EDGE_COEFFICIENT * (1 + edge_workload) * EDGE_BASIC_COST  # 计算出的VM单位价格
        edge_cost = edge_cost_coefficient * edge_vm  # 时间槽内边缘最终成本
        cloud_cost = cloud_price * cloud_vm * task_length  # 云服务器最终成本

        cost = edge_cost + cloud_cost  # 最终成本

        # 记录使用数据
        self.usage_record.append([edge_vm, task_length])

        # 更新系统状态
        releasedVM = self.update_record()
        remain_capacity += releasedVM

        self.task_counter += 1
        if self.task_counter == TOTAL_TASK_NUM:
            self.done = True
            self.state = [remain_capacity, releasedVM, 0, 0, 0, self.task_counter]
        else:
            task_info = self.task_generator()
            cloud_price = self.cloud_price_generator()

            self.state = [remain_capacity, releasedVM, task_info[0], task_info[1], cloud_price, self.task_counter]

        next_state = copy.deepcopy(self.state)

        return next_state, -cost, self.done, info


if __name__ == '__main__':
    env = Env(seed=1)
    done = False
    ep_r = 0
    ep_steps = 0
    ep_num = 0
    env.reset()



    for t in range(10000):
        ep_steps += 1
        action = env.action_sample()
        next_state, reward, done, info = env.step(action)
        ep_r += reward
        if done:
            print(f"Total timesteps: {t+1}, Episode Num: {ep_num+1} Episode Steps: {ep_steps} Reward: {ep_r}")
            state, done = env.reset(), False
            ep_r = 0
            ep_steps = 0
            ep_num += 1






