"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001                   # learning rate
DECAY = 0.001
EPSILON = 0.0 if DECAY is not None else 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 200   # target update frequency
MEMORY_CAPACITY = 5000
env = gym.make('DuplicatedInput-v0')
# env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = 20
N_STATES = 2
MAX_STEPS = 10000000
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


print(env.action_space.sample())
print(env.observation_space)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()


def index_to_action(index):
    sub_action1 = index // 10
    sub_action2 = (index // 5) % 2
    sub_action3 = index % 5

    return [sub_action1, sub_action2, sub_action3]

def plotLearning(scores, filename, x=None, window=50, fcfs_scores=None, random_scores=None):
    N = len(scores)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])

    if x is None:
        x = [i for i in range(N)]

    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)

ep_r_trace = []
ep_counter = 0

position = 0
print("Collecting Experience...")
for i in range(MAX_STEPS):
    # Add index to observation
    s = [env.reset(), position]
    env.render()
    ep_r = 0
    while True:
        a = dqn.choose_action(s)

        action_tuple = index_to_action(a)

        # Change the position according to action
        if action_tuple[0] == 0:
            position -= 1
        else:
            position += 1

        s_, r, done, info = env.step(action_tuple)

        # Convert state to vector
        s_ = [s_, position]


        # Modify the reward
        # r = r - 0.5

        dqn.store_transition(s, a, r, s_)

        s = s_

        ep_r += r

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if EPSILON < 0.9:
                EPSILON = EPSILON + DECAY

        if done:
            env.render()
            ep_counter += 1
            print("---------------------------------episode:", ep_counter, "ep_r:", ep_r, "epsilon", EPSILON, "-------------------------------------------------------")
            ep_r_trace.append(ep_r)
            ep_r = 0
            position = 0
            s = [env.reset(), position]
            done = False
            if ep_counter % 1000 == 0:
                plotLearning(ep_r_trace, filename="DuplicatedInput-v0/res-"+str(ep_counter)+".jpg", window=50)

