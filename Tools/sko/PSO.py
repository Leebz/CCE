#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np
from Tools.sko.tools import func_transformer
from .base import SkoBase
import matplotlib.pyplot as plt
from Enviroments import CCE_ENV_MODIFIED as E
from Configs.config import Config
import copy

class PSO(SkoBase):
    """

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations

    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration

    """

    def __init__(self, dim=100, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5):
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.dim = dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter

        self.env = self.init_env()

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}




    def update_V(self):
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X = self.X + self.V

        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.get_res()
        self.Y = np.array(self.Y)
        self.Y = self.Y.reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None):

        plt.ion()
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            print(f"Iter_num{iter_num}")
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)

            plt.plot(self.gbest_y_hist, "r")
            plt.pause(0.1)
        plt.ioff()
        return self

    def init_env(self):
        config = Config()
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

        return E.Env(**envargs)

    def get_res(self):
        reward_trace = []
        env = copy.copy(self.env)
        action = self.X
        for i in range(self.pop):
            _, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            while True:
                # 执行动作
                next_state, reward, done, info = env.step(action[i][episode_timesteps])
                episode_timesteps += 1
                episode_reward += reward
                if done:
                    # print(f"Episode Steps: {episode_timesteps} Reward: {episode_reward}")
                    reward_trace.append(-episode_reward)
                    break
        return reward_trace

    fit = run
