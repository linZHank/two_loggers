#! /usr/bin/env python
"""
Evaluate trained DQN model on double_escape task
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
from numpy import random
from numpy import pi
import time
from datetime import datetime
import matplotlib.pyplot as plt
import rospy

from envs.double_escape_discrete_env import DoubleEscapeDiscreteEnv
from agents.dqn import DQNAgent


if __name__ == "__main__":
    env=DoubleEscapeDiscreteEnv()
    agent0 = DQNAgent(env=env, name='logger0_eval')
    agent1 = DQNAgent(env=env, name='logger1_eval')
    model_path_0 = os.path.join(sys.path[0], 'saved_models/double_escape_discrete/dqn/2020-06-07-18-26/logger0/models/5839800.h5')
    model_path_1 = os.path.join(sys.path[0], 'saved_models/double_escape_discrete/dqn/2020-06-07-18-26/logger1/models/5839800.h5')
    agent0.load_model(model_path=model_path_0)
    agent1.load_model(model_path=model_path_1)
    agent0.epsilon = 0.
    agent1.epsilon = 0.
    num_episodes = 1000
    num_steps = env.max_steps
    episodic_returns = []
    episode_counter = 0
    step_counter = 0
    success_counter = 0
    episodic_qvals_mae = np.zeros(num_episodes)
    while episode_counter<num_episodes:
        start_time = time.time()
        qvals_diff = []
        # reset env and get state from it
        obs, rewards, done = env.reset(), [], False
        # next 3 lines generate state based on
        state_0 = obs.copy() # + 0.5*random.randn(obs.shape[0])
        state_1 = obs.copy() # + 0.5*random.randn(obs.shape[0])
        state_1[:6] = state_1[-6:]
        if 'blown' in env.status:
            continue
        for st in range(num_steps):
            # take actions, no action will take if deactivated
            act0 = agent0.epsilon_greedy(state_0)
            act1 = agent1.epsilon_greedy(state_1)
            qval0 = np.max(agent0.qnet_active(np.expand_dims(state_0,axis=0)))
            qval1 = np.max(agent1.qnet_active(np.expand_dims(state_1,axis=0)))
            qvals_diff.append(np.absolute(qval0-qval1))
            act = np.array([act0, act1])
            # step env
            next_obs, rew, done, info = env.step(act)
            if 'blown' in info:
                break
            obs = next_obs.copy()
            state_0 = obs.copy() # + 0.5*random.randn(obs.shape[0])
            state_1 = obs.copy() # + 0.5*random.randn(obs.shape[0])
            state_1[:6] = state_1[-6:]
            step_counter += 1
            rewards.append(rew)
            # log step
            if info.count('escaped')==2:
                success_counter += 1
            rospy.logdebug("\n-\nepisode: {}, step: {} \nstate: {} \naction: {} \nnext_state: {} \nreward: {} \ndone: {} \ninfo: {} \nsucceed: {}\n-\n".format(episode_counter+1, st+1, obs, act, next_obs, rew, done, info, success_counter))
            if done:
                episodic_qvals_mae[episode_counter] = sum(qvals_diff)/len(qvals_diff)
                episodic_returns.append(sum(rewards))
                episode_counter += 1
                rospy.loginfo("\n================================================================\nEpisode: {} \nSteps: {} \nEpisodicReturn: {} \nEndState: {} \nTotalSuccess: {} \nTimeConsumed: {} \n================================================================n".format(episode_counter, st+1, episodic_returns[-1], info, success_counter, sum(qvals_diff)/len(qvals_diff)))
                break
    qvals_mae_mean = np.mean(episodic_qvals_mae)
    qvals_mae_std = np.std(episodic_qvals_mae)
    print("MeanQValMAE: {}, StdQValMAE: {}".format(qvals_mae_mean, qvals_mae_std))

    # plot averaged returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Returns')
    ax.plot(episodic_returns)
    plt.show()
