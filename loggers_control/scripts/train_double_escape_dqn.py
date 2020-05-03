#! /usr/bin/env python
"""
An implementation of Deep Q-network (DQN) for solo_escape_task
DQN is a Model free, off policy, reinforcement learning algorithm (https://deepmind.com/research/dqn/)
Author: LinZHanK (linzhank@gmail.com)
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
    agent0 = DQNAgent(env=env, name='dqn_logger0', layer_sizes=[128,128], warmup_episodes=10)
    agent1 = DQNAgent(env=env, name='dqn_logger1', layer_sizes=[128,128], warmup_episodes=10)
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    num_episodes = 100
    num_steps = 100 #env.max_steps
    num_samples0, num_samples1 = 1, 1 # sample k times to train q-net
    episodic_returns, sedimentary_returns = [], []
    step_counter = 0
    success_counter = 0
    start_time = time.time()
    for ep in range(num_episodes):
        done = False
        rewards = []
        # reset env and get state from it
        obs = env.reset()
        if 'blown' in env.status:
            obs = env.reset()
            continue
        agent0.linear_epsilon_decay(episode=ep, decay_period=12)
        agent0.linear_epsilon_decay(episode=ep, decay_period=12)
        for st in range(num_steps):
            # take actions, no action will take if deactivated
            act0 = agent0.epsilon_greedy(obs)
            act1 = agent1.epsilon_greedy(obs)
            act = [act0, act1]
            # step env
            next_obs, rew, done, info = env.step(act)
            # store transitions and train
            agent0.replay_memory.store([obs.copy(), act0, rew, done, next_obs])
            agent1.replay_memory.store([obs.copy(), act1, rew, done, next_obs])
            # train agent0
            if ep >= agent0.warmup_episodes:
                for _ in range(num_samples0):
                    agent0.train()
            # train agent1
            if ep >= agent1.warmup_episodes:
                for _ in range(num_samples0):
                    agent0.train()
            # log step
            step_counter += 1
            rewards.append(rew)
            if done and info.count('escaped')==2:
                success_counter += 1
            rospy.logdebug("\n-\nepisode: {}, step: {}, epsilon0: {}, epsilon1: {} \nstate: {} \naction: {} \nnext_state: {} \nreward: {} \ndone: {} \ninfo: {} \nsucceed: {}\n-\n".format(ep+1, st+1, agent0.epsilon, agent1.epsilon, obs, act, next_obs, rew, done, info, success_counter))
            obs = next_obs.copy()
            if done:
                # summarize episode
                episodic_returns.append(sum(rewards))
                sedimentary_returns.append(sum(episodic_returns)/(ep+1))
                rospy.loginfo("\n================================================================\nEpisode: {} \nSteps: {} \nEpsilon0: {} \nEpsilon1: {} \nEpisodicReturn: {} \nAveragedReturn: {} \nEndState: {} \nTotalSuccess: {} \nTimeElapsed: {} \n================================================================n".format(ep+1, st+1, agent0.epsilon, agent1.epsilon, episodic_returns[-1], sedimentary_returns[-1], info, success_counter, time.time()-start_time))
                break

    # plot averaged returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()
    # save model
    agent0.save_model()
    agent0.save_params()
    agent1.save_model()
    agent1.save_params()
    # save returns
    np.save(os.path.join(agent0.model_dir, 'ep_returns.npy'), episodic_returns)
    np.save(os.path.join(agent1.model_dir, 'ep_returns.npy'), episodic_returns)
