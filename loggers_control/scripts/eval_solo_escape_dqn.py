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

from envs.solo_escape_discrete_env import SoloEscapeDiscreteEnv
from agents.dqn import DQNAgent


if __name__ == "__main__":
    env=SoloEscapeDiscreteEnv()
    agent = DQNAgent(env=env, name='dqn_logger_eval')
    model_path = os.path.join(sys.path[0], 'saved_models/dqn/solo_escape_discrete/2020-05-01-22-29/dqn_logger/models/1000000.h5')
    agent.load_model(model_path=model_path)
    agent.epsilon = 0.01
    num_episodes = 10
    num_steps = env.max_steps
    episodic_returns, sedimentary_returns = [], []
    step_counter = 0
    success_counter = 0
    for ep in range(num_episodes):
        start_time = time.time()
        done = False
        rewards = []
        # reset env and get state from it
        obs = env.reset()
        for st in range(num_steps):
            # take actions, no action will take if deactivated
            act = agent.epsilon_greedy(obs)
            # step env
            next_obs, rew, done, info = env.step(act)
            step_counter += 1
            rewards.append(rew)
            if info == "escaped":
                success_counter += 1
            # log step
            rospy.logdebug("\n-\nepisode: {}, step: {} \nstate: {} \naction: {} \nnext_state: {} \nreward: {} \ndone: {} \ninfo: {} \nsucceed: {}\n-\n".format(ep+1, st+1, obs, act, next_obs, rew, done, info, success_counter))
            obs = next_obs.copy()
            if done:
                # summarize episode
                episodic_returns.append(sum(rewards))
                sedimentary_returns.append(sum(episodic_returns)/(ep+1))
                rospy.loginfo("\n================================================================\nEpisode: {} \nSteps: {} \nEpsilon: {} \nEpisodicReturn: {} \nAveragedReturn: {} \nEndState: {} \nTotalSuccess: {} \nTimeElapsed: {} \n================================================================n".format(ep+1, st+1, agent.epsilon, episodic_returns[-1], sedimentary_returns[-1], info, success_counter, time.time()-start_time))
                break

    # plot averaged returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Returns')
    ax.plot(episodic_returns)
    plt.show()
