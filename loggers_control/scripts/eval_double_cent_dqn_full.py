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
    agent = DQNAgent(env=env, name='double_logger_eval', dim_state=env.observation_space[0],
                     num_actions=env.action_space[0]**2)
    model_path = os.path.join(sys.path[0], 'saved_models/double_escape_discrete/cent_dqn_full/2020-06-25-21-25/cent_dqn_full/models/8033600.h5')
    agent.load_model(model_path=model_path)
    agent.epsilon = 0.
    num_episodes = 1000
    num_steps = env.max_steps
    episodic_returns = []
    episode_counter = 0
    step_counter = 0
    success_counter = 0
    while episode_counter<num_episodes:
        start_time = time.time()
        qvals_diff = []
        # reset env and get state from it
        obs, rewards, done = env.reset(), [], False
        if 'blown' in env.status:
            continue
        for st in range(num_steps):
            # take actions, no action will take if deactivated
            act = agent.epsilon_greedy(obs)
            act0 = int(act/env.action_space[0])
            act1 = act%env.action_space[0]
            action = np.array([act0, act1])
            action = np.array([act0, act1])
            # step env
            next_obs, rew, done, info = env.step(action)
            if 'blown' in info:
                break
            obs = next_obs.copy()
            step_counter += 1
            rewards.append(rew)
            # log step
            if info.count('escaped')==2:
                success_counter += 1
            rospy.logdebug("\n-\nepisode: {}, step: {} \nstate: {} \naction: {} \nnext_state: {} \nreward: {} \ndone: {} \ninfo: {} \nsucceed: {}\n-\n".format(episode_counter+1, st+1, obs, act, next_obs, rew, done, info, success_counter))
            if done:
                # summarize episode
                episodic_returns.append(sum(rewards))
                episode_counter += 1
                rospy.loginfo("\n================================================================\nEpisode: {} \nSteps: {} \nEpisodicReturn: {} \nEndState: {} \nTotalSuccess: {} \n================================================================n".format(
                    episode_counter,
                    st+1,
                    episodic_returns[-1], 
                    info, success_counter, 
                ))
                break

    # plot averaged returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Returns')
    ax.plot(episodic_returns)
    plt.show()
    
