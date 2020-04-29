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
from datetime import datetime
import matplotlib.pyplot as plt
import rospy



from envs.solo_escape_discrete_env import SoloEscapeDiscreteEnv
from agents.dqn import DQNAgent

import pdb

if __name__ == "__main__":
    env=SoloEscapeDiscreteEnv()
    agent = DQNAgent(env=env, name='dqn_logger')
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_dir = sys.path[0]+"/saved_models/solo_escape/dqn/"+date_time
    num_episodes = 10000
    num_steps = env.max_steps
    num_samples = 1 # sample k times to train q-net
    episodic_returns, sedimentary_returns = [], []
    step_counter = 0
    success_counter = 0
    for ep in range(num_episodes):
        done = False
        rewards = []
        # reset env and get state from it
        obs = env.reset()
        agent.linear_epsilon_decay(episode=ep, decay_period=int(num_episodes/10))
        for st in range(num_steps):
            # take actions, no action will take if deactivated
            act = agent.epsilon_greedy(obs)
            # step env
            next_obs, rew, done, info = env.step(act)
            # store transitions and train
            agent.replay_memory.store([obs.copy(), act, rew, done, next_obs])
            if ep >= agent.warmup_episodes:
                for _ in range(num_samples):
                    agent.train()
                step_counter += 1
            rewards.append(rew)
            if info == "escaped":
                success_counter += 1
            # log step
            rospy.logdebug("\n-\nepisode: {}, step: {}, epsilon: {} \nstate: {} \naction: {} \nnext_state: {} \nreward: {} \ndone: {} \ninfo: {} \nsucceed: {}\n-\n".format(ep+1, st+1, agent.epsilon, obs, act, next_obs, rew, done, info,success_counter))
            obs = next_obs.copy()
            if done:
                # summarize episode
                episodic_returns.append(sum(rewards))
                sedimentary_returns.append(sum(episodic_returns)/(ep+1))
                rospy.loginfo("\n================================================================\nepisode: {} \nsteps: {} \nepisodic return: {} \naveraged return: {} \nsucceed: {}\n================================================================n".format(ep+1, st+1, episodic_returns[-1], sedimentary_returns[-1], success_counter))
                break

    # plot averaged returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()
    # save model
    agent.save_model()
    agent.save_params()
    # save returns
    np.save(os.path.join(agent.model_dir, 'ep_returns.npy'), episodic_returns)
