"""
An implementation of Deep Q-network (DQN) for solo_escape_task
DQN is a Model free, off policy, reinforcement learning algorithm (https://deepmind.com/research/dqn/)
Author: LinZHanK (linzhank@gmail.com)
"""
from __future__ import absolute_import, division, print_function

import sys
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import rospy
import pickle

from envs.solo_escape_task_env import SoloEscapeEnv
from utils import data_utils, solo_utils, tf_utils
from utils.data_utils import bcolors
from agents.dqn import DQNAgent
from agents import dqn

if __name__ == "__main__":
    # instantiate an env
    env = SoloEscapeEnv()
    env.reset()
    # load agent parameters
    model_dir = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/dqn/2019-10-09-11-05/"
    params_path = os.path.join(model_dir,'agent/agent_parameters.pkl')
    with open(params_path, 'rb') as f:
        agent_params = pickle.load(f)
    # load agent model
    agent = DQNAgent(agent_params)
    agent.load_model(os.path.join(model_dir, 'agent/model.h5'))
    # evaluation params
    num_episodes = 100
    num_steps = 400
    ep = 0
    # start evaluating
    while ep < num_episodes:
        obs, _ = env.reset()
        done = False
        ep_rewards = []
        state = solo_utils.obs_to_state(obs)
        for st in range(num_steps):
            action_index = np.argmax(agent.qnet_active.predict(state.reshape(1,-1)))
            action = agent_params["actions"][action_index]
            obs, rew, done, info = env.step(action)
            if info['status'] == "north" or info['status'] == "west" or info['status'] == "south" or info['status'] == "east" or info['status'] == "sdoor":
                done = True
            elif  info['status'] == "blew":
                done = True
                ep -= 1
            next_state = solo_utils.obs_to_state(obs)
            state = next_state
            ep_rewards.append(rew)
            # logging
            rospy.logwarn(
                "Episode: {}, Step: {}: \nstate: {}, action: {}, next state: {} \nreward/episodic_return: {}/{}, status: {}, succeeded: {}".format(
                    ep+1,
                    st+1,
                    state,
                    action,
                    next_state,
                    rew,
                    sum(ep_rewards),
                    info["status"],
                    env.success_count
                )
            )
            if done:
                ep += 1
                break

    print("Loggers succeeded escaping {} out of {}".format(env.success_count, num_episodes))
    env.reset()
