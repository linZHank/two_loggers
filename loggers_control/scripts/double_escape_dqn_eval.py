"""
An implementation of Deep Q-network (DQN) for solo_escape_task
DQN is a Model free, off policy, reinforcement learning algorithm (https://deepmind.com/research/dqn/)
Author: LinZHanK (linzhank@gmail.com)
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import pickle
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import rospy

from envs.double_escape_task_env import DoubleEscapeEnv
from utils import data_utils, double_utils, tf_utils
from utils.data_utils import bcolors
from agents.dqn import DQNAgent

if __name__ == "__main__":
    # load agent models
    model_dir = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/2019-06-07-23-52/"
    params0_path = os.path.join(model_dir,"agent_0/agent0_parameters.pkl")
    with open(params0_path, "rb") as f:
        agent0_params = pickle.load(f)
    params1_path = os.path.join(model_dir,"agent_1/agent1_parameters.pkl")
    with open(params1_path, "rb") as f:
        agent1_params = pickle.load(f)
    # instantiate agents
    agent_0 = DQNAgent(agent0_params)
    agent_0.load_model(os.path.join(model_dir, "agent_0/model.h5"))
    agent_1 = DQNAgent(agent1_params)
    agent_1.load_model(os.path.join(model_dir, "agent_1/model.h5"))
    # instantiate env
    env = DoubleEscapeEnv()
    env.reset()
    # evaluation params
    num_episodes = 100
    num_steps = 400
    # start evaluating
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        state_agt0 = double_utils.obs_to_state(obs, "all")
        state_agt1 = double_utils.obs_to_state(obs, "all")
        for st in range(num_steps):
            agent0_acti = np.argmax(agent_0.qnet_active.predict(state_agt0.reshape(1,-1)))
            agent0_action = agent0_params["actions"][agent0_acti]
            agent1_acti = np.argmax(agent_1.qnet_active.predict(state_agt1.reshape(1,-1)))
            agent1_action = agent1_params["actions"][agent1_acti]
            obs, rew, done, info = env.step(agent0_action, agent1_action)
            if info['status'] == "north" or info['status'] == "west" or info['status'] == "south" or info['status'] == "east" or info['status'] == "sdoor" or info['status'] == "blew":
                done = True
            next_state_agt0 = double_utils.obs_to_state(obs, "all")
            next_state_agt1 = double_utils.obs_to_state(obs, "all")
            state_agt0 = next_state_agt0
            state_agt1 = next_state_agt1
            # logging
            print(
                bcolors.OKGREEN,
                "Episode: {}, Step: {} \naction0: {}->{}, action0: {}->{}, agent_0 state: {}, agent_1 state: {}, reward: {}, status: {} \nsuccess_count: {}".format(
                    ep,
                    st,
                    agent0_acti,
                    agent0_action,
                    agent1_acti,
                    agent1_action,
                    next_state_agt0,
                    next_state_agt1,
                    rew,
                    info["status"],
                    env.success_count
                ),
                bcolors.ENDC
            )
            if done:
                break

    print("Loggers succeeded escaping {} out of {}".format(env.success_count, num_episodes))
