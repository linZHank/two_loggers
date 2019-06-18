#! /usr/bin/env python

"""
Evaluation of VPG for single logger robot's solo escape task
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

from envs.solo_escape_task_env import SoloEscapeEnv
from utils import data_utils, solo_utils, tf_utils
from utils.data_utils import bcolors
from agents.vpg import VPGAgent

if __name__ == "__main__":
    # Main really starts here
    rospy.init_node("double_escape_dqn_test", anonymous=True, log_level=rospy.INFO)
    # load agent parameters
    params_dir = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/vpg/2019-06-07-04-13/"
    params_path = os.path.join(os.path.dirname(params_dir),"agent_parameters.pkl")
    with open(params_path, "rb") as f:
        agent_params = pickle.load(f)
    # instantiate an vpg agent and load model
    agent = VPGAgent(agent_params)
    agent.load_model("/home/linzhank/ros_ws/src/two_loggers/loggers_control/saved_models/solo_escape/vpg/2019-06-07-04-13/agent/model.h5")
    # make an instance from env class
    env = SoloEscapeEnv()
    env.reset()
    # eval params
    num_episodes = 10
    num_steps = 256
    # start evaluation
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        state = solo_utils.obs_to_state(obs)
        for st in range(num_steps):
            # pick an action
            act_id = agent.greedy_action(state)
            action = agent.actions[act_id]
            # take the action
            obs, _, done, info = env.step(action)
            state = solo_utils.obs_to_state(obs)
            # logging
            rospy.loginfo("Episode: {}, Step: {} \naction: {}, state: {}, done: {}".format(
                ep+1,
                st+1,
                action,
                state,
                done
            ))
            if done:
                break
