#! /usr/bin/env python
from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
import tensorflow as tf
import rospy

from envs.solo_escape_task_env import SoloEscapeEnv
from utils import gen_utils, solo_utils, tf_utils
from utils.gen_utils import bcolors
from agents.dqn import DQNAgent

if __name__ == "__main__":
    # Main really starts here
    # start_time = time.time()
    rospy.init_node("solo_escape_dqn", anonymous=True, log_level=rospy.INFO)
    # make an instance from env class
    env = SoloEscapeEnv()
    env.reset()
    # hyper-parameters
    hyp_params = {}
    hyp_params["dim_state"] = len(solo_utils.obs_to_state(env.observation))
    hyp_params["actions"] = np.array([np.array([.5, -1]), np.array([.5, 1])])
    hyp_params["num_episodes"] = 400
    hyp_params["num_steps"] = 200
    hyp_params["batch_size"] = 100
    hyp_params["epsilon"] = 0.2
    hyp_params["gamma"] = 0.99
    # instantiate agent
    agent = DQNAgent(hyp_params)
    agent.train(env)

#     def train(self, env):
#         pass
