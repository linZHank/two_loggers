#! /usr/bin/env python
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
    hyp_params["num_episodes"] = 500
    hyp_params["num_steps"] = 500
    hyp_params["batch_size"] = 512
    hyp_params["epsilon"] = 1
    hyp_params["gamma"] = 0.99
    hyp_params["update_step"] = 1000
    hyp_params["wall_bonus"] = True
    hyp_params["door_bonus"] = True
    hyp_params["dist_bonus"] = False
    hyp_params["model_path"] = os.path.dirname(sys.path[0])+"/dqn_model/"+datetime.now().strftime("%Y-%m-%d-%H-%M")+"/model.ckpt"
    # instantiate agent
    agent = DQNAgent(hyp_params)
    agent.train(env)
