#! /usr/bin/env python
from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
import tensorflow as tf

from envs.solo_escape_task_env import SoloEscapeEnv
from utils import gen_utils, solo_utils, tf_utils
from utils.gen_utils import bcolors

if __name__ == "__main__":
    # Main really starts here
    start_time = time.time()
    rospy.init_node("solo_escape_dqn", anonymous=True, log_level=rospy.INFO)
    # make an instance from env class
    env = SoloEscapeEnv()
    env.reset()
    dim_state = len(solo_utils.obs_to_state(env.observation))
    actions = np.array([np.array([.5, -1]), np.array([.5, 1])])
    num_actions = len(actions)
    agent = DQN()
    agent.train(env)

#     def train(self, env):
#         pass
