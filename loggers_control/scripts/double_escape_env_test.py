#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import math
import time
import random

import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Twist

from envs.double_escape_task_env import DoubleEscapeEnv
from utils import double_utils


if __name__ == "__main__":
    num_episodes = 100
    num_steps = 100
    rospy.init_node("double_escape_env_test" , anonymous=True, log_level=rospy.INFO)

    escaper = DoubleEscapeEnv()
    actions = np.array([[1,-1],[1,1]])
    escaper.reset() # the first reset always set the model at (0,0)
    for ep in range(num_episodes):
        obs, info = escaper.reset()
        # state_logger0 = double_utils.obs_to_state(obs, "logger_0")
        rospy.loginfo("Loggers were reset with observation: {} \nwith information: {}".format(obs, info))
        for st in range(num_steps):
            action_0 = random.choice(actions)
            action_1 = random.choice(actions)
            obs, rew, done, info = escaper.step(action_0, action_1)
            state_logger0 = double_utils.obs_to_state(obs, "logger_0")
            state_logger1 = double_utils.obs_to_state(obs, "logger_1")
            state_all = double_utils.obs_to_state(obs, "all")
            if info["status"] == "escaped":
                done = True
            elif info["status"] == "tdoor":
                done = False
            elif info["status"] == "sdoor":
                done = True
            elif info["status"] == "blew":
                done = True
            elif info["status"] == "trapped":
                done = False
            else:
                done = True
            rospy.loginfo("Episode: {}, Step: {}, \nlogger_0 state: {}, logger_1 state: {}, state_all: {} \naction_0: {}, action_1: {} \ninfo: {}".format(ep, st, state_logger0, state_logger1, state_all, action_0, action_1, info))
            if done:
                 break
    # stop loggers in the end
    escaper.step(np.zeros(2), np.zeros(2))
