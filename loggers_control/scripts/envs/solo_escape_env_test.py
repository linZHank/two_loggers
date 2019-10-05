#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import math
import time
import random
import rospy
from std_srvs.srv import Empty

from solo_escape_task_env import SoloEscapeEnv


if __name__ == "__main__":
  rospy.init_node("solo_escape_env_test" , anonymous=True, log_level=rospy.INFO)
  num_episodes = 10
  num_steps = 20

  escaper = SoloEscapeEnv()
  for ep in range(num_episodes):
    obs, info = escaper.reset()
    rospy.loginfo("Logger was reset with observation: {} \nwith information: {}".format(obs, info))
    for st in range(num_steps):
      action = np.random.randn(2)
      obs, rew, done, info = escaper.step(action)
      rospy.loginfo("Episode: {}, Step: {}, action: {}".format(ep, st, action))

  # rospy.spin()
