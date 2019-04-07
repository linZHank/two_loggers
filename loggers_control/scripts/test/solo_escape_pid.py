from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(0, "~/ros_ws/src/two_loggers/loggers_control/scripts/envs")
sys.path.insert(0, "~/ros_ws/src/two_loggers/loggers_control/scripts/utils")
import numpy as np
import rospy
import random
import os
import time
from solo_escape_task_env import SoloEscapeEnv
import gen_utils, solo_utils, tf_utils

if __name__ == "__main__":
    rospy.init_node("solo_escape_pid", anonymous=True, log_level=rospy.INFO)
    env = SoloEscapeEnv()
    env.reset()
    num_episodes = 20
    # pid params
    kp_lin = 5
    kd_lin = 0.5
    kp_ang = 10
    kd_ang = 0.1
    #
    p_exit = np.array([0,-6.04])
    vec_x = np.array([1, 0])
    vec_y = np.array([0, 1])
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        state = solo_utils.obs_to_state(obs)
        while True:
            env.step()
