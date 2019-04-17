#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(0, "/home/piofagivens/ros_ws/src/two_loggers/loggers_control/scripts/envs")
sys.path.insert(0, "/home/piofagivens/ros_ws/src/two_loggers/loggers_control/scripts/utils")
import termios
import tty
import os
import numpy as np
import math
import time
import random
import rospy
from std_srvs.srv import Empty

from solo_escape_task_env import SoloEscapeEnv

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

if __name__ == "__main__":
    rospy.init_node("solo_escape_env_test" , anonymous=True, log_level=rospy.INFO)
    num_episodes = 20
    env = SoloEscapeEnv()
    env.reset()
    env.reset()
    done = False
    ep = 1
    st = 1
    while True:
        char = getch()
        if char == "q":
            action = np.array([.8, 1])
            _, _, done, _ = env.step(action)
        elif char == "e":
            action = np.array([.8, -1])
            _, _, done, _ = env.step(action)
        elif char == "s":
            exit(0)
        st += 1
        rospy.loginfo("Episode: {}, Step: {}, Action: {}".format(ep, st,action ))
        if done:
            _, _ = env.reset()
            ep += 1
            st = 1
            if ep > num_episodes:
                break
