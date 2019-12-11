#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import sys

import numpy as np
from numpy import pi
import rospy
import tf

from solo_escape_task_env import SoloEscapeEnv


if __name__ == "__main__":
    num_episodes = 10
    num_steps = 8

    escaper = SoloEscapeEnv()
    escaper.reset()
    # test 1
    # for ep in range(num_episodes):
    #     obs, info = escaper.reset()
    #     rospy.loginfo("Logger was reset with observation: {} \nwith information: {}".format(obs, info))
    #     for st in range(num_steps):
    #         action = np.random.randn(2)
    #         obs, rew, done, info = escaper.step(action)
    #         rospy.loginfo("Episode: {}, Step: {}, action: {}".format(ep, st, action))

    # test 2
    for ep in range(num_episodes):
        obs, info = escaper.reset()
        action = np.random.randn(2)
        rospy.loginfo("Logger was reset with observation: {} \nwith information: {}".format(obs, info))
        for st in range(num_steps):
            obs, rew, done, info = escaper.step(action)
            rospy.loginfo("Episode: {}, Step: {}, action: {}".format(ep, st, action))

    # test 3
    # x = np.linspace(-4.5, 4.5, num=num_episodes)
    # y = np.linspace(-4.5, 4.5, num=num_episodes)
    # theta = np.linspace(-pi, pi, num=num_episodes)
    #
    # for ep in range(num_episodes):
    #     pose = [x[ep], y[ep], theta[ep]]
    #     obs, info = escaper.reset(init_pose=pose)
    #     action = np.zeros(2)
    #     # rospy.loginfo("Logger was reset with observation: {} \nwith information: {}".format(obs, info))
    #     for st in range(num_steps):
    #         obs, rew, done, info = escaper.step(action)
    #         # rospy.loginfo("Episode: {}, Step: {}, action: {}".format(ep, st, action))
    pass
    rospy.logwarn("test finished")
