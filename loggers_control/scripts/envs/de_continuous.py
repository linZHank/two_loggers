#!/usr/bin/env python
"""
Double escape environment with ccontinuous action space
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import math
import numpy as np
from numpy import pi
from numpy import random
import time

import rospy
import tf
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, SetLinkState, GetModelState, GetLinkState
from gazebo_msgs.msg import ModelState, LinkState, ModelStates, LinkStates
from geometry_msgs.msg import Pose, Twist

from .de import DoubleEscape


class DoubleEscapeContinuous(DoubleEscape):

    def __init__(self):
        super(DoubleEscapeContinuous, self).__init__()
        self.env_type = 'continuous'
        self.name = 'double_escape_continuous'
        self.action_space_shape = (2,2)

    def step(self, action):
        """
        obs, rew, done, info = env.step(action_indices)
        """
        assert action.shape==self.action_space_shape
        rospy.logdebug("\nStart environment step")
        self._take_action(action)
        obs = self._get_observation()
        self.y = np.array([obs[0,1], obs[1,1]])
        # update status
        reward, done = self._compute_reward()
        self.prev_y = self.y.copy()
        info = self.status
        self.step_counter += 1
        rospy.logdebug("End environment step\n")

        return obs, reward, done, info


if __name__ == "__main__":
    env = DoubleEscapeContinuous()
    num_steps = env.max_episode_steps
    obs = env.reset()
    ep, st = 0, 0
    o = env.reset()
    for t in range(num_steps):
        a = np.random.randint(0,4,2)
        o, r, d, i = env.step(a)
        st += 1
        rospy.loginfo("\n-\nepisode: {}, step: {} \nobs: {}, act: {}, reward: {}, done: {}, info: {}".format(ep+1, st, o, a, r, d, i))
        if d:
            ep += 1
            st = 0
            obs = env.reset()
