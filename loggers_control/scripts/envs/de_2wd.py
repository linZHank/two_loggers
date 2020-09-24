#!/usr/bin/env python
"""
Double escape environment with discrete action space.
Name with 2wd may be disguise, but really AWD. 
Robot 0 has 4 options of action, which is the same as in 'de';
Robot 1 only has 2 options that are linear cmd_vel's.
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


class DoubleEscape2WD(DoubleEscape):

    def __init__(self):
        super(DoubleEscape2WD, self).__init__()
        self.name = 'double_escape_2wd'
        self.action_space_shape = (2,)
        self.action_reservoir_0 = np.array([[1.5,pi/3], [1.5,-pi/3], [-1.5,pi/3], [-1.5,-pi/3]])
        self.action_reservoir_1 = np.array([[.75,0], [-.75,0]])

    def step(self, action_indices):
        """
        obs, rew, done, info = env.step(action_indices)
        """
        assert 0<=action_indices[0]<self.action_reservoir_0.shape[0]
        assert 0<=action_indices[1]<self.action_reservoir_1.shape[0]
        rospy.logdebug("\nStart environment step")
        actions = np.zeros((2,2))
        actions[0] = self.action_reservoir_0[action_indices[0]]
        actions[1] = self.action_reservoir_1[action_indices[1]]
        self._take_action(actions)
        self._get_observation()
        # update status
        reward, done = self._compute_reward()
        self.prev_obs = self.obs.copy() # make sure this happened after reward computing
        info = self.status
        self.step_counter += 1
        if self.step_counter>=self.max_episode_steps:
            rospy.logwarn("Step: {}, \nMax step reached...".format(self.step_counter))
        rospy.logdebug("End environment step\n")

        return self.obs, reward, done, info


if __name__ == "__main__":
    env = DoubleEscape2WD()
    num_steps = env.max_episode_steps
    obs = env.reset()
    ep, st = 0, 0
    o = env.reset()
    for t in range(num_steps):
        a = np.zeros(2)
        a[0] = np.random.randint(0,4)
        a[1] = np.random.randint(0,2)
        o, r, d, i = env.step(a)
        st += 1
        rospy.loginfo("\n-\nepisode: {}, step: {} \nobs: {}, act: {}, reward: {}, done: {}, info: {}".format(ep+1, st, o, a, r, d, i))
        if d:
            ep += 1
            st = 0
            obs = env.reset()
