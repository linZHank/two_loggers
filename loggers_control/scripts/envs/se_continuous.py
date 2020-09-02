#!/usr/bin/env python
"""
Solo escape environment with ccontinuous action space
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
from numpy import pi
from numpy import random

import rospy
import tf
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState, ModelStates 
from geometry_msgs.msg import Pose, Twist
from .se import SoloEscape


class SoloEscapeContinuous(SoloEscape):
    
    def __init__(self):
        super(SoloEscapeContinuous, self).__init__()
        # env properties
        self.env_type = 'continuous'
        self.name = 'solo_escape_continuous'
        self.action_space_shape = (2,)
        # robot properties
    def step(self, action):
        """
        obs, rew, done, info = env.step(action_index)
        """
        assert action.shape==self.action_space_shape
        rospy.logdebug("\nStart Environment Step")
        self._take_action(action)
        obs = self._get_observation()
        self.y = obs[1]
        # compute reward and done
        reward, done = self._compute_reward()
        self.prev_y = self.y.copy()
        info = self.status
        self.step_counter += 1 # make sure inc step counter before compute reward
        rospy.logdebug("End Environment Step\n")

        return obs, reward, done, info

if __name__ == "__main__":
    env = SoloEscapeContinuous()
    num_steps = env.max_episode_steps
    obs = env.reset()
    ep, st = 0, 0
    for t in range(env.max_episode_steps):
        a = t%2
        o, r, d, i = env.step(a)
        st += 1
        rospy.loginfo("\n-\nepisode: {}, step: {} \nobs: {}, act: {}, reward: {}, done: {}, info: {}".format(ep+1, st, o, a, r, d, i))
        if d:
            ep += 1
            st = 0
            obs = env.reset()
            
