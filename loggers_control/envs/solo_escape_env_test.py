#!/usr/bin/env python
from __future__ import print_function

import gym
import numpy as np
import time
import random
from gym import wrappers
import rospy
import rospkg
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, LinkStates

# import our training environment
import solo_escape_task_env # need write task env

rospy.init_node('env_test', anonymous=True, log_level=rospy.DEBUG)    
env = gym.make('SoloEscape-v0')

# test env with random sampled actions
for episode in range(16):
  state, info = env.reset()
  env.step(np.array([0,0]))
  done = False
  for step in range(64):
    action = np.array([1, 0]) # env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print("Episode : {}, Step: {}, \nCurrent position: {}, Reward: {:.4f}".format(
      episode,
      step,
      info["current_position"],
      reward
    ))
    if done:
      break