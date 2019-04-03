! /usr/bin/env python
"""
Tools for solo escape tasks
"""
import numpy as np
import csv
import pickle
import tensorflow as tf
import os
from datetime import datetime
import pickle
import csv
import matplotlib.pyplot as plt

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
        return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def obs_to_state(observation):
    """
    Convert observation to state
    Args:
        observation = {
            "pose": Pose(),
            "twist": Twist()
        }
    Returns:
        state = np.array([x, y, x_dot, y_dot, cos(theta), sin(theta), theta_dot])
    """
    pass
      # x = model_states.pose[-1].position.x
      # y = model_states.pose[-1].position.y
      # v_x = model_states.twist[-1].linear.x
      # v_y = model_states.twist[-1].linear.y
      # quat = (
      #   model_states.pose[-1].orientation.x,
      #   model_states.pose[-1].orientation.y,
      #   model_states.pose[-1].orientation.z,
      #   model_states.pose[-1].orientation.w
      # )
      # euler = tf.transformations.euler_from_quaternion(quat)
      # cos_yaw = math.cos(euler[2])
      # sin_yaw = math.sin(euler[2])
      # yaw_dot = model_states.twist[-1].angular.z

def judge_robot(observation):
    pass
    # if self.curr_pose[0] > 4.79:
    #     reward = -0.
    #     self.status = "east"
    #     self._episode_done = True
    #     rospy.logwarn("Logger is too close to east wall!")
    # elif self.curr_pose[0] < -4.79:
    #     reward = -0.
    #     self.status = "west"
    #     self._episode_done = True
    #     rospy.logwarn("Logger is too close to west wall!")
    # elif self.curr_pose[1] > 4.79:
    #     reward = -0.
    #     self.status = "north"
    #     self._episode_done = True
    #     rospy.logwarn("Logger is too close to north wall!")
    # elif self.curr_pose[1]<=-4.79 and np.absolute(self.curr_pose[0])>1 :
    #     reward = -0.
    #     self.status = "south"
    #     self._episode_done = True
    #     rospy.logwarn("Logger is too close to south wall!")
    # elif -6<self.curr_pose[1]<-4.79 and np.absolute(self.curr_pose[0])>0.79:
    #     reward = 0.
    #     self.status = "door"
    #     self._episode_done = True
    #     rospy.logwarn("Logger is stuck at the door!")
