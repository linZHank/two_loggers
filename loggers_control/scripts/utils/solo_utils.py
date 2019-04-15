#! /usr/bin/env python
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

import rospy
import tf
from geometry_msgs.msg import Pose, Twist


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
    x = observation["pose"].position.x
    y = observation["pose"].position.y
    v_x = observation["twist"].linear.x
    v_y = observation["twist"].linear.y
    quat = (
        observation["pose"].orientation.x,
        observation["pose"].orientation.y,
        observation["pose"].orientation.z,
        observation["pose"].orientation.w
    )
    euler = tf.transformations.euler_from_quaternion(quat)
    cos_yaw = np.cos(euler[2])
    sin_yaw = np.sin(euler[2])
    yaw_dot = observation["twist"].angular.z
    state = np.array([x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot])

    return state

def adjust_reward(rew, info, delta_d,
                  wall_bonus_flag, door_bonus_flag, dist_bonus_flag):
    adj_reward = rew
    if info["status"] == "escaped":
        adj_reward = 10*rew
        done = True
    elif info["status"] == "sdoor":
        if door_bonus_flag:
            adj_reward = 1./100
        done = True
    elif info["status"] == "tdoor":
        if door_bonus_flag:
            adj_reward = 1./100
        done = False
    elif info["status"] == "trapped":
        if dist_bonus_flag:
            if delta_d < 0:
                adj_reward = -1./1e3 # negative, if getting further from exit
            else:
                adj_reward = 1./1e4 # positive, if getting closer to exit
    else:
        if wall_bonus_flag:
            adj_reward = -1./1e2
        done = True

    return adj_reward, done
