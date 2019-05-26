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

def adjust_reward(train_params, env):
                  # rew, info, delta_d, done, num_episodes
                  # time_bonus_flag, wall_bonus_flag,
                  # door_bonus_flag, dist_bonus_flag):
    done = env._episode_done
    adj_reward = env.reward
    info = env.info
    if info["status"] == "escaped":
        if train_params["success_bonus"]:
            adj_reward += train_params["success_bonus"]
        done = True
    elif info["status"] == "sdoor":
        if train_params["door_bonus"]:
            adj_reward += train_params["door_bonus"]
        done = True
    elif info["status"] == "tdoor":
        if train_params["door_bonus"]:
            adj_reward += train_params["door_bonus"]
    elif info["status"] == "trapped":
        if train_params["time_bonus"]:
            adj_reward += train_params["time_bonus"]
    else: # hit wall
        if train_params["wall_bonus"]:
            adj_reward += train_params["wall_bonus"]
        done = True
    if env.steps >= train_params['num_steps']:
        done = True

    return adj_reward, done
