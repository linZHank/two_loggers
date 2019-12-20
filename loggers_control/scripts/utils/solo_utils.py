#! /usr/bin/env python
"""
Tools for solo escape tasks
"""
import os
import sys
import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
import csv
import pickle
import matplotlib.pyplot as plt

import rospy
import tf
from geometry_msgs.msg import Pose, Twist

# make arg parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datetime', type=str, default='')
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--time_bonus', type=float, default=0)
    parser.add_argument('--wall_bonus', type=float, default=0)
    parser.add_argument('--door_bonus', type=float, default=0)
    parser.add_argument('--success_bonus', type=float, default=0)
    return parser.parse_args()

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

def reward_to_go(ep_rewards):
    """
    Don't let past reward affect your current action
    """
    n = len(ep_rewards)
    rtgs = np.zeros_like(ep_rewards)
    for i in reversed(range(n)):
        rtgs[i] = ep_rewards[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def create_train_params(date_time, complete_episodes, source, normalize, num_episodes, num_steps, time_bonus, wall_bonus, door_bonus, success_bonus):
    """
    Create training parameters dict based on args
    """
    train_params = {}
    train_params["date_time"] = date_time
    train_params['source'] = source
    train_params['normalize'] = normalize
    train_params['complete_episodes'] = complete_episodes
    train_params["num_episodes"] = num_episodes
    train_params["num_steps"] = num_steps
    train_params["time_bonus"] = time_bonus
    train_params["wall_bonus"] = wall_bonus
    train_params["door_bonus"] = door_bonus
    train_params["success_bonus"] = success_bonus

    return train_params
