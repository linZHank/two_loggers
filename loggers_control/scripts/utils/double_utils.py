#! /usr/bin/env python
"""
Tools for double escape tasks
"""
import numpy as np
import rospy
import tf
from geometry_msgs.msg import Pose, Twist

def obs_to_state(observation, mode):
    """
    Convert observation to state
    Args:
        observation = {
            "log": {Pose(), Twist()}
            "logger_0": {Pose(), Twist()}
            "logger_1": {Pose(), Twist()}
        }
        mode: "logger_0", "logger_1" or "all"
    Returns:
        state = [state_log, state_robot]
            state_log = [x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot]
            state_robot = [x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot]
    """
    # compute log state
    x = observation["log"]["pose"].position.x
    y = observation["log"]["pose"].position.y
    v_x = observation["log"]["twist"].linear.x
    v_y = observation["log"]["twist"].linear.y
    quat = (
        observation["log"]["pose"].orientation.x,
        observation["log"]["pose"].orientation.y,
        observation["log"]["pose"].orientation.z,
        observation["log"]["pose"].orientation.w
    )
    euler = tf.transformations.euler_from_quaternion(quat)
    cos_yaw = np.cos(euler[2])
    sin_yaw = np.sin(euler[2])
    yaw_dot = observation["log"]["twist"].angular.z
    state_log = np.array([x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot])
    # compute logger_0 state
    x = observation["logger_0"]["pose"].position.x
    y = observation["logger_0"]["pose"].position.y
    v_x = observation["logger_0"]["twist"].linear.x
    v_y = observation["logger_0"]["twist"].linear.y
    quat = (
        observation["logger_0"]["pose"].orientation.x,
        observation["logger_0"]["pose"].orientation.y,
        observation["logger_0"]["pose"].orientation.z,
        observation["logger_0"]["pose"].orientation.w
    )
    euler = tf.transformations.euler_from_quaternion(quat)
    cos_yaw = np.cos(euler[2])
    sin_yaw = np.sin(euler[2])
    yaw_dot = observation["logger_0"]["twist"].angular.z
    state_logger0 = np.array([x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot])
    # compute logger_1 state
    x = observation["logger_1"]["pose"].position.x
    y = observation["logger_1"]["pose"].position.y
    v_x = observation["logger_1"]["twist"].linear.x
    v_y = observation["logger_1"]["twist"].linear.y
    quat = (
        observation["logger_1"]["pose"].orientation.x,
        observation["logger_1"]["pose"].orientation.y,
        observation["logger_1"]["pose"].orientation.z,
        observation["logger_1"]["pose"].orientation.w
    )
    euler = tf.transformations.euler_from_quaternion(quat)
    cos_yaw = np.cos(euler[2])
    sin_yaw = np.sin(euler[2])
    yaw_dot = observation["logger_1"]["twist"].angular.z
    state_logger1 = np.array([x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot])

    if mode == "logger_0":
        return np.concatenate((state_log, state_logger0), axis=0)
    elif mode == "logger_1":
        return np.concatenate((state_log, state_logger1), axis=0)
    elif mode == "all":
        return np.concatenate((state_log, state_logger0, state_logger1), axis=0)

def adjust_reward(train_params, env):
    done = env._episode_done
    adj_reward = env.reward
    info = env.info
    if info["status"] == "escaped":
        if train_params["success_bonus"]:
            adj_reward = train_params["success_bonus"]
        done = True
    elif info["status"] == "sdoor":
        if train_params["door_bonus"]:
            adj_reward = train_params["door_bonus"]
        done = True
    elif info["status"] == "tdoor":
        if train_params["door_bonus"]:
            adj_reward = train_params["door_bonus"]
    elif info["status"] == "trapped":
        if train_params["time_bonus"]:
            adj_reward = -1./train_params["num_steps"]
    elif info["status"] == "blew":
        done = True
    else: # hit wall
        if train_params["wall_bonus"]:
            adj_reward = train_params["wall_bonus"]
        done = True

    return adj_reward, done
