#! /usr/bin/env python
"""
Tools for double escape tasks
"""
import numpy as np
import rospy
import tf
from geometry_msgs.msg import Pose, Twist

def obs_to_state(observation, robot_name):
    """
    Convert observation to state
    Args:
        observation = {
            "log": {Pose(), Twist()}
            "logger_0": {Pose(), Twist()}
            "logger_1": {Pose(), Twist()}
        }
        robot_name: "logger_0" or "logger_1"
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
    # compute robot state
    x = observation[robot_name]["pose"].position.x
    y = observation[robot_name]["pose"].position.y
    v_x = observation[robot_name]["twist"].linear.x
    v_y = observation[robot_name]["twist"].linear.y
    quat = (
        observation[robot_name]["pose"].orientation.x,
        observation[robot_name]["pose"].orientation.y,
        observation[robot_name]["pose"].orientation.z,
        observation[robot_name]["pose"].orientation.w
    )
    euler = tf.transformations.euler_from_quaternion(quat)
    cos_yaw = np.cos(euler[2])
    sin_yaw = np.sin(euler[2])
    yaw_dot = observation[robot_name]["twist"].angular.z
    state_robot = np.array([x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot])

    return np.concatenate((state_log, state_robot), axis=0)
