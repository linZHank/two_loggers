#! /usr/bin/env python
"""
Tools for double escape tasks
"""
import numpy as np
import rospy
import tf
import math
import random
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
            adj_reward += train_params['time_bonus']
    elif info["status"] == "blew":
        done = True
    else: # hit wall
        if train_params["wall_bonus"]:
            adj_reward += train_params["wall_bonus"]
        done = True
    if env.steps >= train_params['num_steps']:
        done = True

    return adj_reward, done

"""
generate a random rod position (center of the rod) and orientation
in the room with center at (0, 0), width 10 meters and depth 10 meters
the robot has 0.2 meters as radius
"""
def random_rod_position(number):
    def angleRange(x, y, room, L):
        min = 0
        max = 0
        dMinX = abs(x-room[0])
        dMaxX = abs(x-room[1])
        dMinY = abs(y-room[2])
        dMaxY = abs(y-room[3])

        if dMinX < L:
            if dMinY < L:
                min = -0.5*math.pi+math.acos(dMinY/L)
                max = math.pi-math.acos(dMinX/L)
            elif dMaxY < L:
                min = -math.pi+math.acos(dMinX/L)
                max = 0.5*math.pi-math.acos(dMaxY/L)
            else:
                min = -math.pi + math.acos(dMinX/L)
                max = math.pi-math.acos(dMinX/L)
        elif dMaxX < L:
            if dMinY < L:
                min = math.acos(dMaxX/L)
                max = 1.5*math.pi-math.acos(dMinY/L)
            elif dMaxY < L:
                min = 0.5*math.pi+math.acos(dMaxY/L)
                max = 2*math.pi-math.acos(dMaxX/L)
            else:
                min = math.acos(dMaxX/L)
                max = 2*math.pi-math.acos(dMaxX/L)
        else:
            if dMinY < L:
                min = -0.5*math.pi+math.acos(dMinY/L)
                max = 1.5*math.pi-math.acos(dMinY/L)
            elif dMaxY < L:
                min = 0.5*math.pi+math.acos(dMaxY/L)
                max = 2.5*math.pi-math.acos(dMaxY/L)
            else:
                min = -math.pi
                max = math.pi
        return min, max

    rodPostionVec = []
    # create a room with boundary to initialize the
    mag = 4.78
    rodLen = 2
    room = [-mag, mag, -mag, mag]
    for i in range(number):
        rx = random.uniform(-mag, mag)
        ry = random.uniform(-mag, mag)
        minAngle, maxAngle = angleRange(rx, ry, room, rodLen)
        angle = random.uniform(minAngle, maxAngle)
        rodcx = rx + 0.5*rodLen*math.cos(angle)
        rodcy = ry + 0.5*rodLen*math.sin(angle)
        rodPostionVec.append([rodcx, rodcy, angle])



    return rodPostionVec
