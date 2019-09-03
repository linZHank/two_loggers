#! /usr/bin/env python
"""
Tools for double escape tasks
"""
import numpy as np
import rospy
import tf
import math
import random
import argparse
from geometry_msgs.msg import Pose, Twist

# make arg parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datetime', type=str, default='')
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--num_steps', type=int, default=400)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=0.0004)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--sample_size', type=int, default=512)
    parser.add_argument('--layer_sizes', nargs='+', type=int, help='use space to separate layer sizes, e.g. --layer_sizes 4 16 = [4,16]', default=256 256)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--memory_cap', type=int, default=400000)
    parser.add_argument('--update_step', type=int, default=10000)
    parser.add_argument('--time_bonus', type=float, default=0)
    parser.add_argument('--wall_bonus', type=float, default=0)
    parser.add_argument('--door_bonus', type=float, default=0)
    parser.add_argument('--success_bonus', type=float, default=0)
    return parser.parse_args()

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

def create_pose_buffer(num_poses=1024):
    """
    generate a random rod pose in the room
    with center at (0, 0), width 10 meters and depth 10 meters.
    The robot has 0.2 meters as radius
    Args:
        num_poses: int number of poses going to be created
    Returns:
        pose_vectors: list of pose vectors, [[x1,y1,th1],[x2,y2,th2],...,[xn,yn,thn]]
    """
    def angleRange(x, y, room, L):
        """
        Compute rod angle based on a given robot position
        """
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

    pose_vectors = []
    mag = 4.78
    len_rod = 2
    room = [-mag, mag, -mag, mag] # create a room with boundary
    for i in range(num_poses):
        # randomize robot position
        rx = random.uniform(-mag, mag)
        ry = random.uniform(-mag, mag)
        # randomize rod pose
        min_angle, max_angle = angleRange(rx, ry, room, len_rod)
        angle = random.uniform(min_angle, max_angle)
        x = rx + 0.5*len_rod*math.cos(angle)
        y = ry + 0.5*len_rod*math.sin(angle)
        # randomize robots orientation
        th_0 = random.uniform(-math.pi, math.pi)
        th_1 = random.uniform(-math.pi, math.pi)
        pose_vectors.append([x, y, angle, th_0, th_1])

    return pose_vectors

def create_agent_params(dim_state, actions, layer_sizes, gamma, learning_rate, batch_size, memory_cap, update_step, epsilon_upper, epsilon_lower):
    """
    Create agent parameters dict based on args
    """
    agent_params = {}
    agent_params["dim_state"] = dim_state
    agent_params["actions"] = actions
    agent_params["layer_sizes"] = layer_sizes
    agent_params["gamma"] = gamma
    agent_params["learning_rate"] = learning_rate
    agent_params["batch_size"] = batch_size
    agent_params["memory_cap"] = memory_cap
    agent_params["update_step"] = update_step
    agent_params['epsilon_upper'] = epsilon_upper
    agent_params['epsilon_lower'] = epsilon_lower

    return agent_params

def create_train_params(date_time, source, normalize, num_episodes, num_steps, time_bonus, wall_bonus, door_bonus, success_bonus):
    """
    Create training parameters dict based on args
    """
    train_params = {}
    train_params["date_time"] = date_time
    train_params['source'] = source
    train_params['normalize'] = normalize
    train_params["num_episodes"] = num_episodes
    train_params["num_steps"] = num_steps
    train_params["time_bonus"] = time_bonus
    train_params["wall_bonus"] = wall_bonus
    train_params["door_bonus"] = door_bonus
    train_params["success_bonus"] = success_bonus

    return train_params

def sum_train_info(train_params, add_ons):
    pass
