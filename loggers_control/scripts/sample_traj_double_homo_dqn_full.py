#! /usr/bin/env python
"""
Sample trajectories with DQN model. Unfortunately, robots orientation cannot be set.
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
from numpy import random
from numpy import pi
import time
from datetime import datetime
import matplotlib.pyplot as plt
import rospy
import tf
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ModelState

from envs.double_escape_discrete_env import DoubleEscapeDiscreteEnv
from agents.dqn import DQNAgent


if __name__ == "__main__":
    env=DoubleEscapeDiscreteEnv()
    agent = DQNAgent(env=env, name='double_logger_sample')
    model_path = os.path.join(sys.path[0], 'saved_models/double_escape_discrete/dqn/2020-05-29-17-33/double_logger/models/5093500.h5')
    agent.load_model(model_path=model_path)
    agent.epsilon = 0.
    num_steps = env.max_steps
    traj = []
    acts = []
    traj_path = os.path.join(sys.path[0], 'saved_trajectories', datetime.now().strftime("%Y-%m-%d-%H-%M"), 'traj.npy')
    # Set double_logger to specified pose
    init_pose = np.array([-2,4,-pi/4]) # modify this to any pose as needed: [x, y, theta]
    env.pausePhysics()
    env.resetWorld()
    double_logger_pose = ModelState()
    double_logger_pose.model_name = "double_logger"
    double_logger_pose.pose.position.z = 0.09
    x = init_pose[0]
    y = init_pose[1]
    theta = init_pose[2]
    quat = tf.transformations.quaternion_from_euler(0, 0, theta)
    double_logger_pose.pose.position.x = x
    double_logger_pose.pose.position.y = y
    double_logger_pose.pose.orientation.z = quat[2]
    double_logger_pose.pose.orientation.w = quat[3]
    env.setModelState(model_state=double_logger_pose)
    env.unpausePhysics()
    for _ in range(15): # zero cmd_vel for another 0.025 sec. Important! Or wrong obs
        env.cmd_vel0_pub.publish(Twist())
        env.cmd_vel1_pub.publish(Twist())
        env.rate.sleep()
    env.pausePhysics()

    # sampling trajectory
    obs = env._get_observation()
    # obs = env.reset()
    state_0 = obs.copy()
    state_1 = obs.copy()
    state_1[:6] = state_1[-6:]
    for st in range(num_steps):
        traj.append(obs)
        act0 = agent.epsilon_greedy(state_0)
        act1 = agent.epsilon_greedy(state_1)
        act = np.array([act0, act1])
        acts.append(act)
        # step env
        next_obs, rew, done, info = env.step(act)
        if 'blown' in info:
            break
        obs = next_obs.copy()
        state_0 = obs.copy() # + 0.5*random.randn(obs.shape[0])
        state_1 = obs.copy() # + 0.5*random.randn(obs.shape[0])
        state_1[:6] = state_1[-6:]
        if done:
            break
    # save traj
    traj.append(obs)
    traj = np.squeeze(np.array([traj]))
    acts = np.squeeze(np.array([acts]))
    if not os.path.exists(os.path.dirname(traj_path)):
        os.makedirs(os.path.dirname(traj_path))
    np.save(traj_path, traj)
    np.save(os.path.join(os.path.dirname(traj_path), 'acts.npy'), acts)
