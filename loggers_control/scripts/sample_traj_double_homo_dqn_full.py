#! /usr/bin/env python
"""
Sample a trajectory with specified DQN model and initial conditions.
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


def _extract_yaw(orientation):
    quat = np.zeros(4)
    quat[0] =  orientation.x
    quat[1] =  orientation.y
    quat[2] =  orientation.z
    quat[3] =  orientation.w
    euler = tf.transformations.euler_from_quaternion(quat)
    yaw = euler[2]

    return yaw
    
if __name__ == "__main__":
    env=DoubleEscapeDiscreteEnv()
    agent = DQNAgent(env=env, name='double_logger_eval')
    model_path = os.path.join(sys.path[0], 'saved_models/double_escape_discrete/dqn/2020-05-29-17-33/double_logger/models/5093500.h5')
    agent.load_model(model_path=model_path)
    agent.epsilon = 0.
    num_steps = env.max_steps
    # Set double_logger to specified pose
    init_pose = np.array([0,0,0,pi/4,-pi/4]) # modify this to any pose as needed: [x, y, theta, th0, th1]
    env.reset()
    # Set double_logger pose
    double_logger_pose = ModelState()
    double_logger_pose.model_name = "double_logger"
    double_logger_pose.pose.position.z = 0.09
    x = init_pose[0]
    y = init_pose[1]
    theta = init_pose[2]
    th0 = init_pose[3]
    th1 = init_pose[4]
    quat = tf.transformations.quaternion_from_euler(0, 0, theta)
    # q0 = tf.transformations.quaternion_from_euler(0, 0, th0)
    # q1 = tf.transformations.quaternion_from_euler(0, 0, th1)
    double_logger_pose.pose.position.x = x
    double_logger_pose.pose.position.y = y
    double_logger_pose.pose.orientation.z = quat[2]
    double_logger_pose.pose.orientation.w = quat[3]
    env.setModelState(model_state=double_logger_pose)
    env.unpausePhysics()
    # Make sure double_logger not moving
    for _ in range(10): 
        env.cmd_vel0_pub.publish(Twist())
        env.cmd_vel1_pub.publish(Twist())
        env.rate.sleep()
    # Identify indices of robot0 and robot1 in link_states
    i0 = env.link_states.name.index("double_logger::logger0-chassis")
    i1 = env.link_states.name.index("double_logger::logger1-chassis")
    ori0 = env.link_states.pose[i0].orientation
    ori1 = env.link_states.pose[i1].orientation
    yaw0 = _extract_yaw(ori0)
    yaw1 = _extract_yaw(ori1)
    spin_cmd_vel = Twist()
    spin_cmd_vel.angular.z = pi/4
    while np.absolute(yaw0-th0) >= pi/10:
        env.cmd_vel1_pub.publish(Twist())
        env.cmd_vel0_pub.publish(spin_cmd_vel)
        ori0 = env.link_states.pose[i0].orientation 
        yaw0 = _extract_yaw(ori0)
        env.rate.sleep()
    print("robot 0 orientation set")
    env.cmd_vel0_pub.publish(Twist())
    while np.absolute(yaw1-th1) >= pi/10:
        env.cmd_vel0_pub.publish(Twist())
        env.cmd_vel1_pub.publish(spin_cmd_vel)
        ori1 = env.link_states.pose[i1].orientation 
        yaw1 = _extract_yaw(ori1)
        env.rate.sleep()
    print("robot 1 orientation set")
    env.cmd_vel1_pub.publish(Twist())
    # Set double_logger pose again
    env.pausePhysics()
    env.setModelState(model_state=double_logger_pose)
    # for _ in range(10): 
    #     env.cmd_vel0_pub.publish(Twist())
    #     env.cmd_vel1_pub.publish(Twist())
    #     env.rate.sleep()

