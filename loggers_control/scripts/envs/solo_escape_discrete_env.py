#!/usr/bin/env python
"""
Task environment of logger solo escaping: discrete action space
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
from numpy import pi
from numpy import random
import time

import rospy
import tf
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState, LinkState, ModelStates, LinkStates
from geometry_msgs.msg import Pose, Twist


class SoloEscapeDiscreteEnv(object):
    """
    SoloEscapeDiscrete Env Class
    """
    def __init__(self):
        rospy.init_node("solo_escape_discrete_env", anonymous=True, log_level=rospy.INFO)
        # env properties
        self.name = 'solo_escape_discrete'
        self.rate = rospy.Rate(1000) # gazebo world is running at 1000 Hz
        self.max_steps = 999
        self.step_counter = 0
        self.observation_space = (6,) # x, y, x_d, y_d, th, th_d
        self.action_space = (4,)
        self.actions = np.array([[1.5,pi/3], [1.5,-pi/3], [-1.5,pi/3], [-1.5,-pi/3]])
        # robot properties
        self.spawning_pool = np.array([np.inf]*3)
        self.model_states = ModelStates()
        self.status = 'deactivated'
        self.world_name = rospy.get_param('/world_name')
        self.exit_width = rospy.get_param('/exit_width')
        # services
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        # topic publisher
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        # subscriber
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback) # model states are under monitoring

    def pausePhysics(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/pause_physics service call failed")

    def unpausePhysics(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/unpause_physics service call failed")

    # def resetSimulation(self):
    #     rospy.wait_for_service("/gazebo/reset_simulation")
    #     try:
    #         self.reset_simulation_proxy()
    #     except rospy.ServiceException as e:
    #         rospy.logerr("/gazebo/reset_simulation service call failed")

    def resetWorld(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_world service call failed")

    def setModelState(self, model_state):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state_proxy(model_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

    def reset(self):
        """
        Reset environment
        Usage:
            obs = env.reset()
        Args:
        Returns:
            obs
        """
        rospy.logdebug("\nStart Environment Reset")
        self.unpausePhysics()
        # zero cmd_vel
        zero_cmd_vel = Twist()
        for _ in range(15): # zero cmd_vel for about 0.025 sec. Important! Or wrong obs
            self.cmd_vel_pub.publish(zero_cmd_vel)
            self.rate.sleep()
        # set init pose
        self.pausePhysics()
        self.resetWorld()
        self._set_pose()
        self.unpausePhysics()
        for _ in range(15): # zero cmd_vel for another 0.025 sec. Important! Or wrong obs
            self.cmd_vel_pub.publish(zero_cmd_vel)
            self.rate.sleep()
        self.pausePhysics()
        # get obs
        obs = self._get_observation()
        # reset params
        self.step_counter = 0
        rospy.logerr("\nEnvironment Reset!!!\n")

        return obs

    def step(self, action):
        """
        Manipulate the environment with an action
        obs, rew, done, info = env.step(action)
        """
        rospy.logdebug("\nStart Environment Step")
        self.unpausePhysics()
        self._take_action(action)
        self.pausePhysics()
        obs = self._get_observation()
        # compute reward and done
        reward, done = self._compute_reward()
        self.step_counter += 1 # make sure inc step counter before compute reward
        info = self.status
        rospy.logdebug("End Environment Step\n")

        return obs, reward, done, info

    def _set_pose(self):
        """
        Set logger with a random or given pose
        Args:
        Returns:
        """
        logger_pose = ModelState()
        logger_pose.model_name = "logger"
        logger_pose.reference_frame = "world"
        logger_pose.pose.position.z = 0.1
        if sum(np.isinf(self.spawning_pool)): # inialize randomly
            x = random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            quat = tf.transformations.quaternion_from_euler(0, 0, random.uniform(-pi, pi))
        else: # inialize accordingly
            assert np.absolute(self.spawning_pool[0]) <= 4.5
            assert np.absolute(self.spawning_pool[1]) <= 4.5
            assert -pi<=self.spawning_pool[2]<= pi # theta within [-pi,pi]
            x = self.spawning_pool[0].copy()
            y = self.spawning_pool[1].copy()
            quat = tf.transformations.quaternion_from_euler(0, 0, self.spawning_pool[2].copy())
        logger_pose.pose.position.x = x
        logger_pose.pose.position.y = y
        logger_pose.pose.orientation.z = quat[2]
        logger_pose.pose.orientation.w = quat[3]
        self.setModelState(model_state=logger_pose)

    def _get_observation(self):
        """
        Get observation of double_logger's state
        Args:
        Returns:
            obs: array([x,y,xdot,ydot,theta,thetadot])
        """
        obs = np.zeros(self.observation_space[0])
        id_logger = self.model_states.name.index("logger")
        logger_pose = self.model_states.pose[id_logger]
        logger_twist = self.model_states.twist[id_logger]
        quat = [
            logger_pose.orientation.x,
            logger_pose.orientation.y,
            logger_pose.orientation.z,
            logger_pose.orientation.w
        ]
        euler = tf.transformations.euler_from_quaternion(quat)
        obs[0] = logger_pose.position.x
        obs[1] = logger_pose.position.y
        obs[2] = logger_twist.linear.x
        obs[3] = logger_twist.linear.y
        obs[4] = euler[2]
        obs[5] = logger_twist.angular.z
        # update status
        if obs[0] > 4.7:
            self.status = "east"
        elif obs[0] < -4.7:
            self.status = "west"
        elif obs[1] > 4.7:
            self.status = "north"
        elif -6<=obs[1]<=-4.7:
            if np.absolute(obs[0]) > self.exit_width/2.:
                self.status = "south"
            else:
                if np.absolute(obs[0]) > (self.exit_width/2.-0.255): # robot_radius=0.25
                    self.status = 'door' # stuck at door
                else:
                    self.status = "trapped" # tunneling through door
        elif obs[1] < -6.25:
            self.status = "escaped"
        else:
            self.status = "trapped"

        return obs

    def _take_action(self, i_act):
        """
        Publish cmd_vel according to an action index
        Args:
            action: int(scalar)
        Returns:
        """
        assert 0<=i_act<=self.action_space[0]
        rospy.logdebug("\nStart Taking Action")
        cmd_vel = Twist()
        cmd_vel.linear.x = self.actions[i_act][0]
        cmd_vel.angular.z = self.actions[i_act][1]
        for _ in range(30): # ~20 Hz
            self.cmd_vel_pub.publish(cmd_vel)
            self.rate.sleep()
        rospy.logdebug("cmd_vel: {}".format(cmd_vel))
        rospy.logdebug("End Taking Action\n")

    def _compute_reward(self):
        """
        Compute reward and done based on current status
        Return:
            reward:
            done
        """
        rospy.logdebug("\nStart Computing Reward")
        reward, done = 0, False
        if self.status == 'escaped':
            reward = 300.
            done = True
            rospy.logerr("\n!!!!!!!!!!!!!!!!\nLogger Escaped !\n!!!!!!!!!!!!!!!!")
        elif self.status == 'trapped':
            reward = -0.1
            done = False
            rospy.logdebug("\nLogger is trapped\n")
        else: # collision
            reward = -100.
            done = True
            rospy.logdebug("\nLogger had a collision\n")
        rospy.logdebug("reward: {}, done: {}".format(reward, done))
        # check if steps out of range
        if self.step_counter >= self.max_steps:
            done = True
            rospy.logwarn("Step: {}, \nMax step reached, env will reset...".format(self.step_counter))
        rospy.logdebug("End Computing Reward\n")

        return reward, done

    def _model_states_callback(self, data):
        self.model_states = data

if __name__ == "__main__":
    env = SoloEscapeDiscreteEnv()
    num_episodes = 4
    num_steps = env.max_steps
    for ep in range(num_episodes):
        obs = env.reset()
        rospy.logdebug("obs: {}".format(obs))
        for st in range(num_steps):
            act = random.randint(env.action_space[0])
            obs, rew, done, info = env.step(act)
            rospy.loginfo("\n-\nepisode: {}, step: {} \nobs: {}, act: {}, reward: {}, done: {}, info: {}".format(ep, st, obs, act, rew, done, info))
            if done:
                break
