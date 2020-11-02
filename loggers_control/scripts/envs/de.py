#!/usr/bin/env python
"""
Doubld escape environment with discrete action space
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import math
import numpy as np
from numpy import pi
from numpy import random
import time

import rospy
import tf
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, SetLinkState, GetModelState, GetLinkState
from gazebo_msgs.msg import ModelState, LinkState, ModelStates, LinkStates
from geometry_msgs.msg import Pose, Twist


class DoubleEscape(object):

    def __init__(self, debug=False):
        self.env_type = 'discrete'
        self.name = 'double_escape_discrete'
        if debug:
            rospy.init_node(self.name, anonymous=True, log_level=rospy.DEBUG)
        else:
            rospy.init_node(self.name, anonymous=True, log_level=rospy.INFO)
        # env properties
        self.rate = rospy.Rate(1000)
        self.max_episode_steps = 1000
        self.observation_space_shape = (3,6) # {r1, r2, s}: x, y, x_d, y_d, th, th_d
        self.action_space_shape = ()
        self.action_reservoir = np.array([[3,pi/2], [3,-pi/2], [-3,pi/2], [-3,-pi/2]])
        # robot properties
        self.model_states = ModelStates()
        self.obs = np.zeros(self.observation_space_shape)
        self.prev_obs = np.zeros(self.observation_space_shape)
        self.status = ['deactivated']*2
        self.world_name = rospy.get_param('/world_name')
        self.exit_width = rospy.get_param('/exit_width')
        # services
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        # topic publisher
        self.cmd_vel0_pub = rospy.Publisher("/logger0/cmd_vel", Twist, queue_size=1)
        self.cmd_vel1_pub = rospy.Publisher("/logger1/cmd_vel", Twist, queue_size=1)
        # topic subscriber
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback)
        rospy.Subscriber("/gazebo/link_states", LinkStates, self._link_states_callback)

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

    def reset(self, init_pose=None):
        """
        Reset environment
        Usage:
            obs = env.reset()
        """
        rospy.logdebug("\nStart Environment Reset")
        # set init pose
        self.resetWorld()
        self.obs = self._set_pose(init_pose)
        self.prev_obs = self.obs.copy()
        self.step_counter = 0
        # self.y = np.array([obs[0,1], obs[1,1]])
        # self.prev_y = self.y.copy()
        rospy.logerr("\nEnvironment Reset!!!\n")

        return self.obs

    def step(self, action_indices):
        """
        obs, rew, done, info = env.step(action_indices)
        """
        assert 0<=action_indices[0]<self.action_reservoir.shape[0]
        assert 0<=action_indices[1]<self.action_reservoir.shape[0]
        rospy.logdebug("\nStart environment step")
        actions = np.zeros((2,2))
        actions[0] = self.action_reservoir[action_indices[0]]
        actions[1] = self.action_reservoir[action_indices[1]]
        self._take_action(actions)
        self._get_observation()
        # update status
        reward, done = self._compute_reward()
        self.prev_obs = self.obs.copy() # make sure this happened after reward computing
        info = self.status
        self.step_counter += 1
        if self.step_counter>=self.max_episode_steps:
            rospy.logwarn("Step: {}, \nMax step reached...".format(self.step_counter))
        rospy.logdebug("End environment step\n")

        return self.obs, reward, done, info

    def _set_pose(self, pose=None):
        """
        Set double_logger with a random or given pose.
        """
        rospy.logdebug("\nStart setting model pose")
        double_logger_pose = ModelState()
        double_logger_pose.model_name = "double_logger"
        # logger_pose.reference_frame = "world"
        double_logger_pose.pose.position.z = 0.09
        if pose is None: # random pose
            x = random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            th = random.uniform(-pi, pi)
            while any([
                    np.abs(x + 2*np.sin(th)) > 4.8,
                    np.abs(y - 2*np.cos(th)) > 4.8
            ]):
                th = random.uniform(-pi, pi)
            quat = tf.transformations.quaternion_from_euler(0, 0, th)
            rospy.logdebug("Set model pose @ {}".format((x,y,th)))
        else: # inialize accordingly
            assert pose.shape==(3,)
            assert pose[0] <= 4.5
            assert pose[1] <= 4.5
            assert -pi<=pose[2]<= pi # theta within [-pi,pi]
            assert np.abs(pose[0] + 2*np.sin(pose[2])) <= 4.8
            assert np.abs(pose[1] - 2*np.cos(pose[2])) <= 4.8
            x = pose[0]
            y = pose[1]
            th = pose[2]
            quat = tf.transformations.quaternion_from_euler(0, 0, th)
            rospy.logdebug("Set model pose @ {}".format(pose))

        double_logger_pose.pose.position.x = x
        double_logger_pose.pose.position.y = y
        double_logger_pose.pose.orientation.z = quat[2]
        double_logger_pose.pose.orientation.w = quat[3]
        # set pose until on spot
        self.unpausePhysics()
        zero_vel = np.zeros((2,2))
        self._take_action(zero_vel)
        self.setModelState(model_state=double_logger_pose)
        self._take_action(zero_vel)
        self._get_observation()
        self.pausePhysics()
        rospy.logdebug("\nEnd setting model pose")

        return self.obs

    def _get_observation(self):
        """
        Get observation of double_logger's state
        Args:
        Returns:
            obs: array([...pose+vel0...,pose+vell...pose+vel1...])
        """
        def extract_link_state(link_id):
            link_obs = np.zeros(6)
            pose = self.link_states.pose[link_id]
            twist = self.link_states.twist[link_id]
            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            euler = tf.transformations.euler_from_quaternion(quat)
            link_obs[0] = pose.position.x
            link_obs[1] = pose.position.y
            link_obs[2] = twist.linear.x
            link_obs[3] = twist.linear.y
            link_obs[4] = euler[2]
            link_obs[5] = twist.angular.z
            return link_obs

        rospy.logdebug("\nStart getting observation")
        # identify index of logger0, log, logger1
        id_logger0 = self.link_states.name.index("double_logger::logger0-chassis")
        id_log = self.link_states.name.index("double_logger::log")
        id_logger1 = self.link_states.name.index("double_logger::logger1-chassis")
        # extract states of interested links
        logger0_obs = extract_link_state(id_logger0)
        log_obs = extract_link_state(id_log)
        logger1_obs = extract_link_state(id_logger1)
        self.obs[0] = logger0_obs
        self.obs[1] = logger1_obs
        self.obs[-1] = log_obs
        # compute logger0's status
        if self.obs[0,0] > 4.7:
            self.status[0] = 'east'
        elif self.obs[0,0] < -4.7:
            self.status[0] = 'west'
        elif self.obs[0,1] > 4.7:
            self.status[0] = 'north'
        elif -6<=self.obs[0,1]<-4.7:
            if np.absolute(self.obs[0,0])>self.exit_width/2.:
                self.status[0] = 'south'
            else:
                if np.absolute(self.obs[0,0])>(self.exit_width/2.-0.255): # robot_radius=0.25
                    self.status[0] = 'door' # stuck at door
                else:
                    self.status[0] = 'trapped' # through door
        elif self.obs[0,1] < -6.25:
            self.status[0] = 'escaped'
        else:
            self.status[0] = 'trapped'
        # compute logger1's status
        if self.obs[1,0] > 4.7:
            self.status[1] = 'east'
        elif self.obs[1,0] < -4.7:
            self.status[1] = 'west'
        elif self.obs[1,1] > 4.7:
            self.status[1] = 'north'
        elif -6<=self.obs[1,1]<-4.7:
            if np.absolute(self.obs[1,0])>self.exit_width/2.:
                self.status[1] = 'south'
            else:
                if np.absolute(self.obs[1,0])>(self.exit_width/2.-0.255): # robot_radius=0.25
                    self.status[1] = 'door' # stuck at door
                else:
                    self.status[1] = 'trapped' # through door
        elif self.obs[1,1] < -6.25:
            self.status[1] = 'escaped'
        else:
            self.status[1] = 'trapped'
        # detect if simulation blow up
        if self.link_states.pose[id_logger0].position.z > 0.1 or self.link_states.pose[id_logger0].position.z < 0.080:
            self.status[0] = 'blown'
        if self.link_states.pose[id_logger1].position.z > 0.1 or self.link_states.pose[id_logger1].position.z < 0.080:
            self.status[1] = 'blown'
        rospy.logdebug("\nEnd getting observation")

    def _take_action(self, actions):
        """
        Publish cmd_vel according to an action index
        Args:
            i_act: array([ia0, ia1])
        Returns:
        """
        rospy.logdebug("\nStart Taking Action")
        cmd_vel0 = Twist()
        cmd_vel0.linear.x = actions[0,0]
        cmd_vel0.angular.z = actions[0,1]
        cmd_vel1 = Twist()
        cmd_vel1.linear.x = actions[1,0]
        cmd_vel1.angular.z = actions[1,1]
        self.unpausePhysics()
        for _ in range(50):
            self.cmd_vel0_pub.publish(cmd_vel0)
            self.cmd_vel1_pub.publish(cmd_vel1)
            self.rate.sleep()
        rospy.logdebug("cmd_vel0: {} \ncmd_vel1: {}".format(cmd_vel0, cmd_vel1))
        self.pausePhysics()
        rospy.logdebug("End Taking Action\n")

    def _compute_reward(self):
        """
        Compute reward and done based on current status
        Return:
            reward
            done
        """
        rospy.logdebug("\nStart Computing Reward")
        reward, done = -.1*np.ones(2), False
        if any([
                'north' in self.status,
                'south' in self.status,
                'west' in self.status,
                'east' in self.status,
                'door' in self.status,
                'blown' in self.status,
        ]):
            # reward = -100*np.ones(2)
            done = True
        elif self.status.count('escaped')==2:
            reward = 100*np.ones(2)
            done = True
            rospy.logerr("\n!!!!!!!!!!!!!!!!\nLogger Escaped !\n!!!!!!!!!!!!!!!!")
        else:
            reward[0] = 100*(self.prev_obs[0,1]-self.obs[0,1] + abs(self.prev_obs[0,0])-abs(self.obs[0,0]))
            reward[1] = 100*(self.prev_obs[1,1]-self.obs[1,1] + abs(self.prev_obs[1,0])-abs(self.obs[1,0]))
            # if self.obs[0,1]<-5:
            #     reward[0] = 10*(self.prev_obs[0,1]-self.obs[0,1]) - .1
            # if self.obs[1,1]<-5:
            #     reward[1] = 10*(self.prev_obs[1,1]-self.obs[1,1]) - .1

        rospy.logdebug("End Computing Reward\n")

        return reward, done

    def _model_states_callback(self, data):
        self.model_states = data

    def _link_states_callback(self, data):
        self.link_states = data

if __name__ == "__main__":
    env = DoubleEscape()
    num_steps = 4*env.max_episode_steps
    obs = env.reset()
    ep, st = 0, 0
    o = env.reset()
    a = np.random.randint(0,4,2)
    for t in range(num_steps):
        s0 = o[[0,1]].flatten()
        s1 = o[[1,0]].flatten()
        o_, r, d, i = env.step(a)
        s0_ = o_[[0,1]].flatten()
        s1_ = o_[[1,0]].flatten()
        st += 1
        rospy.loginfo("\n-\nepisode: {}, step: {} \nobs: {}, act: {}, reward: {}, done: {}, info: {} \nstate_0: {} \nstate_1: {} \nstate_0_: {} \nstate_1_: {}".format(ep+1, st, o, a, r, d, i, s0, s1, s0_, s1_))
        o = o_.copy()
        if d:
            ep += 1
            st = 0
            o = env.reset()
            a = np.random.randint(0,4,2)
