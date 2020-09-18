#!/usr/bin/env python
"""
Solo escape environment with discrete action space
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
from numpy import pi
from numpy import random

import rospy
import tf
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose, Twist


class SoloEscape:
    
    def __init__(self):
        # env properties
        self.env_type = 'discrete'
        self.name = 'solo_escape_discrete'
        rospy.init_node(self.name, anonymous=True, log_level=rospy.DEBUG)
        self.rate = rospy.Rate(1000) # gazebo world is running at 1000 hz
        self.max_episode_steps = 1000
        self.observation_space_shape = (6,) # x, y, x_d, y_d, th, th_d
        self.action_space_shape = ()
        self.action_reservoir = np.array([[1.5,pi/3], [1.5,-pi/3], [-1.5,pi/3], [-1.5,-pi/3]])
        # robot properties
        self.model_states = ModelStates()
        self.obs = np.zeros(self.observation_space_shape)
        self.prev_obs = np.zeros(self.observation_space_shape)
        self.status = 'deactivated'
        self.world_name = rospy.get_param('/world_name')
        self.exit_width = rospy.get_param('/exit_width')
        # services
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
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
        rospy.logdebug("\nStart environment reset")
        # set init pose
        self.resetWorld()
        self.obs = self._set_pose(init_pose)
        self.prev_obs = self.obs.copy()
        self.step_counter = 0
        # self.y = obs[1]
        # self.prev_y = obs[1]
        rospy.logerr("\nEnvironment reset!!!")

        return self.obs

    def step(self, action_index):
        """
        obs, rew, done, info = env.step(action_index)
        """
        assert 0<=action_index<self.action_reservoir.shape[0]
        rospy.logdebug("\nStart Environment Step")
        action = self.action_reservoir[action_index]
        self._take_action(action)
        self._get_observation()
        # compute reward and done
        reward, done = self._compute_reward()
        self.prev_obs = self.obs.copy()
        info = self.status
        self.step_counter += 1 # make sure inc step counter before compute reward
        if self.step_counter>=self.max_episode_steps:
            rospy.logwarn("Step: {}, \nMax step reached...".format(self.step_counter))
        rospy.logdebug("End Environment Step\n")

        return self.obs, reward, done, info

    def _set_pose(self, pose=None):
        """
        Set logger with a random or a given pose
        Args:
            pose: array([x,y,\omega])
        Returns:
        """
        rospy.logdebug("\nStart setting pose...")
        logger_pose = ModelState()
        logger_pose.model_name = "logger"
        logger_pose.reference_frame = "world"
        logger_pose.pose.position.z = 0.1
        if pose is None: # random pose
            x = random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            th = random.uniform(-pi, pi)
        else: # inialize accordingly
            assert pose.shape==(3,)
            assert pose[0] <= 4.5
            assert pose[1] <= 4.5
            assert -pi<=pose[2]<= pi # theta within [-pi,pi]
            x = pose[0]
            y = pose[1]
            th = pose[2]
        quat = tf.transformations.quaternion_from_euler(0, 0, th)
        logger_pose.pose.position.x = x
        logger_pose.pose.position.y = y
        logger_pose.pose.orientation.z = quat[2]
        logger_pose.pose.orientation.w = quat[3]
        # set pose until on spot
        self.unpausePhysics()
        zero_vel = np.zeros(2)
        self._take_action(zero_vel)
        self.setModelState(model_state=logger_pose)
        self._take_action(zero_vel)
        self._get_observation()
        self.pausePhysics()
        rospy.logdebug("\nEND setting pose...")

        return self.obs

    def _get_observation(self):
        """
        Get observation of double_logger's state
        Args:
        Returns:
            obs: array([x,y,xdot,ydot,theta,thetadot])
        """
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
        self.obs[0] = logger_pose.position.x
        self.obs[1] = logger_pose.position.y
        self.obs[2] = logger_twist.linear.x
        self.obs[3] = logger_twist.linear.y
        self.obs[4] = euler[2]
        self.obs[5] = logger_twist.angular.z
        # update status
        if self.obs[0] > 4.7:
            self.status = "east"
        elif self.obs[0] < -4.7:
            self.status = "west"
        elif self.obs[1] > 4.7:
            self.status = "north"
        elif -6<=self.obs[1]<=-4.7:
            if np.absolute(self.obs[0]) > self.exit_width/2.:
                self.status = "south"
            else:
                if np.absolute(self.obs[0]) > (self.exit_width/2.-0.255): # robot_radius=0.25
                    self.status = 'door' # stuck at door
                else:
                    self.status = "trapped" # tunneling through door
        elif self.obs[1] < -6.25:
            self.status = "escaped"
        else:
            self.status = "trapped"

    def _take_action(self, action):
        """
        Publish cmd_vel according to an action index
        Args:
            action: int(scalar)
        Returns:
        """
        rospy.logdebug("\nStart taking action")
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]
        self.unpausePhysics()
        for _ in range(50): 
            self.cmd_vel_pub.publish(cmd_vel)
            self.rate.sleep()
        rospy.logdebug("cmd_vel: {}".format(cmd_vel))
        self.pausePhysics()
        rospy.logdebug("\nEnd taking action")

    def _compute_reward(self):
        """
        Compute reward and done based on current status
        Return:
            reward:
            done
        """
        rospy.logdebug("\nStart Computing Reward")
        reward, done = -.1, False
        if self.status == 'escaped':
            reward = 100.
            done = True
            rospy.logerr("\n!!!!!!!!!!!!!!!!\nLogger Escaped !\n!!!!!!!!!!!!!!!!")
        else:
            if self.status == 'trapped':
                if self.obs[1]<-5:
                    reward = 10*(self.prev_obs[1] - self.obs[1]) - 0.1
            else:
                reward = -100.
                done = True
        rospy.logdebug("End Computing Reward\n")

        return reward, done

    def _model_states_callback(self, data):
        self.model_states = data

if __name__ == "__main__":
    env = SoloEscape()
    num_steps = env.max_episode_steps
    obs = env.reset()
    ep, st = 0, 0
    for t in range(env.max_episode_steps):
        a = t%2
        o, r, d, i = env.step(a)
        st += 1
        rospy.loginfo("\n-\nepisode: {}, step: {} \nobs: {}, act: {}, reward: {}, done: {}, info: {}".format(ep+1, st, o, a, r, d, i))
        if d:
            ep += 1
            st = 0
            obs = env.reset()
            
