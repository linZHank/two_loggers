#!/usr/bin/env python
"""
Task environment of double_logger cooperatively escaping: discrete action space.
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
from gazebo_msgs.srv import SetModelState, SetLinkState, GetModelState, GetLinkState
from gazebo_msgs.msg import ModelState, LinkState, ModelStates, LinkStates
from geometry_msgs.msg import Pose, Twist


class DoubleEscapeDiscreteEnv(object):
    """
    DoubleEscapeDiscrete Env Class
    """
    def __init__(self):
        rospy.init_node("double_escape_discrete_env", anonymous=True, log_level=rospy.DEBUG)
        # env properties
        self.name = 'double_escape_discrete'
        self.rate = rospy.Rate(1000)
        self.max_steps = 999
        self.step_counter = 0
        self.observation_space = (18,) # x, y, x_d, y_d, th, th_d
        self.action_space = (10,)
        self.actions0 = np.array([[2,1], [2,-1], [-2,1], [-2,-1], [0,0]])
        self.actions1 = self.actions0.copy()
        # robot properties
        self.spawning_pool = np.array([np.inf]*3)
        self.model_states = ModelStates()
        self.link_states = ModelStates()
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
        self.cmdvel0_pub = rospy.Publisher("/cmd_vel0", Twist, queue_size=1)
        self.cmdvel1_pub = rospy.Publisher("/cmd_vel1", Twist, queue_size=1)
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

    def setLinkState(self, link_state):
        rospy.wait_for_service('/gazebo/set_link_state')
        try:
            self.set_link_state_proxy(link_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

    def reset(self, init_pose=[]):
        """
        Reset environment
        Usage:
            obs = env.reset()
        Args:
        Returns:
            obs
        """
        rospy.logdebug("\nStart Environment Reset")
        # zero cmd_vels
        zero_cmd_vel = Twist()
        for _ in range(15): # zero cmd_vel for about 0.025 sec. Important! Or wrong obs
            self.cmdvel0_pub.publish(zero_cmd_vel)
            self.cmdvel1_pub.publish(zero_cmd_vel)
            self.rate.sleep()
        # set init pose
        self.pausePhysics()
        self.resetWorld()
        self._set_pose()
        self.unpausePhysics()
        for _ in range(15): # zero cmd_vel for another 0.025 sec. Important! Or wrong obs
            self.cmdvel0_pub.publish(zero_cmd_vel)
            self.cmdvel1_pub.publish(zero_cmd_vel)
            self.rate.sleep()
        # get obs
        obs = self._get_observation()
        # reset params
        self.status = 'trapped'
        self.step_counter = 0
        rospy.logerr("\nEnvironment Reset!!!\n")

        return obs

    def step(self, action_0, action_1):
        """
        Manipulate logger_0 with action_0, logger_1 with action_1
        obs, rew, done, info = env.step(action_0, action_1)
        """
        rospy.logdebug("\nStart Environment Step")
        self._take_action(action_0, action_1)
        obs = self._get_observation()
        reward, done = self._compute_reward()
        info = self._post_information()
        self.steps += 1
        rospy.logdebug("End Environment Step\n")

        return obs, reward, done, info

    def _set_pose(self):
        """
        Set logger with a random or given pose
        """
        logger_pose = ModelState()
        logger_pose.model_name = "double_logger"
        # logger_pose.reference_frame = "world"
        logger_pose.pose.position.z = 0.1
        if sum(np.isinf(self.spawning_pool)): # inialize randomly
            x = random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            theta = random.uniform(-pi, pi)
            # recalc pose if not in cell
            while np.absolute(x+2*np.cos(theta))>4.5 or np.absolute(y+2*np.sin(theta)) > 4.5:
                x = random.uniform(-4, 4)
                y = random.uniform(-4, 4)
                theta = random.uniform(-pi, pi)
            quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        else: # inialize accordingly
            assert np.absolute(self.spawning_pool[0]) <= 4.5
            assert np.absolute(self.spawning_pool[1]) <= 4.5
            assert -pi<=self.spawning_pool[2]<= pi # theta within [-pi,pi]
            assert np.absolute(self.spawning_pool[0]+2*np.cos(self.spawning_pool[2])) <= 4.5
            assert np.absolute(self.spawning_pool[1]+2*np.sin(self.spawning_pool[2])) <= 4.5
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
            obs: array([...pose+vel0...,pose+vell...pose+vel1...])
        """
        def extract_link_obs(link_id):
            link_obs = np.zeros(6)
            pose = self.link_states.pose[link_id]
            twist = self.link_states.twist[link_id]
            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            euler = tf.transformations.euler_from_quaternion(quat)
            link_obs[0] = pose.position.x
            link_obs[1] = pose.position.y
            link_obs[2] = twist.linear.x
            link_obs[3] = twist.linear.y
            link_obs[4] = quat[2]
            link_obs[5] = twist.angular.z
            return link_obs
        # compute obs from link_states
        obs = np.zeros(self.observation_space[0])
        # identify index of logger0, log, logger1
        id_logger0 = self.link_states.name.index("double_logger::logger0-chassis")
        id_log = self.link_states.name.index("double_logger::log")
        id_logger1 = self.link_states.name.index("double_logger::logger1-chassis")
        # extract observation of interested links
        logger0_obs = extract_link_obs(id_logger0)
        log_obs = extract_link_obs(id_log)
        logger1_obs = extract_link_obs(id_logger1)
        obs = np.concatenate((logger0_obs,log_obs,logger1_obs))

        return obs

    # def _set_init(self, init_pose):
    #     """
    #     Set initial condition of two_loggers to a specific pose
    #     Args:
    #         init_pose: [rod::x, y, angle, robot_0::angle, robot_1::angle], set to a random pose if empty
    #     """
    #     rospy.logdebug("\nStart Initializing Robots")
    #     # prepare
    #     self._take_action(np.zeros(2), np.zeros(2))
    #     self.pausePhysics()
    #     self.resetWorld()
    #     # set rod pose
    #     rod_state = ModelState()
    #     rod_state.model_name = "two_loggers"
    #     rod_state.reference_frame = "world"
    #     rod_state.pose.position.z = 0.245
    #     if init_pose: # random initialize
    #         assert -pi<=init_pose[2]<= pi # theta within [-pi,pi]
    #         assert -pi<=init_pose[3]<= pi
    #         assert -pi<=init_pose[4]<= pi
    #     else:
    #         init_pose = generate_random_pose()
    #     rod_state.pose.position.x = init_pose[0]
    #     rod_state.pose.position.y = init_pose[1]
    #     quat = tf.transformations.quaternion_from_euler(0, 0, init_pose[2])
    #     rod_state.pose.orientation.z = quat[2]
    #     rod_state.pose.orientation.w = quat[3]
    #     # call '/gazebo/set_model_state' service
    #     self.setModelState(model_state=rod_state)
    #     rospy.logdebug("two-logges was initialized at {}".format(rod_state))
    #     self.unpausePhysics()
    #     link_states = self.link_states
    #     self.pausePhysics()
    #     # set robot orientation
    #     q0 = tf.transformations.quaternion_from_euler(0, 0, init_pose[3])
    #     q1 = tf.transformations.quaternion_from_euler(0, 0, init_pose[4])
    #     # set white robot orientation
    #     robot0_name = 'two_loggers::link_chassis_0'
    #     robot0_state = LinkState()
    #     robot0_state.link_name = robot0_name
    #     robot0_state.reference_frame = 'world'
    #     robot0_state.pose = link_states.pose[link_states.name.index(robot0_name)]
    #     robot0_state.pose.orientation.z = q0[2]
    #     robot0_state.pose.orientation.z = q0[3]
    #     # set black robot orientation
    #     robot1_name = 'two_loggers::link_chassis_1'
    #     robot1_state = LinkState()
    #     robot1_state.link_name = robot1_name
    #     robot1_state.reference_frame = 'world'
    #     robot1_state.pose = link_states.pose[link_states.name.index(robot1_name)]
    #     robot1_state.pose.orientation.z = q1[2]
    #     robot1_state.pose.orientation.z = q1[3]
    #     # call '/gazebo/set_link_state' service
    #     self.setLinkState(link_state=robot0_state)
    #     self.setLinkState(link_state=robot1_state)
    #     self.unpausePhysics()
    #     self._take_action(np.zeros(2), np.zeros(2))
    #     rospy.logdebug("\ntwo_loggers initialized at {} \nlogger_0 orientation: {} \nlogger_1 orientation".format(rod_state, init_pose[3], init_pose[4]))
    #     # episode should not be done
    #     self._episode_done = False
    #     rospy.logdebug("End Initializing Robots\n")
    #
    # def _get_observation(self):
    #     """
    #     Get observation from link_states
    #     Return:
    #         observation: {"log{"pose", "twist"}", logger0{"pose", "twist"}", logger1{"pose", "twist"}"}
    #     """
    #     rospy.logdebug("\nStart Getting Observation")
    #     link_states = self.link_states
    #     self.pausePhysics()
    #     # the log
    #     id_log = link_states.name.index('two_loggers::link_log')
    #     self.observation['log']['pose'] = link_states.pose[id_log]
    #     self.observation['log']['twist'] = link_states.twist[id_log]
    #     # logger_0
    #     id_logger_0 = link_states.name.index('two_loggers::link_chassis_0')
    #     self.observation['logger_0']['pose'] = link_states.pose[id_logger_0]
    #     self.observation['logger_0']['twist'] = link_states.twist[id_logger_0]
    #     # logger_1
    #     id_logger_1 = link_states.name.index('two_loggers::link_chassis_1')
    #     self.observation['logger_1']['pose'] = link_states.pose[id_logger_1]
    #     self.observation['logger_1']['twist'] = link_states.twist[id_logger_1]
    #     # compute logger_0's status
    #     if self.observation['logger_0']['pose'].position.x > 4.745:
    #         self.status[0] = 'east'
    #     elif self.observation['logger_0']['pose'].position.x < -4.745:
    #         self.status[0] = 'west'
    #     elif self.observation['logger_0']['pose'].position.y > 4.745:
    #         self.status[0] = 'north'
    #     elif -6<=self.observation['logger_0']['pose'].position.y < -4.745:
    #         # if np.absolute(self.observation['logger_0']['pose'].position.x) > 1:
    #         if np.absolute(self.observation['logger_0']['pose'].position.x) > self.exit_width/2.:
    #             self.status[0] = 'south'
    #         else:
    #             # if np.absolute(self.observation['logger_0']['pose'].position.x) > 0.79:
    #             if np.absolute(self.observation['logger_0']['pose'].position.x) > (self.exit_width/2-0.25-0.005): # robot_radius=0.25
    #                 self.status[0] = 'door' # stuck at door
    #             else:
    #                 self.status[0] = 'tunnel' # through door
    #     elif self.observation['logger_0']['pose'].position.y < -6:
    #         self.status[0] = 'escaped'
    #     elif self.observation['logger_0']['pose'].position.z > 0.1 or self.observation['logger_0']['pose'].position.z < 0.08:
    #         self.status[0] = 'blew'
    #     else:
    #         self.status[0] = 'trapped'
    #     # compute logger_1's status
    #     if self.observation['logger_1']['pose'].position.x > 4.745:
    #         self.status[1] = 'east'
    #     elif self.observation['logger_1']['pose'].position.x < -4.745:
    #         self.status[1] = 'west'
    #     elif self.observation['logger_1']['pose'].position.y > 4.745:
    #         self.status[1] = 'north'
    #     elif -6<=self.observation['logger_1']['pose'].position.y < -4.745:
    #         # if np.absolute(self.observation['logger_1']['pose'].position.x) > 1:
    #         if np.absolute(self.observation['logger_1']['pose'].position.x) > self.exit_width/2.:
    #             self.status[1] = 'south'
    #         else:
    #             # if np.absolute(self.observation['logger_1']['pose'].position.x) > 0.79:
    #             if np.absolute(self.observation['logger_1']['pose'].position.x) > (self.exit_width/2-0.25-0.005): # robot_radius=0.25
    #                 self.status[1] = 'door' # stuck at door
    #             else:
    #                 self.status[1] = 'tunnel' # through door
    #     elif self.observation['logger_1']['pose'].position.y < -6:
    #         self.status[1] = 'escaped'
    #     elif self.observation['logger_1']['pose'].position.z > 0.1 or  self.observation['logger_1']['pose'].position.z < 0.08:
    #         self.status[1] = 'blew'
    #     else:
    #         self.status[1] = 'trapped'
    #     self.unpausePhysics()
    #     # logging
    #     rospy.logdebug("Observation Get ==> {}".format(self.observation))
    #     rospy.logdebug("End Getting Observation\n")
    #
    #     return self.observation
    #
    # def _take_action(self, action_0, action_1):
    #     """
    #     Set linear and angular speed for logger_0 and logger_1 to execute.
    #     Args:
    #         action: 2x np.array([v_lin,v_ang]).
    #     """
    #     rospy.logdebug("\nStart Taking Actions")
    #     cmd_vel_0 = Twist()
    #     cmd_vel_0.linear.x = action_0[0]
    #     cmd_vel_0.angular.z = action_0[1]
    #     cmd_vel_1 = Twist()
    #     cmd_vel_1.linear.x = action_1[0]
    #     cmd_vel_1.angular.z = action_1[1]
    #     for _ in range(15):
    #         self.cmdvel0_pub.publish(cmd_vel_0)
    #         self.cmdvel1_pub.publish(cmd_vel_1)
    #         self.rate.sleep()
    #     self.action_0 = action_0
    #     self.action_1 = action_1
    #     rospy.logdebug("\nlogger_0 take action ===> {}\nlogger_1 take action ===> {}".format(cmd_vel_0, cmd_vel_1))
    #     rospy.logdebug("End Taking Actions\n")
    #
    # def _compute_reward(self):
    #     """
    #     Return:
    #         reward: reward in current step
    #     """
    #     rospy.logdebug("\nStart Computing Reward")
    #     if self.status[0] == "escaped" and self.status[1] == "escaped":
    #         self.reward = 1
    #         self.success_count += 1
    #         self._episode_done = True
    #         rospy.logerr("\nDouble Escape Succeed!\n")
    #     else:
    #         self.reward = -0.
    #         self._episode_done = False
    #         rospy.loginfo("The loggers are trapped in the cell...")
    #     rospy.logdebug("Stepwise Reward: {}, Success Count: {}".format(self.reward, self.success_count))
    #     # check if steps out of range
    #     if self.steps > self.max_step:
    #         self._episode_done = True
    #         rospy.logwarn("Step: {}, \nMax step reached, env will reset...".format(self.steps))
    #     rospy.logdebug("End Computing Reward\n")
    #
    #     return self.reward, self._episode_done
    #
    # def _post_information(self):
    #     """
    #     Return:
    #         info: {"system status"}
    #     """
    #     rospy.logdebug("\nStart Posting Information")
    #     self.info["status"] = self.status
    #     rospy.logdebug("End Posting Information\n")
    #
    #     return self.info

    def _model_states_callback(self, data):
        self.model_states = data

    def _link_states_callback(self, data):
        self.link_states = data


if __name__ == "__main__":
    num_episodes = 16
    num_steps = 64

    env = DoubleEscapeDiscreteEnv()
    for ep in range(num_episodes):
        obs = env.reset()
        rospy.logdebug("obs: {}".format(obs))
        # for st in range(num_steps):
        #     act = random.randint(env.action_space[0])
        #     obs, rew, done, info = env.step(act)
        #     rospy.loginfo("\n-\nepisode: {}, step: {} \nobs: {}, reward: {}, done: {}, info: {}".format(ep, st, obs, rew, done, info))
