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


def generate_random_pose():
    """
    Special thanks to Eric Sun for sharing the code to calculate random init pose.

    generate a random rod pose in the room with center at (0, 0), width 10 meters and depth 10 meters. The robot has 0.25 meters as radius
    Returns:
        random_pose
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

    random_pose = []
    mag = 4.
    len_rod = 2
    room = [-mag, mag, -mag, mag] # create a room with boundary
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
    random_pose = [x, y, angle, th_0, th_1]

    return random_pose

class DoubleEscapeContinuousEnv(object):
    """
    DoubleEscapeContinuous Env Class
    """
    def __init__(self):
        rospy.init_node("double_escape_continuous_env", anonymous=True, log_level=rospy.DEBUG)
        # env properties
        self.name = 'double_escape_continuous'
        self.rate = rospy.Rate(1000)
        self.max_steps = 999
        self.step_counter = 0
        self.observation_space = (18,) # x, y, x_d, y_d, th, th_d
        self.action_space = (4,)
        self.action_space_high = np.array([1.5, pi/3, 1.5, pi/3])
        self.action_space_low = np.array([-1.5, -pi/3, -1.5, -pi/3])
        self.action_scale = np.array([1.5, pi/3, 1.5, pi/3])
        # robot properties
        self.spawning_pool = np.array([np.inf]*5)
        self.model_states = ModelStates()
        self.link_states = ModelStates()
        self.status = ['deactivated']*2
        self.world_name = rospy.get_param('/world_name')
        self.exit_width = rospy.get_param('/exit_width')
        # services
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_link_state_proxy = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
        self.get_link_state_proxy = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        # topic publisher
        self.cmd_vel0_pub = rospy.Publisher("/cmd_vel0", Twist, queue_size=1)
        self.cmd_vel1_pub = rospy.Publisher("/cmd_vel1", Twist, queue_size=1)
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
        self.unpausePhysics()
        # zero cmd_vels
        zero_cmd_vel = Twist()
        for _ in range(15): # zero cmd_vel for about 0.025 sec. Important! Or wrong obs
            self.cmd_vel0_pub.publish(zero_cmd_vel)
            self.cmd_vel1_pub.publish(zero_cmd_vel)
            self.rate.sleep()
        # set init pose
        self.pausePhysics()
        self.resetWorld()
        self._set_pose()
        self.unpausePhysics()
        for _ in range(15): # zero cmd_vel for another 0.025 sec. Important! Or wrong obs
            self.cmd_vel0_pub.publish(zero_cmd_vel)
            self.cmd_vel1_pub.publish(zero_cmd_vel)
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
        Manipulate logger0 with action0, logger1 with action1
        obs, rew, done, info = env.step(action_0, action_1)
        """
        rospy.logdebug("\nStart Environment Step")
        self.unpausePhysics()
        self._take_action(action)
        self.pausePhysics()
        obs = self._get_observation()
        # update status
        reward, done = self._compute_reward()
        info = self.status
        self.step_counter += 1
        rospy.logdebug("End Environment Step\n")

        return obs, reward, done, info

    def _set_pose(self):
        """
        Set double_logger with a random or given pose.
        """
        double_logger_pose = ModelState()
        double_logger_pose.model_name = "double_logger"
        # logger_pose.reference_frame = "world"
        double_logger_pose.pose.position.z = 0.09
        if sum(np.isinf(self.spawning_pool)): # inialize randomly
            x = random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            theta = random.uniform(-pi, pi)
            th0 = random.uniform(-pi, pi)
            th1 = random.uniform(-pi, pi)
            # recalc pose if not in cell
            while np.absolute(x+2*np.cos(theta))>4.5 or np.absolute(y+2*np.sin(theta)) > 4.5:
                x = random.uniform(-4, 4)
                y = random.uniform(-4, 4)
                theta = random.uniform(-pi, pi)
        else: # inialize accordingly
            assert np.absolute(self.spawning_pool[0]) <= 4.5
            assert np.absolute(self.spawning_pool[1]) <= 4.5
            assert -pi<=self.spawning_pool[2]<= pi # theta within [-pi,pi]
            assert np.absolute(self.spawning_pool[0]+2*np.cos(self.spawning_pool[2])) <= 4.5
            assert np.absolute(self.spawning_pool[1]+2*np.sin(self.spawning_pool[2])) <= 4.5
            assert -pi<=self.spawning_pool[3]<= pi # theta within [-pi,pi]
            assert -pi<=self.spawning_pool[4]<= pi # theta within [-pi,pi]
            x = self.spawning_pool[0].copy()
            y = self.spawning_pool[1].copy()
            theta = self.spawning_pool[2].copy()
            th0 = self.spawning_pool[3].copy()
            th1 = self.spawning_pool[4].copy()
        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        q0 = tf.transformations.quaternion_from_euler(0, 0, th0)
        q1 = tf.transformations.quaternion_from_euler(0, 0, th1)
        double_logger_pose.pose.position.x = x
        double_logger_pose.pose.position.y = y
        double_logger_pose.pose.orientation.z = quat[2]
        double_logger_pose.pose.orientation.w = quat[3]
        self.setModelState(model_state=double_logger_pose)
        # # set individual loggers pose
        # TODO: set individual orientation
        # # set logger0 pose
        # id_logger0 = self.link_states.name.index("double_logger::logger0-chassis")
        # logger0_pose = LinkState()
        # logger0_pose.link_name = 'double_logger::logger0-chassis'
        # logger0_pose.pose.position.x = self.link_states.pose[id_logger0].position.x
        # logger0_pose.pose.position.y = self.link_states.pose[id_logger0].position.y
        # logger0_pose.pose.position.z = self.link_states.pose[id_logger0].position.z
        # # logger0_pose.pose.orientation.x = self.link_states.pose[id_logger0].orientation.x
        # # logger0_pose.pose.orientation.y = self.link_states.pose[id_logger0].orientation.y
        # logger0_pose.pose.orientation.z = q0[2]
        # logger0_pose.pose.orientation.w = q0[3]
        # self.setLinkState(link_state=logger0_pose)
        # # set logger0 pose
        # id_logger1 = self.link_states.name.index("double_logger::logger1-chassis")
        # logger1_pose = LinkState()
        # logger1_pose.link_name = 'double_logger::logger1-chassis'
        # logger1_pose.pose.position.x = self.link_states.pose[id_logger1].position.x
        # logger1_pose.pose.position.y = self.link_states.pose[id_logger1].position.y
        # logger1_pose.pose.position.z = self.link_states.pose[id_logger1].position.z
        # # logger1_pose.pose.orientation.x = self.link_states.pose[id_logger1].orientation.x
        # # logger1_pose.pose.orientation.y = self.link_states.pose[id_logger1].orientation.y
        # logger1_pose.pose.orientation.z = q1[2]
        # logger1_pose.pose.orientation.w = q1[3]
        # self.setLinkState(link_state=logger1_pose)

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
            link_obs[4] = euler[2]
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
        # compute logger0's status
        if obs[0] > 4.7:
            self.status[0] = 'east'
        elif obs[0] < -4.7:
            self.status[0] = 'west'
        elif obs[1] > 4.7:
            self.status[0] = 'north'
        elif -6<=obs[1]<-4.7:
            if np.absolute(obs[0])>self.exit_width/2.:
                self.status[0] = 'south'
            else:
                if np.absolute(obs[0])>(self.exit_width/2.-0.255): # robot_radius=0.25
                    self.status[0] = 'door' # stuck at door
                else:
                    self.status[0] = 'trapped' # through door
        elif obs[1] < -6.25:
            self.status[0] = 'escaped'
        else:
            self.status[0] = 'trapped'
        # compute logger1's status
        if obs[-6] > 4.7:
            self.status[1] = 'east'
        elif obs[-6] < -4.7:
            self.status[1] = 'west'
        elif obs[-5] > 4.7:
            self.status[1] = 'north'
        elif -6<=obs[-5]<-4.7:
            if np.absolute(obs[-6])>self.exit_width/2.:
                self.status[1] = 'south'
            else:
                if np.absolute(obs[-6])>(self.exit_width/2.-0.255): # robot_radius=0.25
                    self.status[1] = 'door' # stuck at door
                else:
                    self.status[1] = 'trapped' # through door
        elif obs[-5] < -6.25:
            self.status[1] = 'escaped'
        else:
            self.status[1] = 'trapped'
        # detect if simulation blow up
        if self.link_states.pose[id_logger0].position.z > 0.1 or  self.link_states.pose[id_logger0].position.z < 0.080:
            self.status[0] = 'blown'
        if self.link_states.pose[id_logger1].position.z > 0.1 or  self.link_states.pose[id_logger1].position.z < 0.080:
            self.status[1] = 'blown'

        return obs

    def _take_action(self, acts):
        """
        Publish cmd_vel according to an action array
        Args:
            acts: array([lin.x0, ang.z0, lin.x1, ang.z1])
        Returns:
        """
        assert acts.shape==self.action_space
        rospy.logdebug("\nStart Taking Action")
        # clip acts
        lin_x0 = np.clip(acts[0], self.action_space_low[0], self.action_space_high[0]) # -1.5 ~ 1.5
        ang_z0 = np.clip(acts[1], self.action_space_low[1], self.action_space_high[1]) # -pi/3 ~ pi/3
        lin_x1 = np.clip(acts[2], self.action_space_low[2], self.action_space_high[2]) # -1.5 ~ 1.5
        ang_z1 = np.clip(acts[3], self.action_space_low[3], self.action_space_high[3]) # -pi/3 ~ pi/3
        # convert acts to cmd_vels
        cmd_vel0 = Twist()
        cmd_vel0.linear.x = lin_x0
        cmd_vel0.angular.z = ang_z0
        cmd_vel1 = Twist()
        cmd_vel1.linear.x = lin_x1
        cmd_vel1.angular.z = ang_z1
        for _ in range(30): # ~20 Hz
            self.cmd_vel0_pub.publish(cmd_vel0)
            self.cmd_vel1_pub.publish(cmd_vel1)
            self.rate.sleep()
        rospy.logdebug("cmd_vel0: {} \ncmd_vel1: {}".format(cmd_vel0, cmd_vel1))
        rospy.logdebug("End Taking Action\n")

    def _compute_reward(self):
        """
        Compute reward and done based on current status
        Return:
            reward
            done
        """
        rospy.logdebug("\nStart Computing Reward")
        reward, done = 0, False
        if self.status.count('escaped')==2:
            reward = 400.
            done = True
            rospy.logerr("\n!!!!!!!!!!!!!!!!\nLogger Escaped !\n!!!!!!!!!!!!!!!!")
        elif self.status.count('trapped')==1 and self.status.count('escaped')==1:
            reward = -0.
            done = False
            rospy.logdebug("\n1 in 1 out\n")
        elif self.status.count('trapped')==2:
            reward = -0.1
            done = False
            rospy.logdebug("\nLogger is trapped\n")
        elif 'blown' in self.status:
            reward = -1 # not necessary
            done = True
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

    def _link_states_callback(self, data):
        self.link_states = data


if __name__ == "__main__":
    env = DoubleEscapeContinuousEnv()
    num_episodes = 4
    num_steps = env.max_steps
    ep_cntr = 0
    while ep_cntr < num_episodes:
        obs = env.reset()
        rospy.logdebug("obs: {}".format(obs))
        if 'blown' in env.status:
            obs = env.reset()
            continue
        for st in range(num_steps):
            act = random.randn(env.action_space[0]) * env.action_scale
            obs, rew, done, info = env.step(act)
            rospy.loginfo("\n-\nepisode: {}, step: {} \nobs: {}, \nact: {} \nreward: {} \ndone: {} \ninfo: {}".format(ep_cntr+1, st+1, obs, act, rew, done, info))
            if done:
                if st > 5:
                    ep_cntr+=1
                break
