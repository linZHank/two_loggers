#!/usr/bin/env python
"""
Task environment for single logger escaping from the walled cell
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import pi
from numpy import random
import time

import rospy
import tf
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, LinkState, ModelStates, LinkStates
from geometry_msgs.msg import Pose, Twist


class SoloEscapeEnv(object):
    """
    SoloEscape Class
    """
    def __init__(self):
        rospy.init_node("solo_escape_task_env", anonymous=True, log_level=rospy.INFO)
        # simulation parameters
        self.rate = rospy.Rate(100)
        # environment parameters
        self.observation = dict(
            pose=Pose(),
            twist=Twist()
        )
        self.action = np.zeros(2)
        self.info = dict(status="")
        self.reward = 0
        self._episode_done = False
        self.success_count = 0
        self.max_step = 1000
        self.steps = 0
        self.status = "trapped"
        self.model_states = ModelStates()
        # services
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # topic publisher
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        # topic subscriber
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback)

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

    def resetSimulation(self):
        rospy.wait_for_service("/gazebo/reset_simulation")
        try:
            self.reset_simulation_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_simulation service call failed")

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

    def reset(self, init_pose=[]):
        """
        Reset environment
        obs, info = env.reset()
        """
        rospy.logdebug("\nStart Environment Reset")
        self._set_init(init_pose)
        obs = self._get_observation()
        info = self._post_information()
        self.steps = 0
        rospy.logerr("\nEnvironment Reset!!!\n")

        return obs, info

    def step(self, action):
        """
        Manipulate the environment with an action
        obs, rew, done, info = env.step(action)
        """
        rospy.logdebug("\nStart Environment Step")
        self._take_action(action)
        obs = self._get_observation()
        reward, done = self._compute_reward()
        info = self._post_information()
        self.steps += 1
        rospy.logdebug("End Environment Step\n")

        return obs, reward, done, info

    def _set_init(self, init_pose):
        """
        Set initial condition for single logger, Set the logger at a random pose inside cell.
        Args:
            init_pose: [x, y, theta]
        """
        rospy.logdebug("\nStart Initializing Robot")
        # prepare
        self._take_action(np.zeros(2))
        self.pausePhysics()
        self.resetWorld()
        robot_pose = ModelState()
        robot_pose.model_name = "logger"
        robot_pose.reference_frame = "world"
        robot_pose.pose.position.z = 0.2
        if init_pose: # inialize randomly
            assert np.absolute(init_pose[0]) <= 4.5
            assert np.absolute(init_pose[1]) <= 4.5
            assert -pi<=init_pose[2]<= pi # theta within [-pi,pi]
        else: # inialize accordingly
            init_pose = [0]*3
            init_pose[0] = random.uniform(-4.5, 4.5)
            init_pose[1] = random.uniform(-4.5, 4.5)
            init_pose[2] = random.uniform(-pi, pi)
        robot_pose.pose.position.x = init_pose[0]
        robot_pose.pose.position.y = init_pose[1]
        quat = tf.transformations.quaternion_from_euler(0, 0, init_pose[2])
        robot_pose.pose.orientation.z = quat[2]
        robot_pose.pose.orientation.w = quat[3]
        # call '/gazebo/set_model_state' service
        self.setModelState(model_state=robot_pose)
        rospy.logdebug("Logger was initialized at {}".format(robot_pose))
        self.unpausePhysics()
        self._take_action(np.zeros(2))
        # episode should not be done
        self._episode_done = False
        rospy.logdebug("End Initializing Robot\n")

    def _get_observation(self):
        """
        Get observations from env
        Return:
        observation: [x, y, v_x, v_y, cos(yaw), sin(yaw), yaw_dot]
        """
        rospy.logdebug("\nStart Getting Observation")
        model_states = self._get_model_states()
        id_logger = model_states.name.index("logger")
        self.observation["pose"] = model_states.pose[id_logger]
        self.observation["twist"] = model_states.twist[id_logger]
        # compute status
        if self.observation["pose"].position.x > 4.79:
            self.status = "east"
        elif self.observation["pose"].position.x < -4.79:
            self.status = "west"
        elif self.observation["pose"].position.y > 4.79:
            self.status = "north"
        elif -6 <= self.observation["pose"].position.y < -4.79:
            if np.absolute(self.observation["pose"].position.x) > 1:
                self.status = "south"
            else:
                if np.absolute(self.observation["pose"].position.x) > 0.79:
                    self.status = "sdoor" # stuck at door
                else:
                    self.status = "tdoor" # through door
        elif self.observation["pose"].position.y < -6:
            self.status = "escaped"
        elif self.observation['pose'].position.z > 0.25 or self.observation['pose'].position.z < 0.15:
            self.status = "blew"
        else:
            self.status = "trapped"
        rospy.logdebug("Observation Get ==> {}".format(self.observation))
        rospy.logdebug("End Getting Observation\n")

        return self.observation

    def _take_action(self, action):
        """
        Set linear and angular speed for logger to execute.
        Args:
            action: np.array([v_lin, v_ang]).
        """
        rospy.logdebug("\nStart Taking Action")
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]
        self.cmd_vel_pub.publish(cmd_vel)
        for _ in range(15): # 6.667Hz
            self.cmd_vel_pub.publish(cmd_vel)
            self.rate.sleep()
        rospy.logdebug("Logger take action ==> {}".format(cmd_vel))
        rospy.logdebug("End Taking Action\n")

    def _compute_reward(self):
        """
        Return:
            reward: reward in current step
        """
        rospy.logdebug("\nStart Computing Reward")
        if self.status == "escaped":
            self.reward = 1
            self.success_count += 1
            self._episode_done = True
            rospy.logerr("\n!!!\nLogger Escaped !\n!!!")
        else:
            self.reward = -0.
            self._episode_done = False
            rospy.loginfo("\nLogger is trapped\n!!!")
        rospy.logdebug("Stepwise Reward: {}, success count : {}".format(self.reward, self.success_count))
        # check if steps out of range
        if self.steps > self.max_step:
            self._episode_done = True
            rospy.logwarn("Step: {}, \nMax step reached, env will reset...".format(self.steps))
        rospy.logdebug("End Computing Reward\n")

        return self.reward, self._episode_done

    def _post_information(self):
        """
        Return:
            info: {"status"}
        """
        rospy.logdebug("\nStart Posting Information")
        self.info["status"] = self.status
        rospy.logdebug("End Posting Information\n")

        return self.info

    def _model_states_callback(self, data):
        self.model_states = data

    def _get_model_states(self):
        return self.model_states
