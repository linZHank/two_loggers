#!/usr/bin/env python
"""
Task environment for single logger escaping form the walled cell
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import math
import random
import time
import rospy
import tf
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState, LinkState, ModelStates, LinkStates
from geometry_msgs.msg import Pose, Twist


class SoloEscapeEnv(object):
    """
    SoloEscape Class
    """
    def __init__(self):
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
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # topic publisher
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.set_robot_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        # topic subscriber
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback)

    # def pauseSim(self):
    #   rospy.wait_for_service("/gazebo/pause_physics")
    #   try:
    #     self.pause()
    #   except rospy.ServiceException as e:
    #     rospy.logfatal("/gazebo/pause_physics service call failed")
    # def unpauseSim(self):
    #   rospy.wait_for_service("/gazebo/unpause_physics")
    #   try:
    #     self.unpause()
    #   except rospy.ServiceException as e:
    #     rospy.logfatal("/gazebo/unpause_physics service call failed")

    def reset(self):
      """
      Reset environment
      obs, info = env.reset()
      """
      rospy.logdebug("\nStart Environment Reset")
      self._take_action(np.zeros(2))
      self.reset_world()
      self._set_init()
      obs = self._get_observation()
      info = self._post_information()
      self.steps = 0
      rospy.logdebug("End Environment Reset\n")
      rospy.logwarn("\nEnvironment Reset!!!\n")

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

    def _set_init(self):
        """
        Set initial condition for simulation
        Set Logger at a random pose inside cell
        Returns:
        init_position: array([x, y])
        """
        rospy.logdebug("\nStart Initializing Robot")
        # set logger initial pose, using pole coordinate
        mag = random.uniform(0, 4.5) # robot vector magnitude
        ang = random.uniform(-math.pi, math.pi) # robot vector orientation
        x = mag * math.cos(ang)
        y = mag * math.sin(ang)
        w = random.uniform(-1.0, 1.0)
        theta = tf.transformations.euler_from_quaternion([0,0,math.sqrt(1-w**2),w])[2]
        robot_state = ModelState()
        robot_state.model_name = "logger"
        robot_state.pose.position.x = x
        robot_state.pose.position.y = y
        robot_state.pose.position.z = 0.2
        robot_state.pose.orientation.x = 0
        robot_state.pose.orientation.y = 0
        robot_state.pose.orientation.z = math.sqrt(1 - w**2)
        robot_state.pose.orientation.w = w
        robot_state.reference_frame = "world"
        # Give the system a little time to finish initialization
        for _ in range(8):
            self.set_robot_state_pub.publish(robot_state)
            self.rate.sleep()
        self._take_action(np.zeros(2))
        rospy.logwarn("Robot was initialized at {}".format(robot_state))
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
      for _ in range(10):
          self.cmd_vel_pub.publish(cmd_vel)
          self.rate.sleep()
      self.action = action
      rospy.logdebug("Action Taken ===> {}".format(cmd_vel))
      rospy.logdebug("End Taking Action\n")

    def _compute_reward(self):
        """
        Return:
          reward: reward in current step
        """
        rospy.logdebug("\nStart Computing Reward")
        # status
        if self.observation["pose"].position.y < -6:
            self.status = "escaped"
            self.reward = 1
            self.success_count += 1
            self._episode_done = True
            rospy.logerr("\n!!!\nLogger Escaped !\n!!!")
        else:
            self.status = "trapped"
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
            info: {"status": "where the robot at"}
        """
        rospy.logdebug("\nStart Posting Information")
        if self.observation["pose"].position.x > 4.79:
            self.status = "east"
        elif self.observation["pose"].position.x < -4.79:
            self.status = "west"
        elif self.observation["pose"].position.y > 4.79:
            self.status = "north"
        elif self.observation["pose"].position.y < -4.79:
            if np.absolute(self.observation["pose"].position.x) > 1:
                self.status = "south"
            else:
                if np.absolute(self.observation["pose"].position.x) > 0.79:
                    self.status = "sdoor" # stuck at door
                else:
                    self.status = "tdoor" # through door
        elif self.observation["pose"].position.y < -6:
            self.status = "escaped"
        else:
            self.status = "trapped"
        self.info["status"] = self.status
        rospy.logdebug("End Posting Information\n")

        return self.info

    def _model_states_callback(self, data):
        self.model_states = data

    def _get_model_states(self):
        return self.model_states
