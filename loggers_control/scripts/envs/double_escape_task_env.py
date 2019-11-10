#!/usr/bin/env python
"""
Task environment for two loggers escaping from the walled cell, cooperatively.
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


class DoubleEscapeEnv(object):
    """
    DoubleEscape Class
    """
    def __init__(self):
        rospy.init_node("double_escape_task_env", anonymous=True, log_level=rospy.INFO)
        # init simulation parameters
        self.rate = rospy.Rate(100)
        # init environment parameters
        self.observation = dict(
            log=dict(
                pose=Pose(),
                twist=Twist()),
            logger_0=dict(
                pose=Pose(),
                twist=Twist()),
            logger_1=dict(
                pose=Pose(),
                twist=Twist())
        )
        self.action_0 = np.zeros(2)
        self.action_1 = np.zeros(2)
        self.info = dict(status="")
        self.reward = 0
        self._episode_done = False
        self.success_count = 0
        self.max_step = 2000
        self.steps = 0
        self.status = "trapped"
        self.model_states = ModelStates()
        self.link_states = LinkStates()
        # services
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        # topic publisher
        self.cmdvel0_pub = rospy.Publisher("/cmd_vel_0", Twist, queue_size=1)
        self.cmdvel1_pub = rospy.Publisher("/cmd_vel_1", Twist, queue_size=1)
        self.set_model_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.set_link_state_pub = rospy.Publisher("/gazebo/set_link_state", LinkState, queue_size=1)
        # topic subscriber
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback)
        rospy.Subscriber("/gazebo/link_states", LinkStates, self._link_states_callback)

    def pauseSim(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause_proxy()
        except rospy.ServiceException as e:
            rospy.logfatal("/gazebo/pause_physics service call failed")

    def unpauseSim(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_proxy()
        except rospy.ServiceException as e:
            rospy.logfatal("/gazebo/unpause_physics service call failed")

    def reset(self, init_pose=[0,0,0,0,0]):
        """
        Reset environment function
        obs, info = env.reset()
        """
        rospy.logdebug("\nStart Environment Reset")
        self._take_action(np.zeros(2), np.zeros(2))
        self.reset_world()
        self._set_init(init_pose)
        self.pauseSim()
        self.unpauseSim()
        obs = self._get_observation()
        info = self._post_information()
        self.steps = 0
        rospy.logwarn("\nEnd Environment Reset!!!\n")

        return obs, info

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

    def _set_init(self, init_pose):
        """
        Set initial condition of two_loggers to a specific pose
        Args:
            init_pose: [rod{x, y, angle}, robot_0{th_0}, robot_1{th_1}], set to a random pose if empty
        """
        rospy.logdebug("\nStart Initializing Robots")
        model_state = ModelState()
        model_state.model_name = "two_loggers"
        model_state.reference_frame = "world"
        model_state.pose.position.z = 0.25
        if not init_pose:
            # random initialize
            pass
        else:
            model_state.pose.position.x = init_pose[0]
            model_state.pose.position.y = init_pose[1]
            model_state.pose.orientation.z = math.sin(0.5*init_pose[2])
            model_state.pose.orientation.w = math.cos(0.5*init_pose[2])
            # give the system a little time to set model
            for _ in range(10):
                self.set_model_state_pub.publish(model_state)
                self.rate.sleep()
            self._take_action(np.zeros(2), np.zeros(2)) # stablize model setting
            # compute link_state for logger_0 and logger_1
            link_states = self.link_states
            link_name_0 = "two_loggers::link_chassis_0"
            link_name_1 = "two_loggers::link_chassis_1"
            link_state_0 = LinkState()
            link_state_0.link_name = link_name_0
            link_state_0.pose = link_states.pose[link_states.name.index(link_name_0)]
            link_state_0.pose.orientation.z = math.sin(0.5*init_pose[3])
            link_state_0.pose.orientation.w = math.cos(0.5*init_pose[3])
            link_state_1 = LinkState()
            link_state_1.link_name = link_name_1
            link_state_1.pose = link_states.pose[link_states.name.index(link_name_1)]
            link_state_1.pose.orientation.z = math.sin(0.5*init_pose[4])
            link_state_1.pose.orientation.w = math.cos(0.5*init_pose[4])
            # set orientation for both logger_0 and logger_1
            for _ in range(10):
                self.set_link_state_pub.publish(link_state_0)
                self.set_link_state_pub.publish(link_state_1)
                self.rate.sleep()
        # stablize link setting
        self._take_action(np.zeros(2), np.zeros(2))
        rospy.logdebug("\ntwo_loggers initialized at {} \nlogger_0 orientation: {} \nlogger_1 orientation".format(model_state, init_pose[3], init_pose[4]))
        # episode should not be done
        self._episode_done = False
        rospy.logdebug("End Initializing Robots\n")

    def _get_observation(self):
        """
        Get observation from link_states
        Return:
            observation: {"log{"pose", "twist"}", logger0{"pose", "twist"}", logger1{"pose", "twist"}"}
        """
        rospy.logdebug("\nStart Getting Observation")
        link_states = self.link_states
        # the log
        id_log = link_states.name.index("two_loggers::link_log")
        self.observation["log"]["pose"] = link_states.pose[id_log]
        self.observation["log"]["twist"] = link_states.twist[id_log]
        # logger_0
        id_logger_0 = link_states.name.index("two_loggers::link_chassis_0")
        self.observation["logger_0"]["pose"] = link_states.pose[id_logger_0]
        self.observation["logger_0"]["twist"] = link_states.twist[id_logger_0]
        # logger_1
        id_logger_1 = link_states.name.index("two_loggers::link_chassis_1")
        self.observation["logger_1"]["pose"] = link_states.pose[id_logger_1]
        self.observation["logger_1"]["twist"] = link_states.twist[id_logger_1]
        # env status
        # compute status
        if self.observation["logger_0"]["pose"].position.x > 4.79 or self.observation["logger_1"]["pose"].position.x > 4.79:
            self.status = "east"
        elif self.observation["logger_0"]["pose"].position.x < -4.79 or self.observation["logger_1"]["pose"].position.x < -4.79:
            self.status = "west"
        elif self.observation["logger_0"]["pose"].position.y > 4.79 or self.observation["logger_1"]["pose"].position.y > 4.79:
            self.status = "north"
        elif -6<=self.observation["logger_0"]["pose"].position.y < -4.79:
            if np.absolute(self.observation["logger_0"]["pose"].position.x) > 1:
                self.status = "south"
            else:
                if np.absolute(self.observation["logger_0"]["pose"].position.x) > 0.79:
                    self.status = "sdoor" # stuck at door
                else:
                    self.status = "tdoor" # through door
        elif -6<=self.observation["logger_1"]["pose"].position.y < -4.79:
            if np.absolute(self.observation["logger_1"]["pose"].position.x) > 1:
                self.status = "south"
            else:
                if np.absolute(self.observation["logger_1"]["pose"].position.x) > 0.79:
                    self.status = "sdoor" # stuck at door
                else:
                    self.status = "tdoor" # through door
        elif self.observation["logger_0"]["pose"].position.y < -6 and self.observation["logger_1"]["pose"].position.y < -6:
            self.status = "escaped"
        elif self.observation["logger_0"]["pose"].position.z > 0.95 or self.observation["logger_1"]["pose"].position.z > 0.95 or self.observation["logger_0"]["pose"].position.z < 0.85 or self.observation["logger_1"]["pose"].position.z < 0.85:
            self.status = "blew"
        else:
            self.status = "trapped"
        # logging
        rospy.logdebug("Observation Get ==> {}".format(self.observation))
        rospy.logdebug("End Getting Observation\n")

        return self.observation

    def _take_action(self, action_0, action_1):
        """
        Set linear and angular speed for logger_0 and logger_1 to execute.
        Args:
            action: 2x np.array([v_lin,v_ang]).
        """
        rospy.logdebug("\nStart Taking Actions")
        cmd_vel_0 = Twist()
        cmd_vel_0.linear.x = action_0[0]
        cmd_vel_0.angular.z = action_0[1]
        cmd_vel_1 = Twist()
        cmd_vel_1.linear.x = action_1[0]
        cmd_vel_1.angular.z = action_1[1]
        for _ in range(10):
            self.cmdvel0_pub.publish(cmd_vel_0)
            self.cmdvel1_pub.publish(cmd_vel_1)
            self.rate.sleep()
        self.action_0 = action_0
        self.action_1 = action_1
        rospy.logdebug("\nlogger_0 take action ===> {}\nlogger_1 take action ===> {}".format(cmd_vel_0, cmd_vel_1))
        rospy.logdebug("End Taking Actions\n")

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
            rospy.logerr("\nDouble Escape Succeed!\n")
        else:
            self.reward = -0.
            self._episode_done = False
            rospy.loginfo("The log is trapped in the cell...")
        rospy.logdebug("Stepwise Reward: {}, Success Count: {}".format(self.reward, self.success_count))
        # check if steps out of range
        if self.steps > self.max_step:
            self._episode_done = True
            rospy.logwarn("Step: {}, \nMax step reached, env will reset...".format(self.steps))
        rospy.logdebug("End Computing Reward\n")

        return self.reward, self._episode_done

    def _post_information(self):
        """
        Return:
            info: {"system status"}
        """
        rospy.logdebug("\nStart Posting Information")
        self.info["status"] = self.status
        rospy.logdebug("End Posting Information\n")

        return self.info

    def _model_states_callback(self, data):
        self.model_states = data

    def _link_states_callback(self, data):
        self.link_states = data
