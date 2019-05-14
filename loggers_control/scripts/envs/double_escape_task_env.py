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
        # init services
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # init topic publisher
        self.cmdvel0_pub = rospy.Publisher("/cmd_vel_0", Twist, queue_size=1)
        self.cmdvel1_pub = rospy.Publisher("/cmd_vel_1", Twist, queue_size=1)
        self.set_model_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        # self.set_link_state_pub = rospy.Publisher("/gazebo/set_link_state", LinkState, queue_size=10)
        # init topic subscriber
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback)
        rospy.Subscriber("/gazebo/link_states", LinkStates, self._link_states_callback)

    # def pauseSim(self):
    #     rospy.wait_for_service("/gazebo/pause_physics")
    #     try:
    #         self.pause()
    #     except rospy.ServiceException as e:
    #         rospy.logfatal("/gazebo/pause_physics service call failed")
    # def unpauseSim(self):
    #     rospy.wait_for_service("/gazebo/unpause_physics")
    #     try:
    #         self.unpause()
    #     except rospy.ServiceException as e:
    #         rospy.logfatal("/gazebo/unpause_physics service call failed")

    def reset(self):
        """
        Reset environment function
        obs, info = env.reset()
        """
        rospy.logdebug("\nStart Environment Reset")
        self._take_action(np.zeros(2), np.zeros(2))
        self.reset_world()
        self._set_init()
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

    def _set_init(self):
        """
        Set initial condition for two_loggers at a random pose
        """
        rospy.logdebug("\nStart Initializing Robots")
        # set model initial pose using pole coordinate
        mag = random.uniform(0, 3.6) # model vector magnitude
        ang = random.uniform(-math.pi, math.pi) # model vector orientation
        x = mag * math.cos(ang)
        y = mag * math.sin(ang)
        w = random.uniform(-1.0, 1.0)
        theta = tf.transformations.euler_from_quaternion([0,0,math.sqrt(1-w**2),w])[2]
        model_state = ModelState()
        model_state.model_name = "two_loggers"
        model_state.pose.position.x = x
        model_state.pose.position.y = y
        model_state.pose.position.z = 0.25
        model_state.pose.orientation.x = 0
        model_state.pose.orientation.y = 0
        model_state.pose.orientation.z = math.sqrt(1 - w**2)
        model_state.pose.orientation.w = w
        model_state.reference_frame = "world"
        # set orientations for logger_0 and logger_1, by spinning them a little
        spin_vel_0 = random.choice([-2*np.pi, 2*np.pi])
        spin_vel_1 = random.choice([-2*np.pi, 2*np.pi])
        # give the system a little time to finish initialization
        for _ in range(10):
            self.set_model_state_pub.publish(model_state)
            self.rate.sleep()
        # spin robots randomly then stop 'em
        self._take_action(np.array([0,spin_vel_0]), np.array([0,spin_vel_1]))
        self._take_action(np.zeros(2), np.zeros(2)) # stop spinning
        rospy.logwarn("two_loggers were initialized at {}".format(model_state))
        # episode should not be done
        self._episode_done = False
        rospy.logdebug("End Initializing Robots\n")

    def _get_observation(self):
        """
        Get observation from link_states
        Return:
            observation: {"log{"pose", "twist"}", logger0{"pose", "twist"}", logger1{"pose", "twist"}"}
        """
        # model states
        rospy.logdebug("\nStart Getting Observation")
        link_states = self._get_link_states()
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
        elif -6<=self.observation["logger_0"]["pose"].position.y < -4.79 or -6<=self.observation["logger_1"]["pose"].position.y < -4.79:
            if np.absolute(self.observation["logger_0"]["pose"].position.x) > 1 or np.absolute(self.observation["logger_1"]["pose"].position.x) > 1:
                self.status = "south"
            else:
                if np.absolute(self.observation["logger_0"]["pose"].position.x) > 0.79 or np.absolute(self.observation["logger_1"]["pose"].position.x) > 0.79:
                    self.status = "sdoor" # stuck at door
                else:
                    self.status = "tdoor" # through door
        elif self.observation["logger_0"]["pose"].position.y < -6 and self.observation["logger_1"]["pose"].position.y < -6:
            self.status = "escaped"
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
            action: 2 x np.array([v_lin,v_ang]).
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
            info: {"init_pose", "curr_pose", "prev_pose"}
        """
        rospy.logdebug("\nStart Posting Information")
        self.info["status"] = self.status
        rospy.logdebug("End Posting Information\n")

        return self.info

    def _model_states_callback(self, data):
        self.model_states = data

    def _link_states_callback(self, data):
        self.link_states = data

    def _get_model_states(self):
        return self.model_states

    def _get_link_states(self):
        return self.link_states
