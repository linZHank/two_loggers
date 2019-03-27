#!/usr/bin/env python

""" 
Task environment for two loggers escaping from the walled cell
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import math
import random
import time
import rospy
import tf
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose, Twist


class DoubleEscapeEnv(object):
  """ Task environment for a single logger escape from a walled cell
  """
  def __init__(self):
    # init simulation parameters
    self.rate = rospy.Rate(100)
    # init environment parameters
    self.observation = np.array([0., 0., 0., 0., 1., 0., 0.]) # x,y,v_x,v_y,cos_theta,sin_theta, theta_dot
    self.action_0 = np.zeros(2)
    self.action_1 = np.zeros(2)
    self.reward = 0
    self._episode_done = False
    self.success_count = 0
    self.max_step = 2000
    self.step = 0
    # init env info
    self.init_pose = np.zeros(3) # x, y, theta
    self.prev_pose = np.zeros(3)
    self.curr_pose = np.zeros(3)
    self.status = "trapped"
    # init services
    self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    # init topic publisher
    self.cmdvel0_pub = rospy.Publisher(
      "/cmd_vel_0",
      Twist,
      queue_size=1
    ) 
    self.cmdvel1_pub = rospy.Publisher(
      "/cmd_vel_1",
      Twist,
      queue_size=1
    )
    self.set_robot_state_pub = rospy.Publisher(
      "/gazebo/set_model_state",
      ModelState,
      queue_size=10
    )
    # init topic subscriber
    rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback)

  def pauseSim(self):
    rospy.wait_for_service("/gazebo/pause_physics")
    try:
      self.pause()
    except rospy.ServiceException as e:
      rospy.logfatal("/gazebo/pause_physics service call failed")
        
  def unpauseSim(self):
    rospy.wait_for_service("/gazebo/unpause_physics")
    try:
      self.unpause()
    except rospy.ServiceException as e:
      rospy.logfatal("/gazebo/unpause_physics service call failed")
          
  def reset(self):
    """
    reset environment
    obs, info = env.reset() 
    """
    rospy.logwarn("\nEnvironment Reset!!!\n")
    self._take_action(np.zeros(2), np.zeros(2))
    self.reset_world()
    self._set_init()
    obs = self._get_observation()
    info = self._post_information()
    self.step = 0
    rospy.logdebug("Environment Reset Finished")

    return obs, info

  def step(self, action_0, action_1):
    """
    Manipulate logger_0 with action_0, logger_1 with action_1
    obs, rew, done, info = env.step(action_0, action_1)
    """
    self._take_action(action_0, actions_1)
    obs = self._get_observation()
    reward, done = self._compute_reward()
    info = self._post_information()
    self.step += 1

    return obs, reward, done, info
  
  def _set_init(self):
    """ 
    Set initial condition for two_loggers at a random pose inside cell by publishing 
    "/gazebo/set_model_state" topic.
    Returns: 
      init_position: array([x, y]) 
    """
    rospy.logdebug("Start initializing robot....")
    # set logger inside crib, away from crib edges
    mag = random.uniform(0, 3.2) # robot vector magnitude
    ang = random.uniform(-math.pi, math.pi) # robot vector orientation
    x = mag * math.cos(ang)
    y = mag * math.sin(ang)
    w = random.uniform(-1.0, 1.0)
    theta = tf.transformations.euler_from_quaternion([0,0,math.sqrt(1-w**2),w])[2]
    robot_state = ModelState()
    robot_state.model_name = "two_loggers"
    robot_state.pose.position.x = x
    robot_state.pose.position.y = y
    robot_state.pose.position.z = 0.2
    robot_state.pose.orientation.x = 0
    robot_state.pose.orientation.y = 0
    robot_state.pose.orientation.z = math.sqrt(1 - w**2)
    robot_state.pose.orientation.w = w
    robot_state.reference_frame = "world"  
    self.init_pose = np.array([x, y, theta])
    self.curr_pose = self.init_pose
    # set goal point using pole coordinate
    for _ in range(10):
      self.set_robot_state_pub.publish(robot_state)
      self.rate.sleep()
    rospy.logwarn("two_loggers were set at {}".format(self.init_pose))
    # Episode cannot done
    self._episode_done = False
    # Give the system a little time to finish initialization
    rospy.logdebug("Logger Initialized @ ===> {}".format(robot_state))

  def _take_action(self, action_0, action_1):
    """
    Set linear and angular speed for logger_0 and logger_1 to execute.
    Args:
      action: 2x (v_lin,v_ang).
    """
    rospy.logdebug("Start Taking Action....")
    self.action_0 = action_0
    self.action_1 = action_1
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
    rospy.logdebug("\nlogger_0 take action ===> {}\nlogger_1 take action ===> {}".format(cmd_vel_0, cmd_vel_1))
    
  def _get_observation(self):
    """
    Get observations from env
    Return:
      observation: [x, y, v_x, v_y, cos(yaw), sin(yaw), yaw_dot]
    """
    rospy.logdebug("Start Getting Observation....")
    self.prev_pose = self.curr_pose
    model_states = self.get_model_states() # refer to turtlebot_robot_env
    # update previous position
    rospy.logdebug("model_states: {}".format(model_states))
    x = model_states.pose[-1].position.x # turtlebot was the last model in model_states
    y = model_states.pose[-1].position.y
    v_x = model_states.twist[-1].linear.x
    v_y = model_states.twist[-1].linear.y
    quat = (
      model_states.pose[-1].orientation.x,
      model_states.pose[-1].orientation.y,
      model_states.pose[-1].orientation.z,
      model_states.pose[-1].orientation.w
    )
    euler = tf.transformations.euler_from_quaternion(quat)
    cos_yaw = math.cos(euler[2])
    sin_yaw = math.sin(euler[2])
    yaw_dot = model_states.twist[-1].angular.z
    self.curr_pose = np.array([x, y, np.arctan2(sin_yaw,cos_yaw)])
    self.observation = np.array([x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot])
    rospy.logdebug("Observation Get ==> {}".format(self.observation))
    
    return self.observation
    
  def _compute_reward(self):
    """
    Return:
      reward: reward in current step
    """
    rospy.logdebug("Start Computing Reward....")
    if self.curr_pose[1] < -6:
      reward = 1
      self.success_count += 1
      self.status = "escaped"
      self._episode_done = True
      rospy.logerr("\n!!!\nLogger Escaped !\n!!!")
    elif self.curr_pose[0] > 4.75:
      reward = -0.
      self.status = "east"
      self._episode_done = True
      rospy.logwarn("Logger is too close to east wall!")
    elif self.curr_pose[0] < -4.75:
      reward = -0.
      self.status = "west"
      self._episode_done = True
      rospy.logwarn("Logger is too close to west wall!")
    elif self.curr_pose[1] > 4.75:
      reward = -0.
      self.status = "north"
      self._episode_done = True
      rospy.logwarn("Logger is too close to north wall!")
    elif -4.99<self.curr_pose[1]<-4.75 and np.absolute(self.curr_pose[0])>1:
      reward = -0.
      self.status = "south"
      self._episode_done = True
      rospy.logwarn("Logger is too close to south wall!")
    elif self.curr_pose[1] < -5 and np.absolute(self.curr_pose[0])>0.75:
      reward = 0.
      self.status = "door"
      self._episode_done = True
      rospy.logwarn("Logger is stuck at the door!")
    else:
      reward = -0.
      self.status = "trapped"
      self._episode_done = False
      rospy.loginfo("Logger is trapped in the cell...")
    self.reward = reward
    rospy.logdebug("Stepwise Reward Computed ===> {}".format(reward))
    
    return self.reward, self._episode_done 

  def _post_information(self):
    """
    Return:
      info: {"init_pose", "curr_pose", "prev_pose"}
    """
    rospy.logdebug("Start posting information of the task")
    self.info = {
      "initial_pose": self.init_pose,
      "current_pose": self.curr_pose,
      "previous_pose": self.prev_pose,
      "status": self.status
    }
    rospy.logdebug("Information Posted ===> {}".format(self.info))
    
    return self.info

  def _model_states_callback(self, data):
    self.model_states = data

  def _get_model_states(self):
    return self.model_states
