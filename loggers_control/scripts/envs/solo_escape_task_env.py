#!/usr/bin/env python


""" Configure a cart-pole system spawned in Gazebo to be a qualified environment for reinforcement learning 
    Based on cartpole-v0, but increases pole swaying angle limit and modifies reward mechanism"""


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


class SoloEscapeEnv(object):
  """ Task environment for a single logger escape from a walled cell
  """
  def __init__(self):
    # init simulation parameters
    self.rate = rospy.Rate(100)
    # init environment parameters
    self.observation = np.array([0., 0., 0., 0., 1., 0., 0.]) # x,y,v_x,v_y,cos_theta,sin_theta, theta_dot
    self.action = np.zeros(2)
    self.reward = 0
    self._episode_done = False
    self.success_count = 0
    # init env info
    self.init_pose = np.zeros(3) # x, y, theta
    self.prev_pose = np.zeros(3)
    self.curr_pose = np.zeros(3)
    # init services
    self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    # init topic publisher
    self.cmd_vel_pub = rospy.Publisher(
      "/logger/chassis_drive_controller/cmd_vel",
      Twist,
      queue_size=10
    )
    self.set_robot_state_pub = rospy.Publisher(
      "/gazebo/set_model_state",
      ModelState,
      queue_size=100
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
          
  def env_reset(self):
    """ 
    obs, info = env.reset() 
    """
    rospy.logwarn("\nEnvironment Reset!!!\n")
    self.reset_world()
    self._set_init()
    obs = self._get_observation()
    info = self._post_information()
    rospy.logdebug("Environment Reset Finished")

    return obs, info

  def env_step(self, action):
    """
    Manipulate the environment with an action
    """
    self._take_action(action)
    obs = self._get_observation()
    reward, done = self._compute_reward()
    info = self._post_information()

    return obs, reward, done, info

  def get_model_states(self):
    return self.model_states
  
  def _set_init(self):
    """ 
    Set initial condition for simulation
      Set Logger at a random pose inside cell by publishing /gazebo/set_model_state topic
    Returns: 
      init_position: array([x, y]) 
    """
    rospy.logdebug("Start initializing robot....")
    # set logger inside crib, away from crib edges
    mag = random.uniform(0, 4) # robot vector magnitude
    ang = random.uniform(-math.pi, math.pi) # robot vector orientation
    x = mag * math.cos(ang)
    y = mag * math.sin(ang)
    w = random.uniform(-1.0, 1.0)
    theta = tf.transformations.euler_from_quaternion([0,0,math.sqrt(1-w**2),w])[2]
    robot_state = ModelState()
    robot_state.model_name = "logger"
    robot_state.pose.position.x = x
    robot_state.pose.position.y = y
    robot_state.pose.position.z = 0.09
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
    rospy.logwarn("Robot was set at {}".format(self.init_pose))
    # Episode cannot done
    self._episode_done = False
    # Give the system a little time to finish initialization
    rospy.logdebug("Logger Initialized @ ===> {}".format(robot_state))

  def _take_action(self, action):
    """
    Set linear and angular speed for logger to execute.
    Args:
      action: 2-d numpy array.
    """
    rospy.logdebug("Start Taking Action....")
    self.action = action
    cmd_vel = Twist()
    cmd_vel.linear.x = action[0]
    cmd_vel.angular.z = action[1]
    for _ in range(10):
      self.cmd_vel_pub.publish(cmd_vel)
      rospy.logdebug("Action Taken ===> {}".format(cmd_vel))
      self.rate.sleep()
    
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
      self._episode_done = True
      rospy.logerr("\n!!!\nLogger Escaped !\n!!!")
    elif self.curr_pose[0] > 4.75:
      reward = 0
      self._episode_done = True
      rospy.logwarn("Logger is too close to east wall, task will be reset!")
    elif self.curr_pose[0] < -4.75:
      reward = 0
      self._episode_done = True
      rospy.logwarn("Logger is too close to west wall, task will be reset!")
    elif self.curr_pose[1] > 4.75:
      reward = 0
      self._episode_done = True
      rospy.logwarn("Logger is too close to north wall, task will be reset!")
    elif self.curr_pose[1] < -4.75 and np.absolute(self.curr_pose[0])>1:
      reward = 0
      self._episode_done = True
      rospy.logwarn("Logger is too close to south wall, task will be reset!")
    elif self.curr_pose[1] < -5 and np.absolute(self.curr_pose[0])>0.75:
      reward = 0
      self._episode_done = True
      rospy.logwarn("Logger is too close to door, task will be reset!")
    else:
      reward = 0
      self._episode_done = False
      rospy.loginfo("Logger is working on its way to escape...")
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
      "previous_pose": self.prev_pose
    }
    rospy.logdebug("Information Posted ===> {}".format(self.info))
    
    return self.info

  def _model_states_callback(self, data):
    self.model_states = data
