#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import math
import random
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState, LinkState
from std_srvs.srv import Empty

if __name__ == "__main__":
  reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
  reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
  cmd_pub_0 = rospy.Publisher("/cmd_vel_0", Twist, queue_size=1)
  cmd_pub_1 = rospy.Publisher("/cmd_vel_1", Twist, queue_size=1)
  set_log_state_publisher = rospy.Publisher("/gazebo/set_link_state", LinkState, queue_size=100)
  set_robot_state_publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=100)
  rospy.init_node("two_loggers_test" , anonymous=True, log_level=rospy.DEBUG)
  num_episodes = 10
  num_steps = 10
  rate = rospy.Rate(10)

  for ep in range(num_episodes):
    reset_world()
    mag = random.uniform(0, 2) # robot vector magnitude
    ang = random.uniform(-math.pi, math.pi) # robot vector orientation
    x = mag * math.cos(ang)
    y = mag * math.sin(ang)
    w = random.uniform(-1.0, 1.0)    
    log_state = LinkState()
    log_state.link_name = "two_loggers::link_log"
    log_state.pose.position.x = x
    log_state.pose.position.y = y
    log_state.pose.position.z = 0.2
    log_state.pose.orientation.x = 0
    log_state.pose.orientation.y = 0
    # log_state.pose.orientation.z = 0
    # log_state.pose.orientation.w = 1
    log_state.pose.orientation.z = math.sqrt(1 - w**2)
    log_state.pose.orientation.w = w
    log_state.reference_frame = "world"
    rate = rospy.Rate(10)
    for _ in range(num_steps):
      set_log_state_publisher.publish(log_state)
      rate.sleep()
    # for st in range(num_steps):
    #   lin_0 = random.uniform(-.4,.4)
    #   ang_0 = random.uniform(-np.pi/6,np.pi/6)
    #   cmd_vel_0.linear.x = lin_0
    #   cmd_vel_0.angular.z = ang_0
    #   lin_1 = random.uniform(-.4,.4)
    #   ang_1 = random.uniform(-np.pi/6,np.pi/6)
    #   cmd_vel_1.linear.x = lin_1
    #   cmd_vel_1.angular.z = ang_1
    #   rospy.logdebug("step: {}, \ncommand velocity #0: {}, \ncommand velocity #1: {}".format(st, (lin_0,ang_0),(lin_1,ang_1)))
    #   cmd_pub_0.publish(cmd_vel_0)
    #   cmd_pub_1.publish(cmd_vel_1)
    #   rate.sleep()