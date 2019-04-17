#! /usr/bin/env python
from __future__ import absolute_import, print_function

import numpy as np
import math
import time
import random
import rospy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty


if __name__ == "__main__":
  reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
  # cmd_pub = rospy.Publisher("/logger/chassis_drive_controller/cmd_vel", Twist, queue_size=10)
  cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
  set_robot_state_publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=100)
  rospy.init_node("logger_test" , anonymous=True, log_level=rospy.DEBUG)
  cmd_vel = Twist()
  num_episodes = 10
  num_steps = 200
  rate = rospy.Rate(10)

  for ep in range(num_episodes):
    for _ in range(4):
      reset_world()
      rate.sleep()
    mag = random.uniform(0, 4) # robot vector magnitude
    ang = random.uniform(-math.pi, math.pi) # robot vector orientation
    x = mag * math.cos(ang)
    y = mag * math.sin(ang)
    w = random.uniform(-1.0, 1.0)    
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
    rate = rospy.Rate(100)
    for _ in range(4):
      set_robot_state_publisher.publish(robot_state)
      rate.sleep()
    for st in range(num_steps):
      lin = random.uniform(-2,2)
      ang = random.uniform(-np.pi,np.pi)
      cmd_vel.linear.x = lin
      cmd_vel.angular.z = ang
      rospy.logdebug("episode: {}, step: {}, command velocity: {}".format(ep, st, (lin,ang)))
      cmd_pub.publish(cmd_vel)
      rate.sleep()
  cmd_vel.linear.x = 0
  cmd_vel.angular.z = 0
  cmd_pub.publish(cmd_vel)
  for _ in range(4):
    reset_world()
    rate.sleep()
  print("robot stopped")
  
