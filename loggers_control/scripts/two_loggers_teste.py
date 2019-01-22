#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import random
import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

if __name__ == "__main__":
  reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
  reset_simulation()
  rospy.logwarn("Simulation Reset")
  cmd1_pub = rospy.Publisher("cmd_vel_1", Twist, queue_size=10)
  cmd2_pub = rospy.Publisher("cmd_vel_2", Twist, queue_size=10)
  rospy.init_node("loggers_test", anonymous=True, log_level=rospy.DEBUG)
  cmd_vel_1 = Twist()
  cmd_vel_2 = Twist()
  num_step = 256
  rate = rospy.Rate(10)
  
  for st in range(num_step):
    lin1 = random.uniform(-1,1)
    ang1 = random.uniform(-np.pi,np.pi)
    lin2 = random.uniform(-1,1)
    ang2 = random.uniform(-np.pi,np.pi)
    cmd_vel_1.linear.x = lin1
    cmd_vel_1.angular.z = ang1
    cmd_vel_2.linear.x = lin2
    cmd_vel_2.angular.z = ang2
    rospy.logdebug("step: {}, \ncommand velocity 1: {}, \ncommand velocity 2: {}".format(st, (lin1,ang1), (lin2,ang2)))
    cmd1_pub.publish(cmd_vel_1)
    cmd2_pub.publish(cmd_vel_2)
    rate.sleep()
