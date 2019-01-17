from __future__ import print_function

import numpy as np
import time
import random
import rospy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, LinkStates

if __name__ == "__main__":
  cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
  rospy.init_node("logger_test" , anonymous=True, log_level=rospy.DEBUG)
  cmd_vel = Twist()
  num_step = 100
  rate = rospy.Rate(10)
  
  for st in range(num_step):
    lin = random.uniform(-1,1)
    ang = random.uniform(-np.pi,np.pi)
    cmd_vel.linear.x = lin
    cmd_vel.angular.z = ang
    rospy.logdebug("step: {}, command velocity: {}".format(st, (lin,ang)))
    cmd_pub.publish(cmd_vel)
    rate.sleep()
