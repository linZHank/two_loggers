from __future__ import print_function

import numpy as np
import time
import random
import rospy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, LinkStates
from std_srvs.srv import Empty


if __name__ == "__main__":
  reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
  cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
  rospy.init_node("logger_test" , anonymous=True, log_level=rospy.DEBUG)
  cmd_vel = Twist()
  num_episodes = 10
  num_steps = 100
  rate = rospy.Rate(10)

  for ep in range(num_episodes):
    reset_world()
    for st in range(num_steps):
      lin = random.uniform(-1,1)
      ang = random.uniform(-np.pi,np.pi)
      cmd_vel.linear.x = lin
      cmd_vel.angular.z = ang
      rospy.logdebug("step: {}, command velocity: {}".format(st, (lin,ang)))
      cmd_pub.publish(cmd_vel)
      rate.sleep()
