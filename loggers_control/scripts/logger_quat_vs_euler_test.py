from __future__ import print_function

import numpy as np
import tf
import time
import random
import matplotlib.pyplot as plt
import rospy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, LinkStates

class QuatEulerTest():
  def __init__(self):
    self.quat_angle = 0
    self.euler_angle = 0
    rospy.Subscriber("/gazebo/model_states", ModelStates, self.ms_callback)
    self._cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    
  def ms_callback(self, data):
    model_states = data
    quat = (
    model_states.pose[-1].orientation.x,
    model_states.pose[-1].orientation.y,
    model_states.pose[-1].orientation.z,
    model_states.pose[-1].orientation.w
  )
    euler = tf.transformations.euler_from_quaternion(quat)
    half_theta = np.arctan2(quat[2], quat[3])
    self.quat_angle = 2*half_theta
    # if self.quat_angle > np.pi or self.quat_angle < -np.pi:
    #   self.quat_angle -= 2*np.pi
    self.euler_angle = np.arctan2(np.sin(euler[2]), np.cos(euler[2]))
  
if __name__ == "__main__":
  rospy.init_node("logger_test", anonymous=True, log_level=rospy.DEBUG)
  cmd_vel = Twist()
  num_step = 400
  rate = rospy.Rate(10)
  qet = QuatEulerTest()

  x = np.linspace(0, num_step, num_step)
  plt.figure()
  for st in range(num_step):
    ang = -np.pi/20
    cmd_vel.angular.z = ang
    qet._cmd_vel_pub.publish(cmd_vel)
    rospy.logdebug("step: {}, command velocity: {}".format(st, ang))
    
    print("quat: {}, euler: {}".format(qet.quat_angle, qet.euler_angle))
    plt.scatter(st, qet.quat_angle, color="r")
    plt.scatter(st, qet.euler_angle, color="g")
    rate.sleep()
  plt.show()    
  
