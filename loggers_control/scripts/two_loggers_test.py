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
    cmd0_pub = rospy.Publisher("/cmd_vel_0", Twist, queue_size=1)
    cmd1_pub = rospy.Publisher("/cmd_vel_1", Twist, queue_size=1)
    set_robot_state_publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=100)
    rospy.init_node("two_loggers_test" , anonymous=True, log_level=rospy.DEBUG)
    cmd_vel_0 = Twist()
    cmd_vel_1 = Twist()
    num_episodes = 10
    num_steps = 200
    rate = rospy.Rate(10)

    for ep in range(num_episodes):
        mag = random.uniform(0, 3.6) # robot vector magnitude
        ang = random.uniform(-math.pi, math.pi) # robot vector orientation
        x = mag * math.cos(ang)
        y = mag * math.sin(ang)
        w = random.uniform(-1.0, 1.0)
        robot_state = ModelState()
        robot_state.model_name = "two_loggers"
        robot_state.pose.position.x = x
        robot_state.pose.position.y = y
        robot_state.pose.position.z = 0.25
        robot_state.pose.orientation.x = 0
        robot_state.pose.orientation.y = 0
        robot_state.pose.orientation.z = math.sqrt(1 - w**2)
        robot_state.pose.orientation.w = w
        robot_state.reference_frame = "world"
        rate = rospy.Rate(100)
        # stop motion
        cmd0_pub.publish(Twist())
        cmd1_pub.publish(Twist())
        for _ in range(4):
            reset_world()
            set_robot_state_publisher.publish(robot_state)
            rate.sleep()
        for st in range(num_steps):
            lin0 = random.uniform(-1,1)
            ang0 = random.uniform(-np.pi,np.pi)
            lin1 = random.uniform(-1,1)
            ang1 = random.uniform(-np.pi,np.pi)
            cmd_vel_0.linear.x = lin0
            cmd_vel_0.angular.z = ang0
            cmd_vel_1.linear.x = lin1
            cmd_vel_1.angular.z = ang1
            rospy.logdebug("episode: {}, step: {}, cmd_1: {}, cmd_2".format(ep, st, (lin0,ang0), (lin1,ang1)))
            cmd0_pub.publish(cmd_vel_0)
            cmd1_pub.publish(cmd_vel_1)
            rate.sleep()
    cmd0_pub.publish(Twist())
    cmd1_pub.publish(Twist())
    print("robot stopped")
    for _ in range(4):
        reset_world()
        rate.sleep()
