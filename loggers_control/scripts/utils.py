#! /usr/bin/env python
"""
Useful class and functions for loggers tasks
"""
import numpy as np
import csv
import pickle
import tensorflow as tf
import os
from datetime import datetime
import pickle
import csv
import matplotlib.pyplot as plt

class bcolors:
    """ For the purpose of print in terminal with colors """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# bonus functions 0301
# def wallBonusDividedNumsteps(num_steps): return -1./num_steps # bonus for every time step
# def weightedD0(weight,d0,amplifier): return weight*d0*amplifier # bonus for initial distance
# def d0MinusD(d0,d,num_steps): return (d0-d)/num_steps # bonus of approaching the exit
# bonus functions 0303
def bonus_func(n_steps): return 1./n_steps # basic bonus

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
  # Build a feedforward neural network.
  for size in sizes[:-1]:
    x = tf.layers.dense(x, units=size, activation=activation)
  return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

# save pickle
def save_pkl(fname, path, content):
  """
  Save content into path/name as pickle file
  Args:
    fname: file name, str
    path: file path, str
    content: to be saved, array/dict/list...
  """
  fdir = os.path.dirname(path)
  file_path = os.path.join(fdir,fname)
  with open(file_path, "wb") as f:
    pickle.dump(content, f, pickle.HIGHEST_PROTOCOL)

def save_csv(fname, path, content):
  """
  Save content into path/name as csv file
  Args:
    fname: file name, str
    path: file path, str
    content: to be saved, dict
  """
  fdir = os.path.dirname(path)
  file_path = os.path.join(fdir,fname)
  with open(file_path, "w") as f:
    for key in content.keys():
      f.write("{},{}\n".format(key,content[key]))

def plot_returns(returns, mode, save_flag, path):
  """
  Plot rewards
  Args:
    returns: episodic returns, list
    mode: 0 - returns; 1 - accumulated returns; 2 - averaged returns, int
    save_flag: save figure to file or not, bool
    path: file path, str
  """
  assert mode==0 or mode==1 or mode==2
  acc_returns = []
  ave_returns = []
  acc_r = returns[0]
  ave_r = acc_r
  for i,r in enumerate(returns):
    acc_r += r
    ave_r = acc_r/(i+1)
    acc_returns.append(acc_r)
    ave_returns.append(ave_r)
  fig, ax = plt.subplots()
  if mode == 0:
    ax.plot(np.arange(len(returns)), returns)
    ax.set(xlabel="Episode", ylabel="Returns")
    figure_dir = os.path.join(os.path.dirname(path),"episodic_returns.png")
  elif mode == 1:
    ax.plot(np.arange(len(acc_returns)), acc_returns)
    ax.set(xlabel="Episode", ylabel="Accumulated Returns")
    figure_dir = os.path.join(os.path.dirname(path),"accumulated_returns.png")
  else:
    ax.plot(np.arange(len(ave_returns)), ave_returns)
    ax.set(xlabel="Episode", ylabel="Averaged Returns")
    figure_dir = os.path.join(os.path.dirname(path),"averaged_returns.png")
  ax.grid()
  if save_flag:
    plt.savefig(figure_dir)
  else:
    plt.show()
  plt.close(fig)

def obs_to_state(observation):
    pass
      # x = model_states.pose[-1].position.x
      # y = model_states.pose[-1].position.y
      # v_x = model_states.twist[-1].linear.x
      # v_y = model_states.twist[-1].linear.y
      # quat = (
      #   model_states.pose[-1].orientation.x,
      #   model_states.pose[-1].orientation.y,
      #   model_states.pose[-1].orientation.z,
      #   model_states.pose[-1].orientation.w
      # )
      # euler = tf.transformations.euler_from_quaternion(quat)
      # cos_yaw = math.cos(euler[2])
      # sin_yaw = math.sin(euler[2])
      # yaw_dot = model_states.twist[-1].angular.z

def judge_robot(observation):
    pass
    # if self.curr_pose[0] > 4.79:
    #     reward = -0.
    #     self.status = "east"
    #     self._episode_done = True
    #     rospy.logwarn("Logger is too close to east wall!")
    # elif self.curr_pose[0] < -4.79:
    #     reward = -0.
    #     self.status = "west"
    #     self._episode_done = True
    #     rospy.logwarn("Logger is too close to west wall!")
    # elif self.curr_pose[1] > 4.79:
    #     reward = -0.
    #     self.status = "north"
    #     self._episode_done = True
    #     rospy.logwarn("Logger is too close to north wall!")
    # elif self.curr_pose[1]<=-4.79 and np.absolute(self.curr_pose[0])>1 :
    #     reward = -0.
    #     self.status = "south"
    #     self._episode_done = True
    #     rospy.logwarn("Logger is too close to south wall!")
    # elif -6<self.curr_pose[1]<-4.79 and np.absolute(self.curr_pose[0])>0.79:
    #     reward = 0.
    #     self.status = "door"
    #     self._episode_done = True
    #     rospy.logwarn("Logger is stuck at the door!")
