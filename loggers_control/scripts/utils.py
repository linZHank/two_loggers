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
    
# bonus functions
def wallBonusDividedNumsteps(bonus_wall, num_steps): return bonus_wall/num_steps # bonus for every time step
def weightedD0(weight,d0,amplifier): return weight*d0*amplifier # bonus for initial distance
def d0MinusD(d0,d,num_steps): return (d0-d)/num_steps # bonus of approaching the exit
def zero(x,y,z=0): return 0

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
