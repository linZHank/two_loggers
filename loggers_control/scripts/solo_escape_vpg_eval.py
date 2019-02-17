#! /usr/bin/env python

"""
Model based control for turtlebot with vanilla policy gradient in crib environment
Navigate towards preset goal
Author: LinZHanK (linzhank@gmail.com)
Inspired by: https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py
"""
from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(0, "/home/linzhank/ros_ws/src/two_loggers/loggers_control/scripts/envs")

import numpy as np
import tensorflow as tf
import rospy
import random
import os
import time
import datetime
import matplotlib.pyplot as plt

from solo_escape_task_env import SoloEscapeEnv
from utils import bcolors


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
  # Build a feedforward neural network.
  for size in sizes[:-1]:
    x = tf.layers.dense(x, units=size, activation=activation)
  return tf.layers.dense(x, units=sizes[-1], activation=output_activation)  

if __name__ == "__main__":
  # parameters
  dim_state = 7
  num_actions = 3
  hidden_sizes = [64]
  num_episodes = 10
  num_steps = 1024
  # set tf 
  states_ph = tf.placeholder(shape=(None, dim_state), dtype=tf.float32)
  logits = mlp(states_ph, sizes=hidden_sizes+[num_actions])
  actions_id = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
  saver = tf.train.Saver()
  model_path = "/home/linzhank/ros_ws/src/two_loggers/loggers_control/vpg_model-2019-02-15-16-27/model.ckpt"
  rospy.init_node("solo_escape_eval", anonymous=True, log_level=rospy.INFO)
  # make an instance from env class
  escaper = SoloEscapeEnv()
  # Create a tf session
  sess = tf.Session()
  saver.restore(sess, model_path)
  # start ecaluation
  for ep in range(num_episodes):
    state, _ = escaper.env_reset()
    done = False
    rew = 0
    ep_rewards = []
    for st in range(num_steps):
      action_id = sess.run(actions_id, feed_dict={states_ph: state.reshape(1,-1)})[0]
      if action_id == 0: # go straight
        action = np.array([.5, 0])
      elif action_id == 1: # turn left
        action = np.array([0, 1.0])
      else: # turn right
        action = np.array([0, -1.0])
      state, rew, done, info = escaper.env_step(action)
      # deposit reward
      ep_rewards.append(rew)
      rospy.loginfo("Episode: {}, Step: {} \naction: {}, state: {}, reward: {}, done: {}".format(
        ep,
        st,
        action,
        state,
        rew,
        done
      ))
      if done:
        break

