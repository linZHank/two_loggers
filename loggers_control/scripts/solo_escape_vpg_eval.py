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
import pickle
import matplotlib.pyplot as plt

from solo_escape_task_env import SoloEscapeEnv
import utils


if __name__ == "__main__":
  # identify saved model path
  model_path = "/home/linzhank/ros_ws/src/two_loggers/loggers_control/vpg_model-2019-03-08-09-53/model.ckpt"
  # load hyper-parameters
  hyp_param_path = os.path.join(os.path.dirname(model_path),"hyper_parameters.pkl")
  with open(hyp_param_path, "rb") as f:
    hyp_param = pickle.load(f)
  dim_state = hyp_param["statespace_dim"]
  num_actions = hyp_param["actionspace_dim"]
  hidden_sizes = hyp_param["hidden_sizes"]
  # num_spisodes and num_steps are different from training
  num_episodes = 10
  num_steps = 1024
  # set tf 
  states_ph = tf.placeholder(shape=(None, dim_state), dtype=tf.float32)
  logits = utils.mlp(states_ph, sizes=[hidden_sizes]+[num_actions])
  actions_id = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
  saver = tf.train.Saver()
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
      # pick an action
      action_id = sess.run(actions_id, {states_ph: state.reshape(1,-1)})[0]
      if action_id == 0: # forward left
        action = np.array([.5, 1.])
      elif action_id == 1: # forward right
        action = np.array([.5, -1.])
      else: # forward
        action = np.array([.5, 0.])
        rospy.logerr("Moving forward")
      # take the action
      state, rew, done, info = escaper.env_step(action)
      # deposit reward
      ep_rewards.append(rew)
      rospy.loginfo("Episode: {}, Step: {} \naction: {}, state: {}, reward: {}, done: {}".format(
        ep+1,
        st+1,
        action,
        state,
        rew,
        done
      ))
      if done:
        break

