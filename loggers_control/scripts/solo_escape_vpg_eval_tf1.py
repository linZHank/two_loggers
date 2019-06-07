#! /usr/bin/env python

"""
Vanilla Policy Gradient evaluation for single logger robot's solo escape task
"""
from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import tensorflow as tf
import rospy
import random
import os
import time
import datetime
import pickle
import matplotlib.pyplot as plt

from envs.solo_escape_task_env import SoloEscapeEnv
from utils import data_utils, solo_utils, tf_utils


if __name__ == "__main__":
    # identify saved model path
    model_path = os.path.dirname(sys.path[0])+"/vpg_model/2019-03-14-17-52/model.ckpt"
    # load hyper-parameters
    hyp_param_path = os.path.join(os.path.dirname(model_path),"hyper_parameters.pkl")
    with open(hyp_param_path, "rb") as f:
        hyp_param = pickle.load(f)
    dim_state = hyp_param["statespace_dim"]
    num_actions = hyp_param["actionspace_dim"]
    hidden_sizes = hyp_param["hidden_sizes"]
    # num_spisodes and num_steps are different from training
    num_episodes = 10
    num_steps = 200
    # set tf
    states_ph = tf.placeholder(shape=(None, dim_state), dtype=tf.float32)
    logits = tf_utils.mlp(states_ph, sizes=[hidden_sizes]+[num_actions])
    act_id = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
    saver = tf.train.Saver()
    rospy.init_node("solo_escape_eval", anonymous=True, log_level=rospy.INFO)
    # make an instance from env class
    escaper = SoloEscapeEnv()
    # Create a tf session
    sess = tf.Session()
    saver.restore(sess, model_path)
    # start ecaluation
    for ep in range(num_episodes):
        obs, _ = escaper.reset()
        done = False
        state = solo_utils.obs_to_state(obs)
        for st in range(num_steps):
            # pick an action
            act_i = sess.run(act_id, {states_ph: state.reshape(1,-1)})[0]
            if act_i == 0: # forward left
                action = np.array([.5, 1.])
            elif act_i == 1: # forward right
                action = np.array([.5, -1.])
            else: # forward
                action = np.array([.5, 0.])
                rospy.logerr("Moving forward")
            # take the action
            obs, _, done, info = escaper.step(action)
            state = solo_utils.obs_to_state(obs)
            # logging
            rospy.loginfo("Episode: {}, Step: {} \naction: {}, state: {}, done: {}".format(
                ep+1,
                st+1,
                action,
                state,
                done
            ))
            if done:
                break
