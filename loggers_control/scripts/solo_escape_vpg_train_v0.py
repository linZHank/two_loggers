#! /usr/bin/env python
"""
Version: 2019-03-03
Model based control for turtlebot with vanilla policy gradient in crib environment
Navigate towards preset goal
Author: LinZHanK (linzhank@gmail.com)
Inspired by: https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py
"""
from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(0, "/home/linzhank/ros_ws/src/two_loggers/loggers_control/scripts/envs")
import argparse
import numpy as np
import tensorflow as tf
import rospy
import random
import os
import time
from datetime import datetime

from solo_escape_task_env import SoloEscapeEnv
import utils

VERSION="2019-03-04" # make sure this is same as on line #3


def train(agent, model_path,
          dim_state=7, num_actions=3,
          hidden_sizes=[64], learning_rate=1e-3,
          num_episodes=400, num_steps=1000,
          wall_bonus=False, door_bonus=False, distance_bonus=False):
  # make core of policy network
  states_ph = tf.placeholder(shape=(None, dim_state), dtype=tf.float32)
  logits = utils.mlp(states_ph, sizes=hidden_sizes+[num_actions])
  # make action selection op (outputs int actions, sampled from policy)
  actions_id = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
  # make loss function whose gradient, for the right data, is policy gradient
  rtaus_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
  actid_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
  action_masks = tf.one_hot(actid_ph, num_actions)
  log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
  loss = -tf.reduce_mean(rtaus_ph * log_probs)
  # make train op
  train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
  # start a session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  
  # for training policy
  def train_one_episode():
    # make some empty lists for logging.
    batch_states = [] # for observations
    batch_actions = [] # for actions
    batch_rtaus = [] # for R(tau) weighting in policy gradient
    batch_returns = [] # for measuring episode returns
    batch_lengths = [] # for measuring episode lengths
    # reset episode-specific variables
    state, _ = agent.env_reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rewards = []            # list for rewards accrued throughout ep
    dist_0 = np.linalg.norm(state[:2]-np.array([0,-6.02]))
    for st in range(num_steps):
      # save obs
      batch_states.append(state.copy())
      # act in the environment
      action_id = sess.run(actions_id, {states_ph: state.reshape(1,-1)})[0]
      if action_id == 0: # forward left
        action = np.array([.5, 1.])
      elif action_id == 1: # forward right
        action = np.array([.5, -1.])
      else: # forward
        action = np.array([.5, 0.])
        rospy.logerr("Moving forward")
      state, rew, done, info = agent.env_step(action)
      # compute current distance to exit, and distance change
      dist = np.linalg.norm(state[:2]-np.array([0,-6.2]))
      delta_dist = dist_0 - dist
      # adjust reward based on relative distance to the exit
      bonus=0
      if info["status"] == "escaped":
        bonus = rew
      elif info["status"] == "door":
        if door_bonus:
          bonus = utils.bonus_func(num_steps) # small positive bonus if stuck at door
      elif info["status"] == "trapped":
        if distance_bonus:
          if delta_dist < 0:
            bonus = -utils.bonus_func(num_steps) # negtive, if getting further from exit
          else:
            bonus = utils.bonus_func(num_steps)/10. # positive, if getting closer to exit
      else:
        if wall_bonus:
          bonus = -utils.bonus_func(num_steps)*10
      rew += bonus
      # save action, reward
      batch_actions.append(action_id)
      ep_rewards.append(rew)
      # update previous robot's distance to exit
      dist_0 = dist
      # log this step
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
    # if episode is over, record info about episode
    ep_return, ep_length = sum(ep_rewards), len(ep_rewards)
    batch_returns.append(ep_return)
    batch_lengths.append(ep_length)
    # the weight for every logprob(a|s) is R(tau)
    batch_rtaus += [ep_return] * ep_length
    # take a single policy gradient update step
    batch_loss, _ = sess.run([loss, train_op],
                             feed_dict={
                               states_ph: np.array(batch_states),
                               actid_ph: np.array(batch_actions),
                               rtaus_ph: np.array(batch_rtaus)
                             })
    return batch_loss, batch_returns, batch_lengths
  
  # training loop
  episodic_returns = []
  for ep in range(num_episodes):
    batch_loss, batch_returns, batch_lengths = train_one_episode()
    print("epoch: {} \t loss: {} \t return: {}\t ep_len: {}".format(ep, batch_loss, batch_returns[0], np.mean(batch_lengths)))
    episodic_returns.append(batch_returns[0])
    save_path = saver.save(sess, model_path)
    rospy.loginfo("Model saved in path : {}".format(save_path))
    rospy.logerr("Success Count: {}".format(agent.success_count))
  # plot returns and save figure
  utils.plot_returns(returns=episodic_returns, mode=2, save_flag=True, path=model_path)  

if __name__ == "__main__":
  # make arg parser
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str,
                      default="/home/linzhank/ros_ws/src/two_loggers/loggers_control/vpg_model-"+datetime.now().strftime("%Y-%m-%d-%H-%M")+"/model.ckpt")
  parser.add_argument("--hidden_sizes", type=int, default=64)
  parser.add_argument("--learning_rate", type=float, default=1e-3)
  parser.add_argument("--num_episodes", type=int, default=400)
  parser.add_argument("--num_steps", type=int, default=1000)
  parser.add_argument("--wall_bonus", type=bool, default=False)
  parser.add_argument("--door_bonus", type=bool, default=False)
  parser.add_argument("--distance_bonus", type=bool, default=False)
  args = parser.parse_args()

  # Main really starts here
  start_time = time.time()
  rospy.init_node("solo_escape_vpg", anonymous=True, log_level=rospy.INFO)
  # make an instance from env class
  escaper = SoloEscapeEnv()
  statespace_dim = 7 # x, y, x_dot, y_dot, cos_theta, sin_theta, theta_dot
  actionspace_dim = 2
  # train
  train(
    agent=escaper, model_path=args.model_path,
    dim_state=statespace_dim, num_actions=actionspace_dim,
    hidden_sizes=[args.hidden_sizes], learning_rate=args.learning_rate,
    num_episodes=args.num_episodes, num_steps=args.num_steps,
    wall_bonus=args.wall_bonus, door_bonus=args.door_bonus, distance_bonus=args.distance_bonus
  )
  # time
  end_time = time.time()
  training_time = end_time - start_time

  # Main actually ends here
  # store hyper parameters
  hyp_params = {
    "statespace_dim": statespace_dim,
    "actionspace_dim": actionspace_dim,
    "hidden_sizes": args.hidden_sizes,
    "learning_rate": args.learning_rate,
    "num_episodes": args.num_episodes,
    "num_steps": args.num_steps,
  }               
  # store training information
  train_info = hyp_params
  train_info["version"]=VERSION
  train_info["success_count"] = escaper.success_count
  train_info["training_time"] = training_time
  train_info["wall_bonus"] = args.wall_bonus
  train_info["door_bonus"] = args.door_bonus
  train_info["distance_bonus"] = args.distance_bonus

  # save hyper-parameters
  utils.save_pkl(fname="hyper_parameters.pkl",
             path=args.model_path,
             content=hyp_params)
  # save results
  utils.save_csv(fname="results.csv",
                 path=args.model_path,
                 content=train_info)
  
