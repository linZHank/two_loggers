#! /usr/bin/env python
"""
Version: 2019-04-07
Model free on policy control for logger robot with vanilla policy gradient in wall-celled environment
Navigate to escape from the only exit
Author: LinZHanK (linzhank@gmail.com)
Inspired by: https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py
"""
from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(0, "/home/linzhank/ros_ws/src/two_loggers/loggers_control/scripts/envs")
sys.path.insert(0, "/home/linzhank/ros_ws/src/two_loggers/loggers_control/scripts/utils")
import argparse
import numpy as np
import tensorflow as tf
import rospy
import random
import os
import time
from datetime import datetime

from solo_escape_task_env import SoloEscapeEnv
import gen_utils, solo_utils, tf_utils
from gen_utils import bcolors


VERSION="2019-04-07" # make sure this is same as on line #3

def train(env, model_path,
          dim_state=7, num_actions=3, actions=np.zeros((2,2)),
          hidden_sizes=[64], learning_rate=1e-3,
          num_epochs=1000, batch_size=1e4,
          wall_bonus_flag=False, door_bonus_flag=False, dist_bonus_flag=False):
    # make core of policy network
    states_ph = tf.placeholder(shape=(None, dim_state), dtype=tf.float32)
    logits = tf_utils.mlp(states_ph, sizes=hidden_sizes+[num_actions])
    # make action selection op (outputs int actions, sampled from policy)
    act_id = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
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
    def train_one_epoch():
        # make some empty lists for logging.
        batch_states = [] # for observations
        batch_actions = [] # for actions
        batch_rtaus = [] # for R(tau) weighting in policy gradient
        batch_returns = [] # for measuring episode returns
        batch_lengths = [] # for measuring episode lengths
        # reset episode-specific variables
        obs, _, = env.reset() # first obs comes from starting distribution
        done, ep_rewards = False, []
        state = solo_utils.obs_to_state(obs)
        dist_0 = np.linalg.norm(state[:2]-np.array([0,-6.0001]))
        episode = 1
        step = 1
        while True:
            # save state
            batch_states.append(state.copy())
            # take action in env
            act_i = sess.run(act_id, {states_ph: state.reshape(1,-1)})[0]
            action = actions[act_i]
            obs, rew, done, info = env.step(action)
            state = solo_utils.obs_to_state(obs)
            # compute current distance to exit, and distance change
            dist = np.linalg.norm(state[:2]-np.array([0,-6.0001]))
            delta_dist = dist_0 - dist
            # adjust reward based on relative distance to the exit
            rew, done = solo_utils.adjust_reward(rew, info, delta_dist, done,
                                           wall_bonus_flag, door_bonus_flag, dist_bonus_flag)
            # save action_id, reward
            batch_actions.append(act_i)
            ep_rewards.append(rew)
            # update previous robot's distance to exit
            dist_0 = dist
            # log this step
            rospy.loginfo("Episode: {}, Step: {} \naction: {}, state: {}, reward: {}, status: {}".format(
                episode,
                step,
                action,
                state,
                rew,
                info["status"]
            ))
            step += 1
            if done:
                # if episode is over, calculate R(tau)
                ep_return, ep_length = sum(ep_rewards), len(ep_rewards)
                batch_returns.append(ep_return)
                batch_lengths.append(ep_length)
                # R(tau) is the weight of log(pi(a|s))
                batch_rtaus += [ep_return] * ep_length
                # reset
                obs, _ = env.reset()
                done, ep_rewards = False, []
                state = solo_utils.obs_to_state(obs)
                episode += 1
                step = 1
                print(
                    bcolors.OKGREEN,
                    "batch_size limit: {}, current batch_lengths: {}".format(
                        batch_size,
                        len(batch_states)
                    ), bcolors.ENDC
                )
                # end policy sampling if batch size reached
                if len(batch_states) > batch_size:
                    break

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
    num_episodes = 0
    for epoch in range(num_epochs):
        batch_loss, batch_returns, batch_lengths = train_one_epoch()
        episodic_returns += batch_returns
        print("epoch: {:d} \t episode: {:d} \t loss: {:.3f} \t return: {:.3f}\t ep_len: {}".format(
        epoch+1,
        len(episodic_returns),
        batch_loss,
        np.mean(batch_returns),
        np.mean(batch_lengths)
        ))
        save_path = saver.save(sess, model_path)
        rospy.loginfo("Model saved in path : {}".format(save_path))
        rospy.logerr("Success Count: {}".format(env.success_count))
    # plot returns and save figure
    gen_utils.plot_returns(returns=episodic_returns, mode=2, save_flag=True, path=model_path)

if __name__ == "__main__":
    # make arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
    default="/home/linzhank/ros_ws/src/two_loggers/loggers_control/vpg_model-"+datetime.now().strftime("%Y-%m-%d-%H-%M")+"/model.ckpt")
    parser.add_argument("--hidden_sizes", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1e4)
    parser.add_argument("--wall_bonus_flag", type=bool, default=False)
    parser.add_argument("--door_bonus_flag", type=bool, default=False)
    parser.add_argument("--dist_bonus_flag", type=bool, default=False)
    args = parser.parse_args()

    # Main really starts here
    start_time = time.time()
    rospy.init_node("solo_escape_vpg", anonymous=True, log_level=rospy.INFO)
    # make an instance from env class
    env = SoloEscapeEnv()
    env.reset()
    dim_state = len(solo_utils.obs_to_state(env.observation))
    actions = np.array([np.array([.5, -1]), np.array([.5, 1])])
    num_actions = len(actions)

    # train
    train(
        env=env, model_path=args.model_path,
        dim_state=dim_state, num_actions=num_actions, actions=actions,
        hidden_sizes=[args.hidden_sizes], learning_rate=args.learning_rate,
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        wall_bonus_flag=args.wall_bonus_flag, door_bonus_flag=args.door_bonus_flag, dist_bonus_flag=args.dist_bonus_flag
    )
    # time
    end_time = time.time()
    training_time = end_time - start_time

    # Main actually ends here
    # store hyper parameters
    hyp_params = {
        "statespace_dim": dim_state,
        "num_actions": num_actions,
        "hidden_sizes": args.hidden_sizes,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size
    }
    # store training information
    train_info = hyp_params
    train_info["version"]=VERSION
    train_info["success_count"] = env.success_count
    train_info["training_time"] = training_time
    train_info["wall_bonus_flag"] = args.wall_bonus_flag
    train_info["door_bonus_flag"] = args.door_bonus_flag
    train_info["dist_bonus_flag"] = args.dist_bonus_flag

    # save hyper-parameters
    gen_utils.save_pkl(content=hyp_params, path=args.model_path, fname="hyper_parameters.pkl")
    # save results
    gen_utils.save_csv(content=train_info, path=args.model_path, fname="results.csv")
