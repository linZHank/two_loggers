"""
An implementation of Deep Q-network (DQN) for solo_escape_task
DQN is a Model free, off policy, reinforcement learning algorithm (https://deepmind.com/research/dqn/)
Author: LinZHanK (linzhank@gmail.com)
"""
from __future__ import absolute_import, division, print_function

import sys
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import rospy

from envs.solo_escape_task_env import SoloEscapeEnv
from utils import gen_utils, solo_utils, tf_utils
from utils.gen_utils import bcolors
from tensorflow.keras.layers import Dense

if __name__ == "__main__":
    # Main really starts here
    # start_time = time.time()
    rospy.init_node("solo_escape_dqn_test", anonymous=True, log_level=rospy.INFO)
    model_path = os.path.dirname(sys.path[0])+"/dqn_model/2019-04-30-22-17/model.ckpt"    # make an instance from env class
    env = SoloEscapeEnv()
    env.reset()
    # hyper-parameters
    hyp_param_path = os.path.join(os.path.dirname(model_path),"hyper_parameters.pkl")
    with open(hyp_param_path, "rb") as f:
        hyp_params = pickle.load(f)
    dim_state = hyp_params["dim_state"]
    actions = hyp_params["actions"]
    num_episodes = 10
    num_steps = 200
    # qnet model
    qnet = tf.keras.models.Sequential([
        Dense(64, input_shape=(dim_state, ), activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(actions))
    ])
    qnet.load_weights(model_path)
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        state = solo_utils.obs_to_state(obs)
        for st in range(num_steps):
            # pick an action
            act_id = np.argmax(qnet.predict(state.reshape(1,-1)))
            action = actions[act_id]
            obs, _, done,
            # take the action
            obs, rew, done, info = env.step(action)
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
