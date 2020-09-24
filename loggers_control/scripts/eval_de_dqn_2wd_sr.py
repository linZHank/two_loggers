#! /usr/bin/env python
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import time
import rospy

from envs.de_2wd import DoubleEscape2WD
from agents.dqn import DeepQNet


if __name__=='__main__':
    env = DoubleEscape2WD()
    dim_obs = env.observation_space_shape[1]*2
    num_act_0 = env.action_reservoir_0.shape[0]
    num_act_1 = env.action_reservoir_1.shape[0]
    agent_0 = DeepQNet(
        dim_obs=dim_obs,
        num_act=num_act_0
    )
    agent_1 = DeepQNet(
        dim_obs=dim_obs,
        num_act=num_act_1
    )
    # load models
    # q_net_path_0 = './saved_models/double_escape_2WD/dqn/?'
    # q_net_path_1 = './saved_models/double_escape_2WD/dqn/?'
    # agent_0.q.q_net = tf.keras.models.load_model(q_net_path_0)
    # agent_1.q.q_net = tf.keras.models.load_model(q_net_path_1)
    agent_0.epsilon = 0
    agent_1.epsilon = 0
    num_episodes = 10
    ep = 0
    success_counter = 0
    o, d, t = env.reset(), False, 0
    while ep < num_episodes: 
        while 'blown' in env.status: 
            o, d, t = env.reset(), False, 0
        s0 = o[[0,-1]].flatten()
        s1 = o[[1,-1]].flatten()
        # a0 = np.squeeze(agent_0.act(np.expand_dims(s0, axis=0)))
        # a1 = np.squeeze(agent_1.act(np.expand_dims(s1, axis=0)))
        a0 = np.random.randint(0,4)
        a1 = np.random.randint(0,2)
        n_o, r, d, i = env.step(np.array([int(a0), int(a1)]))
        n_s0 = n_o[[0,-1]].flatten()
        n_s1 = n_o[[1,-1]].flatten()
        t += 1
        o = n_o.copy()
        if d or (t==env.max_episode_steps):
            ep +=1
            if i.count('escaped')==2:
                success_counter += 1
            o, d, t = env.reset(), False, 0


