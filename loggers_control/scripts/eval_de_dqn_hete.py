#! /usr/bin/env python
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import time
import rospy

from envs.de import DoubleEscape
from agents.dqn import DeepQNet


if __name__=='__main__':
    env = DoubleEscape()
    dim_obs = env.observation_space_shape[1]*2
    num_act = env.action_reservoir.shape[0]
    agent_0 = DeepQNet(
        dim_obs=dim_obs,
        num_act=num_act
    )
    agent_1 = DeepQNet(
        dim_obs=dim_obs,
        num_act=num_act
    )
    # load models
    q_net_path_0 = './saved_models/double_escape_discrete/dqn/hete_ir/2020-10-01-12-56/agent_0/26206'
    q_net_path_1 = './saved_models/double_escape_discrete/dqn/hete_ir/2020-10-01-12-56/agent_1/26206'
    agent_0.q.q_net = tf.keras.models.load_model(q_net_path_0)
    agent_1.q.q_net = tf.keras.models.load_model(q_net_path_1)
    agent_0.epsilon = 0
    agent_1.epsilon = 0
    num_episodes = 10
    ep = 0
    success_counter = 0
    o, d, t = env.reset(), False, 0
    while ep < num_episodes: 
        while 'blown' in env.status: 
            o, d, t = env.reset(), False, 0
        s0 = o[[0,1]].flatten()
        s1 = o[[1,0]].flatten()
        a0 = np.squeeze(agent_0.act(np.expand_dims(s0, axis=0)))
        a1 = np.squeeze(agent_1.act(np.expand_dims(s1, axis=0)))
        n_o, r, d, i = env.step(np.array([int(a0), int(a1)]))
        n_s0 = n_o[[0,1]].flatten()
        n_s1 = n_o[[1,0]].flatten()
        t += 1
        o = n_o.copy()
        if any([d, t==env.max_episode_steps, 'blown' in env.status]):
            ep +=1
            if i.count('escaped')==2:
                success_counter += 1
            o, d, t = env.reset(), False, 0


