#! /usr/bin/env python
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import time
import rospy

from envs.de import DoubleEscape
from agents.ppo import ProximalPolicyOptimization


if __name__=='__main__':
    env = DoubleEscape()
    dim_obs = env.observation_space_shape[0]*env.observation_space_shape[1]
    num_act = env.action_reservoir.shape[0]
    dim_act = num_act**2
    agent = ProximalPolicyOptimization(
        env_type = env.env_type,
        dim_obs=dim_obs,
        dim_act=dim_act
    )
    logits_net_path = './saved_models/double_escape_discrete/ppo/2020-09-01-18-47/logits_net/100'
    val_net_path = './saved_models/double_escape_discrete/ppo/2020-09-01-18-47/val_net/100'
    # load models
    agent.actor.logits_net = tf.keras.models.load_model(logits_net_path)
    agent.critic.val_net = tf.keras.models.load_model(val_net_path)
    for ep in range(10):
        o, d = env.reset(), False
        for st in range(env.max_episode_steps):
            s = o.flatten()
            a, _, _ = agent.pi_of_a_given_s(np.expand_dims(s, axis=0))
            o, r, d, i= env.step(np.array([int(a/num_act), int(a%num_act)]))
            if d:
                break 


