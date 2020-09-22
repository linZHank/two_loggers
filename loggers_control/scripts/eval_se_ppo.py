#! /usr/bin/env python
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import time
import rospy

from envs.se import SoloEscape
from agents.ppo import PPOBuffer, ProximalPolicyOptimization


if __name__=='__main__':
    env = SoloEscape()
    agent = ProximalPolicyOptimization(
        env_type = env.env_type,
        dim_obs=env.observation_space_shape[0],
        dim_act=env.action_reservoir.shape[0]
    )
    logits_net_path = './saved_models/solo_escape_discrete/ppo/2020-09-09-13-25/logits_net/399'
    val_net_path = './saved_models/solo_escape_discrete/ppo/2020-09-09-13-25/val_net/399'
    # load models
    agent.actor.logits_net = tf.keras.models.load_model(logits_net_path)
    agent.critic.val_net = tf.keras.models.load_model(val_net_path)
    for ep in range(10):
        o, d = env.reset(), False
        for st in range(env.max_episode_steps):
            a, _, _ = agent.pi_of_a_given_s(np.expand_dims(o, axis=0))
            o2, r, d, i= env.step(a)
            o = o2.copy()
            if d:
                break 


