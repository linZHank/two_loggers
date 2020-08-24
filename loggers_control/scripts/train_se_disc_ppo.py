#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
from numpy import random
from numpy import pi
import time
from datetime import datetime
import matplotlib.pyplot as plt
import rospy

from envs.se_dis import SoloEscape
from agents.ppo import PPOBuffer, ProximalPolicyOptimization


if __name__=='__main__':
    env = SoloEscape()
    agent = ProximalPolicyOptimization(
        env_type = env.env_type,
        dim_obs=env.observation_space_shape[0],
        dim_act=env.action_reservoir.shape[0]
    )
    replay_buffer = PPOBuffer(dim_obs=env.observation_space_shape[0], dim_act=env.action_reservoir.shape[0], size=5000, gamma=0.99, lam=0.97)
    model_dir = os.path.join(sys.path[0], 'saved_models', env.name, agent.name, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    # paramas
    steps_per_epoch = replay_buffer.max_size
    epochs = 600
    iters_a = 100
    iters_c = 100
    max_ep_len=1000
    save_freq=50
    # Prepare for interaction with environment
    obs, ep_ret, ep_len = env.reset(), 0, 0
    episode_counter, step_counter = 0, 0
    stepwise_rewards, episodic_returns, sedimentary_returns = [], [], []
    episodic_steps = []
    start_time = time.time()
    # main loop
    for ep in range(epochs):
        for st in range(steps_per_epoch):
            act, val, logp = agent.pi_of_a_given_s(np.expand_dims(obs, axis=0))
            n_obs, rew, done, info = env.step(act)
            rospy.logdebug(
                "\nepisode: {}, step: {} \nstate: {} \naction: {} \nreward: {} \ndone: {} \nn_state: {}".format(episode_counter+1, st+1, obs, act, rew, done, n_obs)
            )
            ep_ret += rew
            ep_len += 1
            stepwise_rewards.append(rew)
            step_counter += 1
            replay_buffer.store(obs, act, rew, val, logp)
            obs = n_obs # SUPER CRITICAL!!!
            # handle episode termination
            timeout = (ep_len==env.max_episode_steps)
            terminal = done or timeout
            epoch_ended = (st==steps_per_epoch-1)
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at {} steps.'.format(ep_len), flush=True)
                if timeout or epoch_ended:
                    _, val, _ = agent.pi_of_a_given_s(np.expand_dims(obs, axis=0))
                else:
                    val = 0
                replay_buffer.finish_path(val)
                if terminal:
                    episode_counter += 1
                    episodic_returns.append(ep_ret)
                    sedimentary_returns.append(sum(episodic_returns)/episode_counter)
                    episodic_steps.append(step_counter)
                    rospy.loginfo(
                        "\n----\nTotalSteps: {} \nEpisode: {}, EpReturn: {}, EpLength: {}\n----\n".format(step_counter, episode_counter, ep_ret, ep_len)
                    )
                obs, ep_ret, ep_len = env.reset(), 0, 0
        # update actor-critic
        loss_pi, loss_v, loss_info = agent.train(replay_buffer.get(), iter_a, iter_c)
        rospy.loginfo(
            "\n====\nEpoch: {} \nStep: {} \nAveReturn: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \nEntropy: {} \nTimeElapsed: {}\n====\n".format(episode_counter, step_counter, sedimentary_returns[-1], loss_pi, loss_v, loss_info['kl'], loss_info['ent'], time.time()-start_time)
        )
        # Save model
        if not ep%save_freq or (ep==epochs-1):
            # save logits_net
            logits_net_path = os.path.join(model_dir, 'logits_net', str(ep))
            if not os.path.exists(os.path.dirname(logits_net_path)):
                os.makedirs(os.path.dirname(logits_net_path))
            agent.actor.logits_net.save(logits_net_path)
            # save val_net
            val_net_path = os.path.join(model_dir, 'val_net', str(ep))
            if not os.path.exists(os.path.dirname(val_net_path)):
                os.makedirs(os.path.dirname(val_net_path))
            agent.critic.val_net.save(val_net_path)

    # Save returns 
    np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
    np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
    np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
    with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
        f.write("{}".format(time.time()-start_time))

    # Test
    input("Press ENTER to test agent...")
    for ep in range(10):
        o, d = env.reset(), False
        for st in range(env.max_episode_steps):
            a, _, _ = agent.pi_of_a_given_s(np.expand_dims(o, axis=0))
            o2, rew, done, info = env.step(a)
            o = o2.copy()
            if d:
                break 

    # plot returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()

