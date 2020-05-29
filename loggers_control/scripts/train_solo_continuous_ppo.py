#! /usr/bin/env python
"""
A Tensorflow 2 implementation of Proximal Policy Gradient (PPO) for solo_escape_task with continuous action space. PPO is a Model free, on-policy, reinforcement learning algorithm (https://arxiv.org/abs/1707.06347).
This script heavily referenced OpenAI Spinningup's implementation (https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo).
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import scipy.signal
import numpy as np
import random
import time
from datetime import datetime
import pickle
import rospy
from envs.solo_escape_continuous_env import SoloEscapeContinuousEnv

import tensorflow as tf
import tensorflow_probability as tfp
print(tf.__version__, tfp.__version__) # Make sure tf==2.1.0, tfp=0.9.0
tfd = tfp.distributions


################################################################
"""
Useful functions
"""
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    Args:
        [x0, x1, x2]
    Returns:
        [x0 + discount*x1 + discount^2*x2, x1 + discount*x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def mlp(sizes, activation, output_activation=None):
    inputs = tf.keras.Input(shape=(sizes[0],))
    x = tf.keras.layers.Dense(sizes[1], activation=activation)(inputs)
    for i in range(2,len(sizes)-1):
        x = tf.keras.layers.Dense(sizes[i], activation=activation)(x)
    outputs = tf.keras.layers.Dense(sizes[-1], activation=output_activation)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
################################################################


################################################################
"""
Actor-critic class
"""
class Actor(tf.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def __call__(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(MLPCategoricalActor, self).__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return tfd.Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(MLPGaussianActor, self).__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = tf.Variable(log_std)
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = tf.squeeze(self.mu_net(obs))
        std = tf.math.exp(self.log_std)
        return tfd.Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return tf.math.reduce_sum(pi.log_prob(act), axis=-1)

class MLPCritic(tf.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super(MLPCritic, self).__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    # @tf.function
    def __call__(self, obs):
        return tf.squeeze(self.v_net(obs), axis=-1)

class MLPActorCritic(tf.Module):
    """
    The core of PPO is actor-critic
    """
    def __init__(self, obs_dim, act_dim, mode='continuous', hidden_sizes=(64,64), activation='tanh'):
        super(MLPActorCritic, self).__init__()
        if mode=='continuous':
            self.actor = MLPGaussianActor(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes, activation=activation)
        if mode=='discrete':
            self.actor = MLPCategoricalActor(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes, activation=activation)
        self.critic = MLPCritic(obs_dim=obs_dim, hidden_sizes=hidden_sizes, activation=activation)

    # @tf.function
    def step(self, obs):
        with tf.GradientTape() as t:
            with t.stop_recording():
                pi_dist = self.actor._distribution(obs)
                a = pi_dist.sample()
                logp_a = self.actor._log_prob_from_distribution(pi_dist, a)
                v = self.critic(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    # @tf.function
    def act(self, obs):
        return self.step(obs)[0]
################################################################


################################################################
"""
Compute losses and gradients
"""
def compute_actor_gradients(data):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
    with tf.GradientTape() as tape:
        tape.watch(ac.actor.trainable_variables)
        pi, logp = ac.actor(obs, act)
        # print("pi: {} \nlogp: {}".format(pi, logp))
        ratio = tf.math.exp(logp - logp_old)
        clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-clip_ratio, 1+clip_ratio), adv)
        ent = tf.math.reduce_sum(pi.entropy(), axis=-1)
        objective = tf.math.minimum(tf.math.multiply(ratio, adv), clip_adv) + .01*ent
        loss_pi = -tf.math.reduce_mean(objective)
        # useful info
        approx_kl = tf.math.reduce_mean(logp_old - logp, axis=-1)
        entropy = tf.math.reduce_mean(ent)
        pi_info = dict(kl=approx_kl, ent=entropy)
    actor_grads = tape.gradient(loss_pi, ac.actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, ac.actor.trainable_variables))

    return loss_pi, pi_info

def compute_critic_gradients(data):
    obs, ret = data['obs'], data['ret']
    with tf.GradientTape() as tape:
        tape.watch(ac.critic.trainable_variables)
        loss_v = tf.keras.losses.MSE(ret, ac.critic(obs))
    critic_grads = tape.gradient(loss_v, ac.critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, ac.critic.trainable_variables))

    return loss_v

def update(buffer):
    data = buffer.get()
    for i in range(train_pi_iters):
        loss_pi, pi_info = compute_actor_gradients(data)
        kl = pi_info['kl']
        if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
    for j in range(train_v_iters):
        loss_v = compute_critic_gradients(data)

    return loss_pi, pi_info, loss_v
################################################################


################################################################
class PPOBuffer:
    """
    A on-policy buffer for storing trajectories experienced by a PPO agent interacting with the environment.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, uses rewards and value estimates from the whole trajectory to compute advantage estimates with GAE-Lambda, as well as compute the rewards-to-go for each state.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr
        # self.ptr, self.path_start_idx = 0, 0

    def get(self):
        """
        Call this to get all of the data from the buffer with advantages normalized (shifted to have mean zero and std one).
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in data.items()}
################################################################


################################################################
"""
Main
"""
if __name__ == "__main__":
    # instantiate env
    env=SoloEscapeContinuousEnv()
    # paramas
    steps_per_epoch=40000
    epochs=200
    gamma=0.99
    clip_ratio=0.2
    pi_lr=3e-4
    vf_lr=1e-3
    train_pi_iters=80
    train_v_iters=80
    lam=0.97
    max_ep_len=1000
    target_kl=0.01
    save_freq=100
    # instantiate actor-critic and replay buffer
    obs_dim=env.observation_space_shape[0]
    act_dim=env.action_space_shape[0]
    ac = MLPActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    buffer = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    # create optimizer
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # Prepare for interaction with environment
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_dir = os.path.join(sys.path[0], 'saved_models', env.name, 'ppo', date_time, 'models')
    start_time = time.time()
    success_counter = 0
    obs, ep_ret, ep_len = env.reset(), 0, 0
    episodes, total_steps = 0, 0
    stepwise_rewards, episodic_returns, averaged_returns = [], [], []
    # main loop
    for ep in range(epochs):
        for st in range(steps_per_epoch):
            act, val, logp = ac.step(obs.reshape(1,-1))
            next_obs, rew, done, info = env.step(act)
            ep_ret += rew
            ep_len += 1
            stepwise_rewards.append(rew)
            total_steps += 1
            buffer.store(obs, act, rew, val, logp)
            obs = next_obs # SUPER CRITICAL!!!
            # handle episode termination
            timeout = (ep_len==env.max_episode_steps)
            terminal = done or timeout
            epoch_ended = (st==steps_per_epoch-1)
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at {} steps.'.format(ep_len))
                if timeout or epoch_ended:
                    _, val, _ = ac.step(obs.reshape(1,-1))
                else:
                    val = 0
                buffer.finish_path(val)
                if terminal:
                    episodes += 1
                    episodic_returns.append(ep_ret)
                    averaged_returns.append(sum(episodic_returns)/episodes)
                    if info == "escaped":
                        success_counter += 1
                    print("\nTotalSteps: {} \nEpisode: {}, Step: {}, EpReturn: {}, EpLength: {}, Success: {} ".format(total_steps, episodes, st+1, ep_ret, ep_len, success_counter))
                obs, ep_ret, ep_len = env.reset(), 0, 0
        # update actor-critic
        loss_pi, pi_info, loss_v = update(buffer)
        print("\n================================================================\nEpoch: {} \nTotalSteps: {} \nAveReturn: {} \nSuccess: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \n Entropy: {} \nTimeElapsed: {}\n================================================================\n".format(ep+1, total_steps, averaged_returns[-1], success_counter, loss_pi, loss_v, pi_info['kl'], pi_info['ent'], time.time()-start_time))
        # save model
        if not ep%save_freq or (ep==epochs-1):
            model_path = os.path.join(model_dir, str(ep))
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            tf.saved_model.save(ac, model_path)
################################################################

    # plot averaged returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()
