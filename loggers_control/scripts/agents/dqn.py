#!/usr/bin/env python
""" 
A DQN agent class 
"""
import tensorflow as tf
import numpy as np
import logging

################################################################
"""
Can safely ignore this block
"""
# restrict GPU and memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
# set log level
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
################################################################

################################################################
class OffPolicyBuffer:
    """
    An off-policy replay buffer for DQN agent
    """
    def __init__(self, dim_obs, size):
        self.obs_buf = np.zeros([size]+[dim_obs], dtype=np.float32)
        self.nobs_buf = np.zeros_like(self.obs_buf)
        self.act_buf = np.zeros(shape=size, dtype=np.int8)
        self.rew_buf = np.zeros(shape=size, dtype=np.float32)
        self.done_buf = np.zeros(shape=size, dtype=np.bool)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, done, nobs):
        self.obs_buf[self.ptr] = obs
        self.nobs_buf[self.ptr] = nobs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1)%self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=1024):
        ids = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs = tf.convert_to_tensor(self.obs_buf[ids], dtype=tf.float32),
            nobs = tf.convert_to_tensor(self.nobs_buf[ids], dtype=tf.float32),
            act = tf.convert_to_tensor(self.act_buf[ids], dtype=tf.int32),
            rew = tf.convert_to_tensor(self.rew_buf[ids], dtype=tf.float32),
            done = tf.convert_to_tensor(self.done_buf[ids], dtype=tf.float32),
        )

        return batch
################################################################

def mlp(dim_inputs, dim_outputs, activation, output_activation=None):
    inputs = tf.keras.Input(shape=(dim_inputs,), name='input')
    features = tf.keras.layers.Dense(128, activation=activation)(inputs)
    features = tf.keras.layers.Dense(128, activation=activation)(features)
    outputs = tf.keras.layers.Dense(dim_outputs, activation=output_activation)(features)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

class Critic(tf.keras.Model):
    
    def __init__(self, dim_obs, num_act, activation, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        inputs = tf.keras.Input(shape=dim_obs, name='inputs')
        # image features
        self.q_net = mlp(dim_inputs=dim_obs, dim_outputs=num_act, activation=activation)
        
    @tf.function
    def call(self, obs):
        return self.q_net(obs)
        # return tf.squeeze(qval, axis=-1)

class DeepQNet(tf.keras.Model):

    def __init__(self, dim_obs, num_act, activation='relu', gamma=0.99, lr=3e-4, polyak=0.995, update_freq=8000, **kwargs):
        super(DeepQNet, self).__init__(name='dqn', **kwargs)
        # params
        self.dim_obs = dim_obs
        self.num_act = num_act
        self.gamma = gamma # discount rate
        self.polyak = polyak
        self.update_freq = update_freq
        self.init_eps = 1.
        self.final_eps = .1
        # model
        self.q = Critic(dim_obs, num_act, activation) 
        self.targ_q = Critic(dim_obs, num_act, activation)
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        # variable
        self.epsilon = self.init_eps
        self.update_counter = 0

    def linear_epsilon_decay(self, episode, decay_period, warmup_episodes):
        episodes_left = decay_period + warmup_episodes - episode
        bonus = (self.init_eps - self.final_eps) * episodes_left / decay_period
        bonus = np.clip(bonus, 0., self.init_eps-self.final_eps)
        self.epsilon = self.final_eps + bonus

    def act(self, obs):
        if np.random.rand() > self.epsilon:
            a = tf.argmax(self.q(obs), axis=-1)
        else:
            a = tf.random.uniform(shape=[1,1], maxval=self.num_act, dtype=tf.dtypes.int32)
        return a

    def train_one_batch(self, data):
        # update critic
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_weights)
            pred_qval = tf.math.reduce_sum(self.q(data['obs']) * tf.one_hot(data['act'], self.num_act), axis=-1)
            targ_qval = data['rew'] + self.gamma*(1-data['done'])*tf.math.reduce_sum(self.targ_q(data['nobs'])*tf.one_hot(tf.math.argmax(self.q(data['nobs']),axis=1), self.num_act),axis=1) # double DQN trick
            loss_q = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval)
        logging.debug("q loss: {}".format(loss_q))
        grads = tape.gradient(loss_q, self.q.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.q.trainable_weights))
        self.update_counter += 1
        if self.polyak > 0:
            # Polyak average update target Q-nets
            q_weights_update = []
            for w_q, w_targ_q in zip(self.q.get_weights(), self.targ_q.get_weights()):
                w_q_upd = self.polyak*w_targ_q
                w_q_upd = w_q_upd + (1 - self.polyak)*w_q
                q_weights_update.append(w_q_upd)
            self.targ_q.set_weights(q_weights_update)
        else:
            if not self.update_counter%self.update_freq:
                self.targ_q.q_net.set_weights(self.q.q_net.get_weights())
                print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nTarget Q-net updated\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")

        return loss_q

