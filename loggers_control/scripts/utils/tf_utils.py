#! /usr/bin/env python
"""
Tensorflow related tools
"""
import numpy as np
import tensorflow as tf

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
        return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def update_mean(old_mean, new_data, sample_size):
    """
    Compute incremental mean
    """
    inc_mean = old_mean + (new_data-old_mean) / sample_size

    return inc_mean

def update_std(old_std, old_mean, inc_mean, new_data, sample_size):
    """
    Compute incremental standard deviation
    """
    old_nVar = np.power(old_std,2)*(sample_size-1)
    inc_std = np.sqrt((old_nVar+(new_data-old_mean)*(new_data-inc_mean)) / sample_size)

def normalize(data, mean, std):
    """
    z-standardize
    """
    normed_data = (data - mean) / np.clip(std, 1e-6, np.inf)

    return normed_data
