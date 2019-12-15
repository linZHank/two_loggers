#! /usr/bin/env python
"""
General helper classes and functions for loggers tasks
"""
import sys
import os
import numpy as np
import csv
import pickle
from datetime import datetime
import pickle
import csv
import matplotlib.pyplot as plt

# colored print
class bcolors:
    """ For the purpose of print in terminal with colors """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def increment_mean(pre_mean, new_data, sample_size):
    """
    Compute incremental mean
    """
    inc_mean = pre_mean + (new_data-pre_mean) / sample_size

    return inc_mean

def increment_std(pre_std, pre_mean, inc_mean, new_data, sample_size):
    """
    Compute incremental standard deviation
    """
    pre_nVar = np.power(pre_std,2)*(sample_size-1)
    inc_std = np.sqrt((pre_nVar+(new_data-pre_mean)*(new_data-inc_mean)) / sample_size)

    return inc_std

def normalize(data, mean, std):
    """
    z-standardize
    """
    normed_data = (data - mean) / np.clip(std, 1e-8, 1e16)

    return normed_data

# save pickle
def save_pkl(content, fdir, fname):
    """
    Save content into path/name as pickle file
    Args:
        path: file path, str
        content: to be saved, array/dict/list...
        fname: file name, str
    """
    file_path = os.path.join(fdir, fname)
    with open(file_path, 'wb') as f:
        pickle.dump(content, f, pickle.HIGHEST_PROTOCOL)

def save_csv(content, fdir, fname):
    """
    Save content into path/name as csv file
    Args:
        content: to be saved, dict
        path: file path, str
        fname: file name, str
    """
    file_path = os.path.join(fdir, fname)
    with open(file_path, 'w') as f:
        for key in content.keys():
            f.write("{},{}\n".format(key,content[key]))

def plot_returns(returns, mode, save_flag, fdir):
    """
    Plot rewards
        Args:
        returns:
            episodic returns: list
            mode: 0 - returns; 1 - accumulated returns; 2 - averaged returns, int
            save_flag: save figure to file or not, bool
            path: file path, str
    """
    assert mode==0 or mode==1 or mode==2
    # compute accumulated returns and averaged returns
    acc_returns = []
    ave_returns = []
    acc_r = returns[0]
    ave_r = acc_r
    for i,r in enumerate(returns):
        acc_r += r
        ave_r = acc_r/(i+1)
        acc_returns.append(acc_r)
        ave_returns.append(ave_r)
    # plot
    fig, ax = plt.subplots()
    if mode == 0: # plot return of each episode
        ax.plot(np.arange(len(returns)), returns)
        ax.set(xlabel='Episode', ylabel='Returns')
        ax.set_ylim([-1,1])
        figure_dir = os.path.join(fdir,'episodic_returns.png')
        ax.grid()
    elif mode == 1: # plot accumulated return of each episode
        ax.plot(np.arange(len(acc_returns)), acc_returns)
        ax.set(xlabel='Episode', ylabel='Accumulated Returns')
        figure_dir = os.path.join(fdir, 'accumulated_returns.png')
        ax.grid()
    else: # plot averaged return of eacj episode
        ax.plot(np.arange(len(ave_returns)), ave_returns)
        ax.set(xlabel='Episode', ylabel='Averaged Returns')
        ax.set_ylim([-1,1])
        figure_dir = os.path.join(fdir, 'averaged_returns.png')
        ax.grid()
    if save_flag:
        plt.savefig(figure_dir)
        # plt.close()
    else:
        plt.show()
