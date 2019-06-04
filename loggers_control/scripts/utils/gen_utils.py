#! /usr/bin/env python
"""
General helper classes and functions for loggers tasks
"""
import numpy as np
import csv
import pickle
import tensorflow as tf
import os
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

# save pickle
def save_pkl(content, path, fname):
    """
    Save content into path/name as pickle file
    Args:
        path: file path, str
        content: to be saved, array/dict/list...
        fname: file name, str
    """
    fdir = os.path.dirname(path)
    file_path = os.path.join(fdir,fname)
    with open(file_path, "wb") as f:
        pickle.dump(content, f, pickle.HIGHEST_PROTOCOL)

def save_csv(content, path, fname):
    """
    Save content into path/name as csv file
    Args:
        content: to be saved, dict
        path: file path, str
        fname: file name, str
    """
    fdir = os.path.dirname(path)
    file_path = os.path.join(fdir,fname)
    with open(file_path, "w") as f:
        for key in content.keys():
            f.write("{},{}\n".format(key,content[key]))

def plot_returns(returns, mode, save_flag, path):
    """
    Plot rewards
    Args:
    returns: episodic returns, list
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
    if mode == 0:
        ax.plot(np.arange(len(returns)), returns)
        ax.set(xlabel="Episode", ylabel="Returns")
        figure_dir = os.path.join(os.path.dirname(path),"episodic_returns.png")
    elif mode == 1:
        ax.plot(np.arange(len(acc_returns)), acc_returns)
        ax.set(xlabel="Episode", ylabel="Accumulated Returns")
        figure_dir = os.path.join(os.path.dirname(path),"accumulated_returns.png")
    else:
        ax.plot(np.arange(len(ave_returns)), ave_returns)
        ax.set(xlabel="Episode", ylabel="Averaged Returns")
        figure_dir = os.path.join(os.path.dirname(path),"averaged_returns.png")
        ax.grid()
    if save_flag:
        plt.savefig(figure_dir)
        plt.close(fig)
    else:
        plt.show()
