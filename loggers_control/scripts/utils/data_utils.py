#! /usr/bin/env python
"""
General helper classes and functions for loggers tasks
"""
import sys
import os
import numpy as np
import argparse
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

# make arg parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=512)
    parser.add_argument("--num_episodes", type=int, default=8000)
    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sample_size", type=int, default=512)
    parser.add_argument("--layer_sizes", nargs="+", type=int, help="use space to separate layer sizes, e.g. --layer_sizes 4 16 = [4,16]", default=8)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--memory_cap", type=int, default=800000)
    parser.add_argument("--update_step", type=int, default=10000)

    return parser.parse_args()

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
    if mode == 0: # plot return of each episode
        ax.plot(np.arange(len(returns)), returns)
        ax.set(xlabel="Episode", ylabel="Returns")
        ax.set_ylim([-1,1])
        figure_dir = os.path.join(os.path.dirname(path),"episodic_returns.png")
        ax.grid()
    elif mode == 1: # plot accumulated return of each episode
        ax.plot(np.arange(len(acc_returns)), acc_returns)
        ax.set(xlabel="Episode", ylabel="Accumulated Returns")
        figure_dir = os.path.join(os.path.dirname(path),"accumulated_returns.png")
        ax.grid()
    else: # plot averaged return of eacj episode
        ax.plot(np.arange(len(ave_returns)), ave_returns)
        ax.set(xlabel="Episode", ylabel="Averaged Returns")
        ax.set_ylim([-1,1])
        figure_dir = os.path.join(os.path.dirname(path),"averaged_returns.png")
        ax.grid()
    if save_flag:
        plt.savefig(figure_dir)
        plt.close(fig)
    else:
        plt.show()
