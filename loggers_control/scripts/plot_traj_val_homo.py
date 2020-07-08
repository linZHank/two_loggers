import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import tensorflow as tf

# load trajectory
traj_dir = os.path.join(sys.path[0], 'saved_trajectories/2020-07-08-14-26/')
traj = np.load(os.path.join(traj_dir, 'traj.npy'))
acts = np.load(os.path.join(traj_dir, 'acts.npy'))
# load models
model_path_0 = os.path.join(sys.path[0], 'saved_models/double_escape_discrete/dqn/2020-05-29-17-33/double_logger/models/1000000.h5')
model_path_1 = os.path.join(sys.path[0], 'saved_models/double_escape_discrete/dqn/2020-05-29-17-33/double_logger/models/3000000.h5')
model_path_2 = os.path.join(sys.path[0], 'saved_models/double_escape_discrete/dqn/2020-05-29-17-33/double_logger/models/5093500.h5')
model_paths = [model_path_0, model_path_1, model_path_2]
# create figure
fig = plt.figure(figsize=(20,12))
ax0 = plt.subplot(1,2,1)
ax_q = (plt.subplot(3,2,2), plt.subplot(3,2,4), plt.subplot(3,2,6))
patches = []
# plot walls
n_wall = mpatches.Rectangle(xy=(-6,5), width=12, height=1, fc='gray')
patches.append(n_wall)
w_wall = mpatches.Rectangle(xy=(-6,-5), width=1, height=10, fc='gray')
patches.append(w_wall)
e_wall = mpatches.Rectangle(xy=(5,-5), width=1, height=10, fc='gray')
patches.append(e_wall)
sw_wall = mpatches.Rectangle(xy=(-6,-6), width=5, height=1, fc='gray')
patches.append(sw_wall)
se_wall = mpatches.Rectangle(xy=(1,-6), width=5, height=1, fc='gray')
patches.append(se_wall)
# draw trajectories
ax0.plot(traj[:,0], traj[:,1], linestyle='--', color='grey')
ax0.plot(traj[:,-6], traj[:,-5], linestyle='--', color='black')
# plot instances of robots, rod and robots' orientation
ax0.plot((traj[0,0], traj[0,-6]), (traj[0,1], traj[0,-5]), linewidth=4, color='orangered', alpha=0.7)
patches.append(mpatches.Circle(xy=traj[0,:2], radius=.25, fill=False, ec='k', alpha=0.7))
patches.append(mpatches.Circle(xy=traj[0,-6:-4], radius=.25, fc='k', alpha=0.7))
ax0.plot((traj[0,0], traj[0,0]+0.4*np.cos(traj[0,4])), (traj[0,1], traj[0,1]+0.4*np.sin(traj[0,4])), color='red')
ax0.plot((traj[0,-6], traj[0,-6]+0.4*np.cos(traj[0,-2])), (traj[0,-5], traj[0,-5]+0.4*np.sin(traj[0,-2])), color='red')
for i in range(1, 11):
    ax0.plot((traj[len(traj)*i/10-1,0], traj[len(traj)*i/10-1,-6]), (traj[len(traj)*i/10-1,1],traj[len(traj)*i/10-1,-5]), linewidth=4, color='orangered', alpha=0.7)
    patches.append(mpatches.Circle(xy=traj[len(traj)*i/10-1,:2], radius=.25, fill=False, ec='k', alpha=0.7))
    patches.append(mpatches.Circle(xy=traj[len(traj)*i/10-1,-6:-4], radius=.25, fc='k', alpha=0.7))
    ax0.plot((traj[len(traj)*i/10-1,0], traj[len(traj)*i/10-1,0]+0.4*np.cos(traj[len(traj)*i/10-1,4])), (traj[len(traj)*i/10-1,1], traj[len(traj)*i/10-1,1]+0.4*np.sin(traj[len(traj)*i/10-1,4])), color='red')
    ax0.plot((traj[len(traj)*i/10-1,-6], traj[len(traj)*i/10-1,-6]+0.4*np.cos(traj[len(traj)*i/10-1,-2])), (traj[len(traj)*i/10-1,-5], traj[len(traj)*i/10-1,-5]+0.4*np.sin(traj[len(traj)*i/10-1,-2])), color='red')

# plot qvals
qvals_0 = np.zeros((3, traj.shape[0]-1))
qvals_1 = np.zeros((3, traj.shape[0]-1))
traj_0 = traj.copy()
traj_1 = traj.copy()
traj_1[:,:6] = traj_1[:,-6:]
for i in range(3):
    dqn = tf.keras.models.load_model(model_paths[i])
    for j in range(qvals_0.shape[1]):
        qvals_0[i,j] = np.squeeze(dqn(np.expand_dims(traj_0[j], axis=0)))[acts[j,0]]
        qvals_1[i,j] = np.squeeze(dqn(np.expand_dims(traj_1[j], axis=0)))[acts[j,1]]
        # qvals_1[i,j] = np.max(dqn(np.expand_dims(traj_1[j], axis=0)))
    ax_q[i].plot(np.arange(j+1), qvals_0[i], color=[.7,.7,.7], label='robot 1')
    ax_q[i].plot(np.arange(j+1), qvals_1[i], color='k', label='robot 2')
# set traj axis
patches_collection = PatchCollection(patches, match_original=True)
ax0.add_collection(patches_collection)
ax0.text(traj[0,0], traj[0,1], 'robot 1', fontsize='large', withdash=True,
         dashdirection=1,
         dashlength=30,
         rotation=0,
         dashrotation=0,
         dashpush=20)
ax0.text(traj[0,-6], traj[0,-5], 'robot 2', fontsize='large', withdash=True,
         dashdirection=1,
         dashlength=30,
         rotation=0,
         dashrotation=0,
         dashpush=20)
ax0.axis([-6,6,-9,6])
ax0.set_xticks(np.arange(-6,7))
ax0.set_yticks(np.arange(-9,7))
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
ax0.grid(color='grey', linestyle=':', linewidth=0.5)
# set qval axes
ax_q[0].margins(x=0, y=0.05)
ax_q[0].set_ylim(0,500)
ax_q[0].set_xticks(len(traj)*np.arange(11)/10, minor=False)
ax_q[0].xaxis.grid(True, which='major')
ax_q[0].yaxis.grid(True, linestyle=(0, (5,10)))
ax_q[0].set_ylabel('Q-values', fontsize='large')
ax_q[0].legend(loc='upper left', fontsize='x-large')
ax_q[0].text(170,20, 'Model version: 1e6', fontsize=14, color='r')
plt.setp(ax_q[0].get_xticklabels(), visible=False)
ax_q[1].margins(x=0, y=0.05)
ax_q[1].set_ylim(0,500)
ax_q[1].set_xticks(len(traj)*np.arange(11)/10, minor=False)
ax_q[1].xaxis.grid(True, which='major')
ax_q[1].yaxis.grid(True, linestyle=(0, (5,10)))
ax_q[1].set_ylabel('Q-values', fontsize='large')
ax_q[1].text(170,20, 'Model version: 3e6', fontsize=14, color='r')
plt.setp(ax_q[1].get_xticklabels(), visible=False)
ax_q[2].margins(x=0, y=0.05)
ax_q[2].set_ylim(0,500)
ax_q[2].set_xticks(len(traj)*np.arange(11)/10, minor=False)
ax_q[2].xaxis.grid(True, which='major')
ax_q[2].yaxis.grid(True, linestyle=(0, (5,10)))
ax_q[2].set_ylabel('Q-values', fontsize='large')
ax_q[2].set_xlabel('Time Step', fontsize='large')
ax_q[2].text(170,20, 'Model version: final', fontsize=14, color='r')
plt.setp(ax_q[2].get_xticklabels())
plt.tight_layout()
plt.show()

