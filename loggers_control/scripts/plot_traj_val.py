import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# load trajectory
traj_path = os.path.join(sys.path[0], 'saved_trajectories', '2020-06-29-00-56', 'traj.npy')
traj = np.load(traj_path)
# create figure
fig = plt.figure(figsize=(20,10))
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
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
# plot instances
ax0.plot((traj[0,0], traj[0,-6]), (traj[0,1], traj[0,-5]), linewidth=4, color='orangered', alpha=0.7)
r0 = mpatches.Circle(xy=traj[0,:2], radius=.25, fill=False, ec='k', alpha=0.7)
r1 = mpatches.Circle(xy=traj[0,-6:-4], radius=.25, fc='k', alpha=0.7)
patches.append(r0)
patches.append(r1)
for i in range(8):
    ax0.plot((traj[int(len(traj)*i/8),0], traj[int(len(traj)*i/8),-6]), (traj[int(len(traj)*i/8),1], traj[int(len(traj)*i/8),-5]), linewidth=4, color='orangered', alpha=0.7)
    patches.append(mpatches.Circle(xy=traj[int(len(traj)*i/8),:2], radius=.25, fill=False, ec='k', alpha=0.7))
    patches.append(mpatches.Circle(xy=traj[int(len(traj)*i/8),-6:-4], radius=.25, fc='k', alpha=0.7))
ax0.plot((traj[-1,0], traj[-1,-6]), (traj[-1,1], traj[-1,-5]), linewidth=4, color='orangered', alpha=0.7)
patches.append(mpatches.Circle(xy=traj[-1,:2], radius=.25, fill=False, ec='k', alpha=0.7))
patches.append(mpatches.Circle(xy=traj[-1,-6:-4], radius=.25, fc='k', alpha=0.7))
# set axis
patches_collection = PatchCollection(patches, match_original=True)
ax0.add_collection(patches_collection)
ax0.axis([-6,6,-9,6])
ax0.set_xticks(np.arange(-6,6))
ax0.set_yticks(np.arange(-9,6))
ax0.grid(color='grey', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()
