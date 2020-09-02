# Two Loggers Gazebo Simulation
**Tutorials of the contents in this repo are all located at [Docs](https://github.com/linZHank/two_loggers/tree/master/Docs). The order is: [system_setup](https://github.com/linZHank/two_loggers/blob/master/Docs/system_setup.md)->[create_urdf_tutorial](https://github.com/linZHank/two_loggers/blob/master/Docs/create_urdf_tutorial.md)->[gazebo_ros_tutorial](https://github.com/linZHank/two_loggers/blob/master/Docs/gazebo_ros_tutorial.md)**

## Setup
- [Ubuntu 16.04](http://releases.ubuntu.com/16.04/) or [Ubuntu 18.04](http://releases.ubuntu.com/18.04/)
- [ROS-Kinetic](http://wiki.ros.org/kinetic) in Ubuntu 16.04 or [ROS-Melodic](http://wiki.ros.org/melodic) in Ubuntu 18.04
- [Python 2.7](https://www.python.org/download/releases/2.7/),
- [TensorFlow 2](https://www.tensorflow.org/) and [TensorFlow Probability](https://www.tensorflow.org/probability)
**The lastest Python2 supported tensorflow2 version is 2.1, tensorflow-probability version is 0.9, please make sure the right
version is installed. Example: `pip install tensorflow==2.1` and `pip install tensorflow-probability==0.9`**

> PyTorch or other deep learning libraries should be no problem working with the environment developed in this repo.

- gazebo_ros_pkgs
``` console
sudo apt-get install ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control
```
- [Create a catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace), assume your workspace is at `~/ros_ws/`
- Clone this repo to your catkin workspace
```console
cd ~/ros_ws/src
git clone https://github.com/linZHank/two_loggers.git
```
- Build ROS packages (`loggers_description`, `loggers_gazebo`, `loggers_control`)

``` console
cd ~/ros_ws/
catkin_make
source devel/setup.bash
```
> [Catkin Command Line Tools](https://catkin-tools.readthedocs.io/en/latest/) is a substitution to `catkin_make`.

- make sure following two lines are in your `~/.bashrc` file.
``` bash
source /opt/ros/melodic/setup.bash
source /home/linzhank/ros_ws/devel/setup.bash
```
> Replace `melodic` with `kinetic` in the lines above if you are using ROS-Kinetic.
## Environments
Two environments are available right now: `this_repo/loggers_control/scripts/envs/se.py` and `this_repo/loggers_control/scripts/envs/de.py` both are with discrete action space.

`se` indicates *solo escape*, the goal is control the **logger** robot exiting the room through the only opening on the south
wall.

`de` indicates *double escape*, the goal is control a two-robot team formed with two **logger** robots exiting the room while
carrying a 2m rod.

**Usage**

> A segment of example code can be found in the end of each script.

Open a terminal and enter the following commands to launch the simulation.
```console
roslaunch loggers_control double_logger_control.launch
```
In a new terminal (tab), enter the following commands to test the environment with random control signals.
```console
cd this_repo/loggers_control/scripts/envs
python de.py
```
> substitute `this_repo` with actual repo path.

> for solo escape env, `roslaunch loggers_control solo_logger_control.launch`, then navigate to the `envs`
> location and run `python se.py`

## Agents
Two DQN agents are available and tested:
1. [DQN](https://www.nature.com/articles/nature14236)
2. [PPO](https://arxiv.org/abs/1707.06347)

Location: `this_repo/loggers_control/scripts/agents`

**Usage**

Open a terminal and enter the following commands to launch the simulation.
```console
roslaunch loggers_control solo_logger_control.launch
```
In a new terminal (tab), enter the following commands to train a ppo controller in `solo escape` environment.
```console
rosrun loggers_control train_se_ppo.py
```
> Since the envs are updated, please refer to `legacy_20200817` branch for tested scripts training distributed controllers using
> DQN algorithm.

> Under developing scripts include new agents, new envs could be found in `devel` branch.
