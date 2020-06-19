# Two Loggers Gazebo Simulation
**Tutorials of the contents in this repo are all located at [Docs](https://github.com/linZHank/two_loggers/tree/master/Docs). The order is: [system_setup](https://github.com/linZHank/two_loggers/blob/master/Docs/system_setup.md)->[create_urdf_tutorial](https://github.com/linZHank/two_loggers/blob/master/Docs/create_urdf_tutorial.md)->[gazebo_ros_tutorial](https://github.com/linZHank/two_loggers/blob/master/Docs/gazebo_ros_tutorial.md)**

## Pre-requisites
- Developing Environment
[Ubuntu 16.04](http://releases.ubuntu.com/16.04/) or [Ubuntu 18.04](http://releases.ubuntu.com/18.04/),
[ROS-Kinetic](http://wiki.ros.org/kinetic) or [ROS-Melodic](http://wiki.ros.org/melodic),
[Python 2.7](https://www.python.org/download/releases/2.7/),
[TensorFlow 2.0](https://www.tensorflow.org/)

- Install gazebo_ros_pkgs
``` console
sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control
```
- Create a catkin workspace, assume your workspace is at `~/ros_ws/`
- Clone this repo to your catkin workspace
```console
cd ~/ros_ws/src
git clone https://github.com/linZHank/two_loggers.git
```
- Build ROS packages (`loggers_description`, `loggers_gazebo`, `loggers_control`)
> [Catkin Command Line Tools](https://catkin-tools.readthedocs.io/en/latest/) is recommanded to build the packages

``` console
cd ~/ros_ws/
catkin build
source devel/setup.bash
```
- make sure following two lines are in your `~/.bashrc` file.
``` bash
source /opt/ros/kinetic/setup.bash
source /home/linzhank/ros_ws/devel/setup.bash
```
## Quick Start
- Evaluate **DQN** agent in double escape environment
```console
roslaunch loggers_control double_logger_control.launch
rosrun loggers_control eval_double_1_dqn.py
```

