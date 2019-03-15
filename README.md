# Two Loggers Gazebo Simulation

## Pre-requisites
- [Ubuntu 16.04](http://releases.ubuntu.com/16.04/)
- [ROS-Kinetic](http://wiki.ros.org/kinetic), `ros-kinetic-desktop-full`
- [catkin-command-line-tools](https://catkin-tools.readthedocs.io/en/latest/)
- [TensorFlow](https://www.tensorflow.org/)

## Quick Start
- Install gazebo_ros_pkgs

``` console
sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control
```
- Create a catkin workspace, refer to this [tutorial](http://wiki.ros.org/catkin/Tutorials/create_a_workspace). Assume your workspace is at `~/ros_ws/`
- Clone this repo to your catkin workspace
```console
cd ~/ros_ws/src
git clone https://github.com/linZHank/two_loggers.git
```
- Build these ROS packages (`loggers_description`, `loggers_gazebo`, `loggers_control`)
``` console
cd ~/ros_ws/
catkin build
source devel/setup.bash
```
> Better make sure following two lines are in your `~/.bashrc` file.
``` console
source /opt/ros/kinetic/setup.bash
source /home/linzhank/ros_ws/devel/setup.bash
```
- Launch a single logger robot simulation
``` console
roslaunch loggers_control single_logger_control.launch 
```

## Notes
- To test out single logger environment`rosrun loggers_control single_logger_test.py`
- To train a neural network model with **Vanilla Policy Gradient** algorithm: `rosrun loggers_control solo_escape_vpg.py`
- A trained neural network model is stored at `this_repo/loggers_control/vpg_model`

# Demo
## Solo Escape with VPG
[![IMAGE ALT TEXT HERE](https://youtu.be/xqkG5bBXyY8)
