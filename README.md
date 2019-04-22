# Two Loggers Gazebo Simulation
**Tutorials of the contents in this repo are all located at [Docs](https://github.com/linZHank/two_loggers/tree/master/Docs). The order is: [system_setup](https://github.com/linZHank/two_loggers/blob/master/Docs/system_setup.md)->[create_urdf_tutorial](https://github.com/linZHank/two_loggers/blob/master/Docs/create_urdf_tutorial.md)->[gazebo_ros_tutorial](https://github.com/linZHank/two_loggers/blob/master/Docs/gazebo_ros_tutorial.md)**

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

## Policy Gradient Example
- Launch a single logger robot simulation
``` console
roslaunch loggers_control single_logger_control.launch
```
- To test out single logger environment`rosrun loggers_control single_logger_test.py`
- To train a neural network model with **Vanilla Policy Gradient** algorithm: `rosrun loggers_control solo_escape_vpg_train.py`
- Trained multilayer perceptrons models are stored at `this_repo/loggers_control/vpg_model`
- To evaluate models: `rosrun loggers_control solo_escape_vpg_eval.py`
  > You'll need to change the model path manually in the script, will write an argparse for this in the future

# Demo
## Solo Escape with VPG
[![IMAGE ALT TEXT HERE](http://i3.ytimg.com/vi/xqkG5bBXyY8/hqdefault.jpg)](https://youtu.be/xqkG5bBXyY8)
