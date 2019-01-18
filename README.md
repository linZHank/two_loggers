# Brief Note

## Pre-requisite
1. [Ubuntu 16.04](http://releases.ubuntu.com/16.04/)
2. [ROS-Kinetic](http://wiki.ros.org/kinetic), `ros-kinetic-desktop-full`
3. (Optional, but recommend)[catkin-command-line-tools](https://catkin-tools.readthedocs.io/en/latest/)

## 
- Create a ros workspace, refer to this [tutorial](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)
Assume your workspace is at `~/ros_ws/`
- 
```bash
cd ~/ros_ws/src
git clone https://github.com/linZHank/two_loggers.git
cd ..
catkin_make
source devel/setup.bash
```
- Launch task in Gazebo `roslaunch loggers_gazebo loggers_world.launch`
- To run test code, `rosrun loggers_gazebo two_loggers_test`

