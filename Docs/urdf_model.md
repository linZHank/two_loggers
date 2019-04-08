# Create URDF Model for Logger Robot
**Please make sure your system is successfully set up following the [system setup guide](https://github.com/linZHank/two_loggers/blob/master/Docs/system_setup.md)**

URDF is a file format which enables visualization of robots' model in [Rviz](http://wiki.ros.org/rviz) or [Gazebo](http://gazebosim.org/). Rviz is a GUI interface of viewing all kinds of ROS related things, mostly the topics. Gazebo is an open sourced dynamic simulation software. Both Rviz and Gazebo should be installed if you installed `desktop-full` version of ROS. Because there exist plenty of Gazebo plugins for ROS, it is very very common for people who are interested in robotics simulation in ROS ecosystem.
> **Note:** there are two kinds of model file formats for Gazebo: ***sdf*** and ***urdf***. If you are not familiar with C++ and do not want to write your own gazebo plugins for ROS, then stick to the ***urdf***.

In the rest of this tutorial, I will walk through some basic steps of creating a urdf model for the single logger robot in this repo.

## Create a ROS Package
> Let's assume you are using [catkin-command-line-tools](https://catkin-tools.readthedocs.io/en/latest/)

1. Navigate to your ROS workspace (e.g. `~/ros_ws`)
    ```console
    cd ~/ros_ws/src
    ```
2. Create a ROS package. You can understand a ROS package
as a folder tailored for ROS. You'll store your robots' urdf file in this package.
    ```console
    catkin create pkg loggers_description
    ```
> If you are using original `catkin_create_pkg`, please refer to the [Creating Package Tutorial](http://wiki.ros.org/ROS/Tutorials/CreatingPackage)

3. Create a folder for storing your URDF
    ```console
    cd ~/ros_ws/src/loggers_description
    mkdir urdf
    ```
## Create Chassis
A robot model is always created follow a *link-joint-link* manner. You can also refer to the [official tutorials](http://wiki.ros.org/urdf/Tutorials) of using urdf model files.
First of all, we need

## Create a Wheel

## Connect Links by a Joint

## Write a Launch File

### Test in Rviz
