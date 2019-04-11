# Create URDF Model for Logger Robot
**Please make sure your system is successfully set up following the [system setup guide](https://github.com/linZHank/two_loggers/blob/master/Docs/system_setup.md)**

URDF is a file format which enables visualization of robots' model in [Rviz](http://wiki.ros.org/rviz) or [Gazebo](http://gazebosim.org/). Rviz is a GUI interface of viewing all kinds of ROS related things, mostly the topics. Gazebo is an open sourced dynamic simulation software. Both Rviz and Gazebo should have been installed if you installed `desktop-full` version of ROS.

## Why Gazebo
To be honest, I don't know why most people are using Gazebo for their robot simulations. It seems Gazebo has plenty of plugins for ROS, which makes manipulating with Gazebo simulated robots within ROS ecosystem much easier.

## URDF vs SDF
There are two kinds of file formats for Gazebo models: ***sdf*** and ***urdf***. I and many people I know have been spending a while to struggle with using ***sdf*** or ***urdf***. In short, **urdf** for your robot, **sdf** for your environment. For those who are familiar with C++ and willing to get hands dirty to write customized ROS plugins, **sdf** is OK for covering all the things. In this tutorial, I am going to show you how to simulate a mobile robot with **urdf** format.

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
