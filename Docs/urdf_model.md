# Create URDF Model for Logger Robot
**Please make sure your system is successfully set up following the [system setup guide](https://github.com/linZHank/two_loggers/blob/master/Docs/system_setup.md)**

URDF is a file format which enables visualization of robots' model in [Rviz](http://wiki.ros.org/rviz) or [Gazebo](http://gazebosim.org/). Rviz is a GUI interface of viewing all kinds of ROS related things, mostly the topics. Gazebo is an open sourced dynamic simulation software. Both Rviz and Gazebo should have been installed if you installed `desktop-full` version of ROS. This tutorial is highly inspired by [Tutorial: Using a URDF in Gazebo](http://gazebosim.org/tutorials/?tut=ros_urdf).

## Why Gazebo
To be honest, I don't know why most people are using Gazebo for their robot simulations. It seems Gazebo has plenty of plugins for ROS, which makes manipulating Gazebo simulated robots within ROS ecosystem much easier.

## URDF vs SDF
There are two kinds of file formats for Gazebo models: *sdf* and *urdf*. I and many people I know have been spending a while to struggle with using sdf or urdf. In short, **urdf** for your robot, **sdf** for your environment. For those who are familiar with C++ and willing to get hands dirty to write customized ROS plugins, **sdf** is OK for covering all the things. In this tutorial, I am going to show you how to simulate a mobile robot with **urdf** format.

A robot model is always created follow a *link-joint-link* manner. You can also refer to the [official tutorials](http://wiki.ros.org/urdf/Tutorials) to better understand what is going on over here. The order of the links does not matter, let's say we are going to build our logger robot from top to bottom.

## Create a ROS Package
> Let's assume you are using [catkin-command-line-tools](https://catkin-tools.readthedocs.io/en/latest/)

1. Navigate to your ROS workspace (e.g. `~/ros_ws`)
    ```console
    cd ~/ros_ws/src
    ```
2. Create a ROS package. You can understand a ROS package as a folder tailored for ROS. You'll store your robots' urdf file in this package.
    ```console
    catkin create pkg loggers_description
    ```
    > If you are using original `catkin_create_pkg`, please refer to the [Creating Package Tutorial](http://wiki.ros.org/ROS/Tutorials/CreatingPackage)

3. Create a folder for storing your URDF
    ```console
    cd ~/ros_ws/src/loggers_description
    mkdir urdf
    cd urdf
    ```

## Create a Dummy Link
1. Fire up your favorite text editor (atom, emacs, gedit, sublime, vim, etc.) to describe your robot in ***urdf*** manner.
    ```console
    roscd loggers_description/urdf
    atom single_logger.urdf.xacro
    ```
> ".xacro" does not make urdf different. It still is urdf, but is allowing using shortcuts to [clean up a urdf file](http://wiki.ros.org/urdf/Tutorials/Using%20Xacro%20to%20Clean%20Up%20a%20URDF%20File)

2. Write header for your robot urdf
    ```xml
    <?xml version="1.0" ?>
    <robot name="logger" xmlns:xacro="https://www.ros.org/wiki/xacro" >
    ```

3. Create a dummy link for the whole simulation world
    ```xml
    <link name="world"/>
    ```

## Create the Hat
Hat is the link on the top of our logger robot. Hat link is a cylinder with radius of 0.02m, length of 0.1m and mass of 0.5kg. For the sake of the convenience, we can define these properties right after the header.
```xml
<!-- properties of link_hat -->
<xacro:property name="R_HAT" value="0.02"/>
<xacro:property name="L_HAT" value="0.1"/>
<xacro:property name="M_HAT" value="0.5"/>
```

A link in *urdf* requires 3 major components: `visual`, `collision` and `inertial`.
> For common shapes, the inertia tensor can be found at [List of moments of inertia](https://en.wikipedia.org/wiki/List_of_moments_of_inertia).

```xml
<link name="link_hat">
    <visual>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <geometry>
            <cylinder length="${L_HAT}" radius="${R_HAT}"/>
        </geometry>
    </visual>
    <collision>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <geometry>
            <cylinder length="${L_HAT}" radius="${R_HAT}"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="${M_HAT}"/>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <inertia ixx="${1/12*M_HAT*(3*R_HAT*R_HAT+L_HAT*L_HAT)}" ixy="0" ixz="0" iyy="${1/12*M_HAT*(3*R_HAT*R_HAT+L_HAT*L_HAT)}" iyz="0" izz="${1/2*M_HAT*R_HAT*R_HAT}"/>
    </inertial>
</link>
```

## Create the Chassis

## Create Wheels

## Connect Links by a Joint

## Write a Launch File

### Test in Rviz
