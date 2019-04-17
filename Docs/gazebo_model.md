# Introduce the Logger Robot into Gazebo
** Important: Before you start this tutorial, please make sure you have completed the one regarding to [Create URDF Model for Logger Robot](https://github.com/linZHank/two_loggers/blob/master/Docs/urdf_model.md)** This tutorial is a modified version of [this gazebo tutorial](http://gazebosim.org/tutorials/?tut=ros_urdf). The extras are we are building a mobile robot here instead of a robotic arm. And of course, we are using a Differential driver to control this robot.

## Spawn Robot Model in Gazebo
The *urdf* model is informative enough, we only need a little more touch to bring it alive in Gazebo.
### 1. Links with Gazebo markups
Open a terminal:
```console
cd ~/ros_ws/src/two_loggers/loggers_description/urdf/
gedit single_logger.gazebo
```
Using different materials on each link to make our robot distinguishable. Fill the contents below in to this file
```xml
<?xml version="1.0"?>
<robot>

    <gazebo reference="link_left_wheel">
        <mu1>1</mu1>
        <mu2>1</mu2>
        <material>Gazebo/Wood</material>
    </gazebo>

    <gazebo reference="link_right_wheel">
        <mu1>1</mu1>
        <mu2>1</mu2>
        <material>Gazebo/Wood</material>
    </gazebo>

    <gazebo reference="link_caster">
        <mu1>0.0001</mu1>
        <mu2>0.0001</mu2>
        <material>Gazebo/DarkGrey</material>
    </gazebo>

    <gazebo reference="link_chassis">
        <material>Gazebo/White</material>
    </gazebo>

    <gazebo reference="link_hat">
        <material>Gazebo/Wood</material>
    </gazebo>

</robot>
```
The tags of `<mu1>` and `<mu2>` set friction for the specified links. The `<material>` tag gives the links textures. A full description of all available materials in Gazebo can be found [here](https://bitbucket.org/osrf/gazebo/src/default/media/materials/scripts/gazebo.material?fileviewer=file-view-default)

### 2. Create a launch file
In order to use `roslaunch`, we need create a launch file. But first, let's create a ROS package for the Gazebo applications. Open a terminal and type:
```console
cd ~/ros_ws/src/two_loggers/
catkin create pkg loggers_gazebo
cd loggers_gazebo
mkdir launch
gedit single_logger_world.launch
```
Copy and paste following contents into this file
```xml
<launch>
    <!-- these are the arguments you can pass this launch file, for example paused:=true -->
    <arg name="paused" default="false"/>
    <arg name="use_sim_time" default="true"/>
    <arg name="gui" default="true"/>
    <arg name="headless" default="false"/>
    <arg name="debug" default="false"/>

    <!-- launch robot in an empty world -->
    <include file="$(find gazebo_ros)/launch/empty_world.launch">
    </include>

    <!-- Load the URDF into the ROS Parameter Server -->
    <param name="robot_description"
        command="$(find xacro)/xacro --inorder '$(find loggers_description)/urdf/single_logger.urdf.xacro'" />

    <!-- Run a python script to the send a service call to gazebo_ros to spawn a URDF robot -->
    <node name="urdf_spawner" pkg="gazebo_ros" type="spawn_model"       respawn="false" output="screen"
        args="-urdf -z 0.2 -Y -1.57 -model logger -param robot_description"/>

</launch>
```
Open a terminal then run:
```console
roslaunch loggers_gazebo single_logger_world.launch
```
> Your allied champion has respawned!
![robot_gazebo_empty](https://github.com/linZHank/two_loggers/blob/master/Docs/images/robot_gazebo_empty.png)


## Gazebo Differential Drive Plug-in

## Make a World for Your Robot

## Control Robot in Gazebo with ROS
