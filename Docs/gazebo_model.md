# Introduce the Logger Robot into Gazebo
**Important: Before you start this tutorial, please make sure you have completed the one regarding to [Create URDF Model for Logger Robot](https://github.com/linZHank/two_loggers/blob/master/Docs/urdf_model.md).** This tutorial is a modified version of [this gazebo tutorial](http://gazebosim.org/tutorials/?tut=ros_urdf). The special part is we are building a mobile robot here instead of a robotic arm. And of course, we are using a Differential driver to control this robot. To make sure our differential drive plugin works, install following packages first.
```console
sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control
```

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
Now we have our ~~hero~~robot spawned in Gazebo, but it really just is a dummy. Cannot move, cannot jump, cannot fly. So, next let's install a differential drive plugin so that we can drive the robot in a Gazebo environment through ROS commands.

Open the Gazebo markup file:
```console
gedit ~/ros_ws/src/two_loggers/loggers_description/urdf/single_logger.gazebo
```
In between `<robot>` and `</robot>` tags, add the following contents:
```xml
<gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
        <legacyMode>false</legacyMode>
        <alwaysOn>true</alwaysOn>
        <updateRate>100</updateRate>
        <leftJoint>joint_chassis_lwheel</leftJoint>
        <rightJoint>joint_chassis_rwheel</rightJoint>
        <wheelSeparation>0.2</wheelSeparation>
        <wheelDiameter>0.18</wheelDiameter>
        <wheelTorque>1</wheelTorque>
        <!--wheelAcceleration>${wheel_accel}</wheelAcceleration-->
        <commandTopic>cmd_vel</commandTopic>
        <odometryTopic>odom</odometryTopic>
        <odometryFrame>odom</odometryFrame>
        <robotBaseFrame>link_chassis</robotBaseFrame>
    </plugin>
</gazebo>
```
You can change control command frequency use `<updateRate>` tag. The `<commandTopic>` tag indicates the ROS topic name with which you can interact with the robot in Gazebo simulation.

To verify this plugin working correctly, launch our robot in a new terminal:
```console
roslaunch loggers_gazebo single_logger_world.launch
```
Open another terminal and check available ROS topics:
```console
rostopic list
```
You'll see `/cmd_vel` appearing in this ROS topic list.

![rostopic_list](https://github.com/linZHank/two_loggers/blob/master/Docs/images/rostopic_list.png)

## Make a World for Your Robot
Using Gazebo's GUI to create a world is pretty straightforward, you can refer to its official [Tutorial](http://gazebosim.org/tutorials?tut=build_world&cat=build_world) to quickly build a world for your robot. The world for this project is stored at [here](https://github.com/linZHank/two_loggers/tree/master/loggers_gazebo/worlds). A simple walled cell with one exit. Now we can spawn the logger robot within such world. Open a text editor from terminal.
```console
gedit ~/ros_ws/src/two_loggers/loggers_gazebo/launch/single_logger_world.launch
```
Substitue following part
```xml
<!-- launch robot in an empty world -->
<include file="$(find gazebo_ros)/launch/empty_world.launch">
</include>
```
with
```xml
<!-- We resume the logic in empty_world.launch, changing only the name of the world to be launched -->
<include file="$(find gazebo_ros)/launch/empty_world.launch">
  <arg name="world_name" value="$(find loggers_gazebo)/worlds/wall_exit.world"/>
  <arg name="debug" value="$(arg debug)" />
  <arg name="gui" value="$(arg gui)" />
  <arg name="paused" value="$(arg paused)"/>
  <arg name="use_sim_time" value="$(arg use_sim_time)"/>
  <arg name="headless" value="$(arg headless)"/>
</include>
```
Your new simulation should look like below
![robot_cell](https://github.com/linZHank/two_loggers/blob/master/Docs/images/robot_cell.png)
## Control Robot in Gazebo with ROS
For the sake of seperating the controlling code apart, we might want to create another package for all kinds of control implementations.
```console
cd ~/ros_ws/src/
catkin create pkg loggers_control
cd loggers_control
mkdir launch
gedit single_logger_control.launch
```
Basically, we just want this new launch file repeat what we have done in the `loggers_gazebo` package.
```xml
<launch>

    <!-- ros_control rrbot launch file -->
    <include file="$(find loggers_gazebo)/launch/single_logger_world.launch" />

    <!-- convert joint states to TF transforms for rviz, etc -->
    <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"
        respawn="false" output="screen">
        <remap from="/joint_states" to="/single_logger/joint_states" />
    </node>

</launch>
```
In a new terminal, run `roslaunch loggers_control single_logger_control.launch`. An you are good to go.
