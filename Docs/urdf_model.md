# Create URDF Model for Logger Robot
**Please make sure your system is successfully set up following the [system setup guide](https://github.com/linZHank/two_loggers/blob/master/Docs/system_setup.md)**

URDF is a file format which enables visualization of robots' model in [Rviz](http://wiki.ros.org/rviz) or [Gazebo](http://gazebosim.org/). Rviz is a GUI interface of viewing all kinds of ROS related things, mostly the topics. Gazebo is an open sourced dynamic simulation software. Both Rviz and Gazebo should have been installed if you installed `desktop-full` version of ROS. If more details of *urdf* is desired, please go through [Building a Visual Robot Model with URDF from Scratch](http://wiki.ros.org/urdf/Tutorials/Building%20a%20Visual%20Robot%20Model%20with%20URDF%20from%20Scratch).

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
1. Fire up your favorite text editor (gedit, emacs, gedit, sublime, vim, etc.) to describe your robot in ***urdf*** manner.
    ```console
    roscd loggers_description/urdf
    gedit single_logger.urdf.xacro
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
Chassis link is in a disc shape, so we can describe it with a cylinder. Similarly, we can define its property after header part.
```xml
<!-- properties of link_chassis -->
<xacro:property name="R_CHASSIS" value="0.2"/>
<xacro:property name="L_CHASSIS" value="0.1"/>
<xacro:property name="M_CHASSIS" value="2"/>
```
Then specify `visual`, `collision` and `inertal`.
```xml
<link name="link_chassis">
    <visual>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <geometry>
            <cylinder length="${L_CHASSIS}" radius="${R_CHASSIS}"/>
        </geometry>
    </visual>
    <collision>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <geometry>
            <cylinder length="${L_CHASSIS}" radius="${R_CHASSIS}"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="2"/>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <inertia ixx="${1/12*M_CHASSIS*(3*R_CHASSIS*R_CHASSIS+L_CHASSIS*L_CHASSIS)}" ixy="0" ixz="0" iyy="${1/12*M_CHASSIS*(3*R_CHASSIS*R_CHASSIS+L_CHASSIS*L_CHASSIS)}" iyz="0" izz="${1/2*M_CHASSIS*R_CHASSIS*R_CHASSIS}"/>
    </inertial>
</link>
```

## Create Wheels
We'll use two cylinders to describe the wheels. Old story, define properties of the wheels to save you money.
```xml
<!-- properties of link_wheel -->
<xacro:property name="R_WHEEL" value="0.09"/>
<xacro:property name="L_WHEEL" value="0.04"/>
<xacro:property name="M_WHEEL" value="0.5"/>
```
Now, things are a little different. Both hat link and chassis link are modeled as cylinders without any rotation. Wheels are different, because we want them to rotate on the ground, but the cylinder's default orientation prevents them doing this. Hence when describing the wheels we need make a little change at the `<origin>` tags. Let's take left wheel as an example, right wheel is exactly the same.
```xml
<link name="link_left_wheel">
    <visual>
        <origin rpy="0 ${PI/2} 0" xyz="0 0 0"/>
        <geometry>
            <cylinder length="${L_WHEEL}" radius="${R_WHEEL}"/>
        </geometry>
    </visual>
    <collision>
        <origin rpy="0 ${PI/2} 0" xyz="0 0 0" />
        <geometry>
            <cylinder length="${L_WHEEL}" radius="${R_WHEEL}"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="${M_WHEEL}"/>
        <origin rpy="0 ${PI/2} 0" xyz="0 0 0"/>
        <inertia ixx="${1/12*M_WHEEL*(3*R_WHEEL*R_WHEEL+L_WHEEL*L_WHEEL)}" ixy="0" ixz="0" iyy="${1/12*M_WHEEL*(3*R_WHEEL*R_WHEEL+L_WHEEL*L_WHEEL)}" iyz="0" izz="${1/2*M_WHEEL*R_WHEEL*R_WHEEL}"/>
    </inertial>
</link>
```
So we rotate the wheel link about the Y-axis of global coordinate system with 90 degrees.

## Connect Links by a Joint
Now, let's connect links with joints. Generally, you need to define the `<origin>`, `<parent link>` and `<child link>` of the joint. `<origin>` can be determined from previous joint position and orientation. The first joint is defined with respect to the origin of the global coordinate system. In this model, we have two types of joints: `continuous` and `fixed`. Let's take the *"chassis wheel joint"* as an example, besides the before mentioned 3 tags, `continuous` joint need to define rotation axis with a `<axis>` tag.
```xml
<joint name="joint_chassis_lwheel" type="continuous">
    <origin rpy="0 0 0" xyz="${R_CHASSIS/2} ${R_CHASSIS/2} 0"/>
    <parent link="link_chassis"/>
    <child link="link_left_wheel" />
    <axis rpy="0 0 0" xyz="1 0 0"/>
</joint>
```

## View Model in Rviz
More details of this *urdf* file can be found [here](https://github.com/linZHank/two_loggers/blob/master/loggers_description/urdf/single_logger.urdf.xacro). Now let's walk through viewing this *urdf* model in *Rviz*.

### 1. Create a launch file
Open a terminal,
```console
cd ~/ros_ws/src/two_loggers/loggers_control
mkdir launch
cd launch
gedit single_logger_rviz.launch
```
Fill your launch file with following contents.
```xml
<launch>
  <param name="robot_description"
    command="$(find xacro)/xacro --inorder '$(find loggers_description)/urdf/single_logger.urdf.xacro'" />

  <!-- send fake joint values -->
  <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher">
    <param name="use_gui" value="TRUE"/>
  </node>

  <!-- Combine joint values -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="state_publisher"/>

  <!-- Show in Rviz   -->
  <node name="rviz" pkg="rviz" type="rviz"/>

</launch>
```
### 2. Launch the model in Rviz
Open a terminal
```console
roslaunch loggers_description single_logger_rviz.launch
```
You will see something similar to the following interface
![rviz empty](https://github.com/linZHank/two_loggers/blob/master/Docs/images/rviz_empty.png)

But where is the model? And what is the error in `Display` pane on left hand side? Don't be panic, switch the `Fixed Frame` from `map` to `link_chassis`.
![switch frame](https://github.com/linZHank/two_loggers/blob/master/Docs/images/switch_fixed_frame.png)

Now the error has gone, but still, where is the model? Did you see the `Add` button on the left to the bottom of the `Display` pane? Click it, scroll down within the `By display type` tag. Select `RobotModel` then click `OK`.
![add model](https://github.com/linZHank/two_loggers/blob/master/Docs/images/add_model.png)

Here comes the model
![rviz model](https://github.com/linZHank/two_loggers/blob/master/Docs/images/rviz_model.png)

### 3. Rviz Configuration
But isn't this painful? Especially when you have to do this every time you want to launch your model in *Rviz*. A solution could be create a *Rviz* configuration file, `gedit single_logger.rviz`.
Fill the file with the following contents
```yaml
Panels:
  - Class: rviz/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded: ~
      Splitter Ratio: 0.5
    Tree Height: 257
  - Class: rviz/Selection
    Name: Selection
  - Class: rviz/Tool Properties
    Expanded:
      - /2D Pose Estimate1
      - /2D Nav Goal1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.588679016
  - Class: rviz/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
  - Class: rviz/Time
    Experimental: false
    Name: Time
    SyncMode: 0
    SyncSource: LaserScan
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.0299999993
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz/RobotModel
      Collision Enabled: false
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
        link_chassis:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        link_left_wheel:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        link_right_wheel:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
      Name: RobotModel
      Robot Description: robot_description
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
  Enabled: true
  Global Options:
    Background Color: 238; 238; 238
    Fixed Frame: link_chassis
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz/Interact
      Hide Inactive Objects: true
    - Class: rviz/MoveCamera
    - Class: rviz/Select
    - Class: rviz/FocusCamera
    - Class: rviz/Measure
    - Class: rviz/SetInitialPose
      Topic: /initialpose
    - Class: rviz/SetGoal
      Topic: /move_base_simple/goal
    - Class: rviz/PublishPoint
      Single click: true
      Topic: /clicked_point
  Value: true
  Views:
    Current:
      Class: rviz/Orbit
      Distance: 10.3891287
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.0599999987
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: -0.543884933
        Y: -0.380490273
        Z: 0.564803362
      Focal Shape Fixed Size: false
      Focal Shape Size: 0.0500000007
      Name: Current View
      Near Clip Distance: 0.00999999978
      Pitch: 0.845398188
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.89539808
    Saved: ~
Window Geometry:
  Camera:
    collapsed: false
  Displays:
    collapsed: false
  Height: 876
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd00000004000000000000028a00000326fc0200000009fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000006400fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000002800000190000000dd00fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261fb0000000c00430061006d00650072006101000001be000001900000001600ffffff000000010000010f000002e2fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a005600690065007700730000000028000002e2000000b000fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000006400000003efc0100000002fb0000000800540069006d00650000000000000006400000030000fffffffb0000000800540069006d00650100000000000004500000000000000000000003b00000032600000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Selection:
    collapsed: false
  Time:
    collapsed: false
  Tool Properties:
    collapsed: false
  Views:
    collapsed: false
  Width: 1600
  X: 0
  Y: 24
```

Then modify last few lines in your launch file `gedit single_logger_rviz.launch` to include this *Rviz* configuration file.
```xml
<!-- Show in Rviz   -->
<node name="rviz" pkg="rviz" type="rviz" args="-d $(find loggers_description)/launch/single_logger.rviz"/>
```
