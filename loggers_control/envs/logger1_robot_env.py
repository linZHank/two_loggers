from __future__ import absolute_import, division, print_function

import numpy as np
import rospy
from gym_gazebo_env import GymGazeboEnv
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, LaserScan, PointCloud2, Imu
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class Logger1RobotEnv(GymGazeboEnv):
  """
  Superclass for all loggers environments. Contains all sensors and actuators methods.
  """

  def __init__(self):
    """
    Initializes a new Loggers environment.
    
    Sensor Topic List:
    * /odom : Odometry readings of the base of the robot    
    Simulation Topic:
    * /gazebo/model_states: Gazebo simulated model states
    Actuators Topic List: 
    * /cmd_vel: differential driving command for logger 0
    * /cmd_vel_1: differential driving command for logger 1
    """
    rospy.logdebug("Start LoggersRobotEnv...")
    # Variables that we give through the constructor.
    # None in this case

    # Controller Vars
    self.controllers_list = ["chassis_drive_controller"]
    self.robot_name_space = "logger"
    self.reset_controls = False

    # We launch the init function of the Parent Class gym_gazebo_env.GymGazeboEnv
    super(Logger1RobotEnv, self).__init__(
      controllers_list=self.controllers_list,
      robot_name_space=self.robot_name_space,
      reset_controls=self.reset_controls,
      start_init_physics_parameters=True,
      reset_world_or_sim="WORLD"
    )

    self.gazebo.unpauseSim()
    #self.controllers_object.reset_controllers()
    self._check_all_sensors_ready()

    # We Start all the ROS related Subscribers and publishers
    rospy.Subscriber("/odom", Odometry, self._odom_callback)
    rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback)

    self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    self._check_publishers_connection()

    self.gazebo.pauseSim()
        
    rospy.logdebug("Finished Loggers INIT...")

    
  # Methods needed by the RobotGazeboEnv
  # ----------------------------
  def _check_all_systems_ready(self):
    """
    Checks that all the sensors, publishers and other simulation systems are
    operational.
    """
    self._check_all_sensors_ready()
    return True


  # TurtleBotEnv virtual methods
    # ----------------------------

  def _check_all_sensors_ready(self):
    rospy.logdebug("START ALL SENSORS READY")
    self._check_odom_ready()
    self._check_model_states_ready()
    rospy.logdebug("ALL SENSORS READY")

  def _check_odom_ready(self):
    self.odom = None
    rospy.logdebug("Waiting for /odom to be READY...")
    while self.odom is None and not rospy.is_shutdown():
      try:
        self.odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
        rospy.logdebug("Current /odom READY=>")
      except:
        rospy.logerr("Current /odom not ready yet, retrying for getting odom")

    return self.odom

  def _check_model_states_ready(self):
    self.model_states = None
    rospy.logdebug("Waiting for /gazebo/model_states to be READY...")
    while self.model_states is None and not rospy.is_shutdown():
      try:
        self.model_states = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5.0)
        rospy.logdebug("Current /gazebo/model_states READY=>")
      except:
        rospy.logerr("Current /gazebo/model_states not ready yet, retrying for getting model_states")
        
    return self.model_states

  # Call back functions read subscribed sensors' data
  # ----------------------------
  def _odom_callback(self, data):
    self.odom = data
    
  def _model_states_callback(self, data):
    self.model_states = data


  def _check_publishers_connection(self):
    """
    Checks that all the publishers are working
    :return:
    """
    rate = rospy.Rate(10)  # 10hz
    while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
      rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
      try:
        rate.sleep()
      except rospy.ROSInterruptException:
        # This is to avoid error when world is rested, time when backwards.
        pass
    rospy.logdebug("_cmd_vel_pub Publisher Connected")
    rospy.logdebug("All Publishers READY")
    
  def get_odom(self):
    return self.odom
        
  def get_model_states(self):
    return self.model_states


  # Methods that the TrainingEnvironment will need to define here as virtual
  # because they will be used in GazeboEnv GrandParentClass and defined in the
  # TrainingEnvironment.
  # ----------------------------
  def _set_init(self):
    """Sets the Robot in its init pose
    """
    raise NotImplementedError()
    
  def _compute_reward(self):
    """Calculates the reward to give based on the observations given.
    """
    raise NotImplementedError()

  def _take_action(self, action):
    """Applies the given action to the simulation.
    """
    raise NotImplementedError()

  def _get_observation(self):
    raise NotImplementedError()

  def _post_information(self):
    raise NotImplementedError()

  def _is_done(self):
    """Checks if episode done based on observations given.
    """
    raise NotImplementedError()
