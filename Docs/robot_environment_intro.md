# A Brief Introduction on Robotic Task Environment

## Available Environemnts
There are two environemnts in total at current stage. One is `solo_escape` the other is `double_escape` The python wrapper of the environments are stored at `thisrepo/loggers_control/scripts/envs/`. An example of using `double_escape` environment is
```python
import numpy as np
import rospy
from envs.double_escape_task_env import DoubleEscapeEnv
rospy.init_node("double_escape_env_test")
env = DoubleEscapeEnv()
env.reset()
for _ in range(1000):
    a0 = np.random.randn(2)
    a1 = np.random.randn(2)
    observation, reward, done, info = env.step(a0, a1) # take a random action
env.step(np.zeros(2), np.zeros(2))
```

## Getting Started
Using the environments requires two steps of operation.
1. Launch simulation
2. Run script
An example of running a test simulation of `double_escape` task is:
- Open a terminal to launch `two_loggers` simulation
```console
roslaunch loggers_control two_loggers_control.launch
```
- Run a script based on the simulation
```console
rosrun loggers_control double_escape_env_test.py
```
[![double_escape_test](http://i3.ytimg.com/vi/BvJD3rj6EMg/hqdefault.jpg)](https://youtu.be/BvJD3rj6EMg)

## Functions
The environments use `reset` function to get back to the initial state and `step` function to take an action to change the environment. `reset` function returns two values: `observation` and `information`. `step` function returns four values: `observation`, `reward`, `done` and `information`.
- **observation** (dict): detailed description of the environment, for example, the *position*, *orientation*, *linear velocity* and *angular velocity* of every link in the robot model at a certain time step.
- **reward** (float): amount of reward achieved by the previous action. In reinforcement learning, your goal is always maximize your total reward.
- **done** (boolean): flag for whether to terminate the current episode. Each time an environemnt was reset, a new episode was created. When certain conditions fufilled, an episode will be marked as **done=True** and will be reset.
- **information** (dict): diagnostic information useful for debugging.

Please refer to the example above for the use of environmental functions.
