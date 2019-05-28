# Grasping Learning Environment

Grasping learning environment is a learning environment which simulates grasping tasks in Gazebo. The environment is written as a Python module with the same interface as the environments used in OpenAI's Gym.

Reference paper: "Leveraging Contact Forces For Learning to Grasp" [arXiv](https://arxiv.org/abs/1809.07004)

## License

- "baseline" (Copyright (c) 2017 OpenAi) in src/baselines_catkin/baselines licensed using The MIT License, see related folder for licensing text
- "catkin" (Copyright (c) 2012 Willow Garage, Inc) in src/catkin/ licensed using the Software License Agreement (BSD License). See related folder for licensing text
- "Boost python" (Copyright © 2002-2015 David Abrahams, Stefan Seefeld) in src/gazebo_grasping/external/lib/ licensed using the Boost Software License - Version 1.0. See related folder for licensing text
- Other content (Copyright (c) 2018 Max Planck Gesellschaft) licensed using the BSD 3-Clause Licence

## Installation

In order to install the GLE follow the steps below:

```
# Clone the repo.
git clone git@git-amd.tuebingen.mpg.de:amd-clmc/grasping_sim.git

# Install Gazebo8 and system dependencies.
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y gazebo8 libgazebo8-dev cmake protobuf-compiler python3-pip

# Install Python dependencies.
sudo pip3 install gym catkin_pkg catkin_tools trollius empy mpi4py tensorflow scipy
```

Next step is to create your grasping workspace, link the src folder to the workspace and build the module, e.g.
```
export GRASPING_WS=$HOME/grasping_ws
mkdir $GRASPING_WS
cd $GRASPING_WS
ln -s /path/to/grasping_sim/src src

catkin init
catkin build -DCMAKE_BUILD_TYPE=Release -DPYTHON_VERSION=3.5 -DPYTHON_EXECUTABLE:FILEPATH=`which python3`

# Set up environment variables.
echo "export GRASPING_WS=$GRASPING_WS" >> $HOME/.bashrc
echo "export GAZEBO_MODEL_PATH=\$GRASPING_WS/src/gazebo_grasping/models" >> $HOME/.bashrc
echo "export GAZEBO_PLUGIN_PATH=\$GRASPING_WS/build" >> $HOME/.bashrc
echo "source \$GRASPING_WS/devel/setup.bash" >> $HOME/.bashrc
echo "source /usr/share/gazebo/setup.sh"
source $HOME/.bashrc
```

## Test example

First test if Gazebo is properly installed by running:

```
gazebo --version
```

This should write out `Gazebo multi-robot simulator, version 8.1.1`.

Then open a terminal and run `python3`. The interpreter should start and input the following:

```
from grasping_env import GraspingEnvPregrasps
env = GraspingEnvPregrasps()
env.render()
```

At this point you should have a few information messages printed out. Ignore the warning that the objects will float - this is the intended behavior since the gravity is turned off.

Now, open another terminal and run `gzclient`. You should see a robot hand along with an object in the middle.
You can now control the simulation from the python. For example, to move the fingers you can do:

```
env.step([1, 1, 1, 1])
env.step([1, 1, 1, 1])
env.step([1, 1, 1, 1])
env.step([1, 1, 1, 1])
```

You can restart the environment by doing:

```
env.reset()
```

Before exiting don't forget to close the environment by doing:

```
env.close()
```
