sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install libignition-transport-dev protobuf-compiler libgazebo7-dev

./bootstrap.sh --with-python=`which python3` --prefix=$HOME/local/boost
./b2 --with-python cxxflags=-fPIC install

export CATKIN_WS=$HOME/catkin_ws
source $CATKIN_WS/devel/setup.bash
export GAZEBO_MODEL_PATH=$CATKIN_WS/src/gazebo_grasping_barebones/models
export GAZEBO_PLUGIN_PATH=$CATKIN_WS/build:$GAZEBO_PLUGIN_PATH

catkin build -DPYTHON_VERSION=3.5 -DPYTHON_EXECUTABLE:FILEPATH=`which python3`
