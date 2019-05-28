FROM ubuntu:xenial

RUN apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys D2486D2DD83DB69272AFE98867170598AF249743

RUN . /etc/os-release \
    && . /etc/lsb-release \
    && echo "deb http://packages.osrfoundation.org/gazebo/$ID-stable $DISTRIB_CODENAME main" > /etc/apt/sources.list.d/gazebo-latest.list

RUN apt-get update && apt-get install -q -y \
    gazebo8 \
    libgazebo8-dev \
    cmake \
    protobuf-compiler \
    python3-pip \
    nano \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENV HOME /grasping_sim

WORKDIR $HOME/catkin_ws
COPY src src

RUN /bin/bash -c "catkin init && \
    catkin build -DCMAKE_BUILD_TYPE=Release -DPYTHON_VERSION=3.5 -DPYTHON_EXECUTABLE:FILEPATH=`which python3` && \
    echo 'export GRASPING_WS=\$HOME/catkin_ws' >> $HOME/.bashrc && \
    echo 'export GAZEBO_MODEL_PATH=\$GRASPING_WS/src/gazebo_grasping/models' >> $HOME/.bashrc && \
    echo 'export GAZEBO_PLUGIN_PATH=\$GRASPING_WS/build' >> $HOME/.bashrc && \
    echo 'source \$GRASPING_WS/devel/setup.bash' >> $HOME/.bashrc && \
    echo 'source /usr/share/gazebo/setup.sh' >> $HOME/.bashrc"

RUN chmod -R 777 $HOME

RUN useradd -d /grasping_sim -M -N -u 1000 local
RUN useradd -d /grasping_sim -M -N -u 6539 hmerzic
# Remove previous entry point.
ENTRYPOINT []
CMD ["bash"]
