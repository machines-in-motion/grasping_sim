cmake_minimum_required(VERSION 2.8.3)
project(gazebo_grasping)

# To make sure python3 is found.
add_definitions(-DPYTHON_VERSION=3.5)
# add_definitions(-DPYTHON_EXECUTABLE:FILEPATH=${PYTHON3})
## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
find_program(PYTHON3 python3 REQUIRED)

find_package(catkin REQUIRED)
find_package(gazebo REQUIRED)
find_package(ignition-transport3 REQUIRED)
file(GLOB Boost_LIBRARIES "external/lib/*")
set(Boost_NO_SYSTEM_PATHS ON)
set(Python_ADDITIONAL_VERSIONS 3.5)
find_package(PythonLibs 3.5 REQUIRED) # sets ${PYTHON_INCLUDE_DIRS}

list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME} gazebo_grasping_msgs
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${IGNITION-TRANSPORT_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}/msgs
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}/msgs
  ${IGNITION-TRANSPORT_LIBRARY_DIRS}
)
add_subdirectory(msgs)

## Declare a C++ library
add_library(${PROJECT_NAME} SHARED
  src/gazebo_world_iface.cc
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
target_link_libraries(${PROJECT_NAME}
  gazebo_grasping_msgs
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  ${IGNITION-TRANSPORT_LIBRARIES}
)
add_dependencies(${PROJECT_NAME} gazebo_grasping_msgs)

add_library(noise SHARED
  src/noise.cc
)
target_link_libraries(noise
  ${catkin_LIBRARIES}
  ${IGNITION-MATH_LIBRARIES}
)

add_library(world_plugin SHARED
  src/world_plugin.cc
)
target_link_libraries(world_plugin
  gazebo_grasping_msgs
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  ${IGNITION-TRANSPORT_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  noise
)
add_dependencies(world_plugin gazebo_grasping_msgs)

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
