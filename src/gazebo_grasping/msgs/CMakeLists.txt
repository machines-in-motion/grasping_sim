find_package(Protobuf REQUIRED)

set(PROTOBUF_IMPORT_DIRS)
foreach(ITR ${GAZEBO_INCLUDE_DIRS})
  if(ITR MATCHES ".*gazebo-[0-9.]+$")
    set(PROTOBUF_IMPORT_DIRS "${ITR}/gazebo/msgs")
  endif()
endforeach()

set (msgs
  barrett_control.proto
  point3d.proto
  observation.proto
  point_of_interest.proto
  world_config.proto
)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${msgs})
add_library(gazebo_grasping_msgs SHARED ${PROTO_SRCS})
target_link_libraries(gazebo_grasping_msgs ${PROTOBUF_LIBRARY})
