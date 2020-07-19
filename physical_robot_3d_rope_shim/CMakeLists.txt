cmake_minimum_required(VERSION 2.8.3)
project(physical_robot_3d_rope_shim)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  arc_utilities
  moveit_ros_planning
  peter_msgs
  roscpp
  tf2_ros
  tf2_msgs
  victor_hardware_interface
  moveit_visual_tools
)

set(CMAKE_CXX_STANDARD 17)

# catkin_python_setup()

###################################################
## Declare things to be passed to other projects ##
###################################################

## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package()

###########
## Build ##
###########

include_directories(
  include
  SYSTEM
  ${catkin_INCLUDE_DIRS}
)

set(FLAGS -O3 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=return-type -Werror=missing-field-initializers)

add_executable(world_interaction
  src/world_interaction.cpp
  src/${PROJECT_NAME}/dual_gripper_shim.cpp
  src/${PROJECT_NAME}/planning_interface.cpp
  src/${PROJECT_NAME}/val_interface.cpp
  src/${PROJECT_NAME}/victor_interface.cpp
  src/${PROJECT_NAME}/scene.cpp
)
target_link_libraries(world_interaction PRIVATE ${catkin_LIBRARIES})
target_compile_options(world_interaction PRIVATE ${FLAGS})