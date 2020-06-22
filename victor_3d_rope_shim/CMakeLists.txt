cmake_minimum_required(VERSION 2.8.3)
project(victor_3d_rope_shim)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  arc_utilities
  moveit_ros_planning
  pcl_ros
  peter_msgs
  roscpp
  tf2_ros
  tf2_msgs
  victor_hardware_interface
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
  src/${PROJECT_NAME}/victor_shim.cpp
  src/${PROJECT_NAME}/Manipulator.cpp
  src/${PROJECT_NAME}/VictorManipulator.cpp
  src/${PROJECT_NAME}/Scene.cpp)
target_link_libraries(world_interaction ${catkin_LIBRARIES})
target_compile_options(world_interaction PRIVATE ${FLAGS})

#############
## Install ##
#############

## Copy launch files
# install(DIRECTORY launch/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
# )

# ## Install robot files
# install(DIRECTORY urdf/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
# )
# install(DIRECTORY Media/models/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/Media/models
# )

# ## Install RVIZ configurations
# install(DIRECTORY rviz/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
# )
