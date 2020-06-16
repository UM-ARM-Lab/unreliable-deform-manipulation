cmake_minimum_required(VERSION 2.8.3)
project(victor_3d_rope)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)
find_package(gazebo REQUIRED)

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

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

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
