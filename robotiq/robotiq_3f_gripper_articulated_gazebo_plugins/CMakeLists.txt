cmake_minimum_required(VERSION 2.8.3)
project(robotiq_3f_gripper_articulated_gazebo_plugins)

set(CMAKE_CXX_STANDARD 20)

find_package(catkin REQUIRED COMPONENTS gazebo_plugins sensor_msgs roscpp gazebo_ros victor_hardware_interface_msgs robotiq_3f_gripper_articulated_msgs)

find_package(gazebo REQUIRED)

catkin_package(
    DEPENDS gazebo_plugins gazebo_ros robotiq_3f_gripper_articulated_msgs roscpp
)

add_library(RobotiqHandPlugin src/RobotiqHandPlugin.cpp src/robotiq_control.cpp)
target_include_directories(RobotiqHandPlugin PUBLIC include ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(RobotiqHandPlugin ${catkin_LIBRARIES})
add_dependencies(RobotiqHandPlugin ${catkin_EXPORTED_TARGETS})

