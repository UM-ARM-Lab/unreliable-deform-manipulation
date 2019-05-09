cmake_minimum_required(VERSION 3.0)
project(latent_constraint_learning)

set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS rospy)
