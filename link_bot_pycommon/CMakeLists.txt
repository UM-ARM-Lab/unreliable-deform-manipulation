cmake_minimum_required(VERSION 2.8.3)
project(link_bot_pycommon)

find_package(catkin REQUIRED COMPONENTS
        rospy
        roscpp
        std_msgs)

catkin_package(
        CATKIN_DEPENDS std_msgs roscpp
)

catkin_python_setup()
