cmake_minimum_required(VERSION 2.8.3)
project(link_bot_pycommon)

find_package(catkin REQUIRED COMPONENTS
        rospy
        roscpp
        pybind11_catkin
        std_msgs)

catkin_package(
        CATKIN_DEPENDS pybind11_catkin std_msgs roscpp
)

catkin_python_setup()

pybind_add_module(pycpp_ros_sub src/link_bot_pycommon/bindings.cpp)
target_include_directories(pycpp_ros_sub SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(pycpp_ros_sub PUBLIC ${catkin_LIBRARIES})
