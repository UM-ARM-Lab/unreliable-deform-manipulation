cmake_minimum_required(VERSION 2.8.3)
project(link_bot_pycommon)

find_package(catkin REQUIRED COMPONENTS
        rospy
        roscpp
        pybind11_catkin
        std_msgs)
find_package(catkin COMPONENTS
        pybind11_catkin
        )

catkin_package(
        CATKIN_DEPENDS pybind11_catkin std_msgs roscpp
)

catkin_python_setup()

if(pybind11_catkin_FOUND)
    pybind_add_module(roscpp_initializer src/link_bot_pycommon/roscpp_initializer.cpp)
    target_include_directories(roscpp_initializer SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
    target_link_libraries(roscpp_initializer PUBLIC ${catkin_LIBRARIES})
endif()
