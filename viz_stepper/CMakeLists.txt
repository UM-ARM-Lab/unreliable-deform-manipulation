cmake_minimum_required(VERSION 3.0.0)
project(viz_stepper)

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp peter_msgs)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_CXX_STANDARD 17)

catkin_python_setup()

catkin_package()

set(viz_stepper_src
        src/viz_stepper/viz_stepper.cpp
        include/viz_stepper/viz_stepper.h)
add_executable(viz_stepper ${viz_stepper_src})
target_include_directories(viz_stepper PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(viz_stepper PUBLIC Qt5::Core Qt5::Widgets ${catkin_LIBRARIES} uuid)
add_dependencies(viz_stepper ${catkin_EXPORTED_TARGETS})
