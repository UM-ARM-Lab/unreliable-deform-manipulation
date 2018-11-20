project(link_bot_gazebo)
cmake_minimum_required(VERSION 3.5)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs)
find_package(gazebo REQUIRED)

catkin_package()

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")
add_library(linkbot_model_plugin SHARED plugins/linkbot_model_plugin.cpp)
target_include_directories(linkbot_model_plugin PUBLIC ${catkin_INCLUDE_DIRS})
message("linkbot_model_plugin ${catkin_LIBRARIES}")
target_link_libraries(linkbot_model_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
