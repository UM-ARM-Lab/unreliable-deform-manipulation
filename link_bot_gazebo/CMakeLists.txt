project(link_bot_gazebo)
cmake_minimum_required(VERSION 3.5)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs geometry_msgs message_generation)
find_package(gazebo REQUIRED)

add_message_files(
        FILES
        LinkBotConfiguration.msg
)

add_service_files(
        FILES
        WorldControl.srv
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(linkbot_model_plugin SHARED plugins/linkbot_model_plugin.cpp)
target_include_directories(linkbot_model_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(linkbot_model_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(linkbot_model_plugin link_bot_gazebo_generate_messages_cpp)

add_library(stepping_plugin SHARED plugins/stepping_plugin.cpp)
target_include_directories(stepping_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(stepping_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(stepping_plugin link_bot_gazebo_generate_messages_cpp)

