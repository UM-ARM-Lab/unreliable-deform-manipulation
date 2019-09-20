project(link_bot_gazebo)
cmake_minimum_required(VERSION 3.5)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs geometry_msgs message_generation sdf_tools)
find_package(cnpy REQUIRED)
find_package(gazebo REQUIRED)


catkin_python_setup()

add_message_files(
        FILES
        LinkBotConfiguration.msg
        LinkBotVelocityAction.msg
        LinkBotForceAction.msg
        WriteSDF.msg
        Position2dEnable.msg
        Position2dAction.msg
        ObjectAction.msg
        IntPoint2D.msg
)

add_service_files(
        FILES
        WorldControl.srv
        QuerySDF.srv
        LinkBotState.srv
        ComputeSDF2.srv
        CameraProjection.srv
        InverseCameraProjection.srv
        LinkBotPositionAction.srv
        LinkBotPath.srv
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        sensor_msgs
        sdf_tools
)

catkin_package(CATKIN_DEPENDS message_runtime)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(multi_link_bot_model_plugin SHARED plugins/multi_link_bot_model_plugin.cpp)
target_include_directories(multi_link_bot_model_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(multi_link_bot_model_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(multi_link_bot_model_plugin link_bot_gazebo_generate_messages)

add_library(stepping_plugin SHARED plugins/stepping_plugin.cpp)
target_include_directories(stepping_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(stepping_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(stepping_plugin link_bot_gazebo_generate_messages)

add_library(collision_map_plugin SHARED plugins/collision_map_plugin.cpp)
target_include_directories(collision_map_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(collision_map_plugin cnpy::cnpy ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} stdc++fs)
add_dependencies(collision_map_plugin link_bot_gazebo_generate_messages)

add_library(position_2d_plugin SHARED plugins/position_2d_plugin.cpp)
target_include_directories(position_2d_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(position_2d_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(position_2d_plugin link_bot_gazebo_generate_messages)

add_library(my_camera_plugin SHARED plugins/my_camera_plugin.cpp)
target_include_directories(my_camera_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(my_camera_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} CameraPlugin)

add_library(test_ray_plugin SHARED plugins/test_ray_plugin.cpp)
target_include_directories(test_ray_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(test_ray_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
