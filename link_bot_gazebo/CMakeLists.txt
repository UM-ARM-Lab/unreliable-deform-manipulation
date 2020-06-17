cmake_minimum_required(VERSION 3.5)
project(link_bot_gazebo)

find_package(catkin REQUIRED COMPONENTS roscpp sdf_tools peter_msgs)
find_package(cnpy REQUIRED)
find_package(gazebo REQUIRED)


set(CMAKE_CXX_STANDARD 17)

catkin_python_setup()

catkin_package()

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(rope_plugin SHARED plugins/rope_plugin.cpp)
target_include_directories(rope_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(rope_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(rope_plugin ${catkin_EXPORTED_TARGETS})

add_library(stepping_plugin SHARED plugins/stepping_plugin.cpp)
target_include_directories(stepping_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(stepping_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(stepping_plugin ${catkin_EXPORTED_TARGETS})

add_library(collision_map_plugin SHARED plugins/collision_map_plugin.cpp)
target_include_directories(collision_map_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(collision_map_plugin cnpy::cnpy ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} stdc++fs)
add_dependencies(collision_map_plugin ${catkin_EXPORTED_TARGETS})

add_library(position_3d_plugin SHARED plugins/position_3d_plugin.cpp)
target_include_directories(position_3d_plugin PUBLIC ${catkin_INCLUDE_DIRS} include)
target_link_libraries(position_3d_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(position_3d_plugin ${catkin_EXPORTED_TARGETS})

add_library(my_camera_plugin SHARED plugins/my_camera_plugin.cpp)
target_include_directories(my_camera_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(my_camera_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} CameraPlugin)
add_dependencies(my_camera_plugin ${catkin_EXPORTED_TARGETS})

add_library(test_ray_plugin SHARED plugins/test_ray_plugin.cpp)
target_include_directories(test_ray_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(test_ray_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(collision_plugin SHARED plugins/ContactPlugin.cpp)
target_include_directories(collision_plugin PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(collision_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

