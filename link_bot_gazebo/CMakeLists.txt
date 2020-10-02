cmake_minimum_required(VERSION 3.5)
project(link_bot_gazebo)

find_package(
        catkin
        REQUIRED COMPONENTS
        roscpp
        sdf_tools
        arm_video_recorder
        arc_utilities
        peter_msgs
        control_msgs
        moveit_msgs
        sensor_msgs
        trajectory_msgs
        tf2_ros
        victor_hardware_interface
)

find_package(cnpy REQUIRED)
find_package(gazebo REQUIRED)


set(CMAKE_CXX_STANDARD 17)

catkin_python_setup()

catkin_package(
        LIBRARIES
        gazebo_ros_moveit_planning_scene

        CATKIN_DEPENDS
        gazebo_msgs
        geometry_msgs
        message_runtime
        moveit_msgs
        arc_utilities
        arm_video_recorder
        roscpp
        sensor_msgs
        std_msgs
        std_srvs
        trajectory_msgs
        peter_msgs
        tf2_ros
        victor_hardware_interface
)

include_directories(SYSTEM ${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(FLAGS ${GAZEBO_CXX_FLAGS} -O3 -g -Wall -Wextra -Wshadow -Werror=return-type -Werror=missing-field-initializers)

add_library(rope_plugin SHARED plugins/rope_plugin.cpp plugins/gazebo_plugin_utils.cpp)
target_include_directories(rope_plugin PUBLIC include)
target_include_directories(rope_plugin SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(rope_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(rope_plugin ${catkin_EXPORTED_TARGETS})
target_compile_options(rope_plugin PRIVATE ${FLAGS})

add_library(stepping_plugin SHARED plugins/stepping_plugin.cpp)
target_include_directories(stepping_plugin PUBLIC include)
target_include_directories(stepping_plugin SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(stepping_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(stepping_plugin ${catkin_EXPORTED_TARGETS})
target_compile_options(stepping_plugin PRIVATE ${FLAGS})

add_library(collision_map_plugin SHARED plugins/collision_map_plugin.cpp)
target_include_directories(collision_map_plugin PUBLIC include)
target_include_directories(collision_map_plugin SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(collision_map_plugin cnpy::cnpy ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} stdc++fs)
add_dependencies(collision_map_plugin ${catkin_EXPORTED_TARGETS})
target_compile_options(collision_map_plugin PRIVATE ${FLAGS})

add_library(position_3d_plugin SHARED plugins/position_3d_plugin.cpp plugins/gazebo_plugin_utils.cpp)
target_include_directories(position_3d_plugin PUBLIC include)
target_include_directories(position_3d_plugin SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(position_3d_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(position_3d_plugin ${catkin_EXPORTED_TARGETS})
target_compile_options(position_3d_plugin PRIVATE ${FLAGS})


# add_library(kinematic_victor_plugin SHARED plugins/kinematic_victor_plugin.cpp plugins/gazebo_plugin_utils.cpp)
# target_include_directories(kinematic_victor_plugin PUBLIC include)
# target_include_directories(kinematic_victor_plugin SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
# target_link_libraries(kinematic_victor_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
# add_dependencies(kinematic_victor_plugin ${catkin_EXPORTED_TARGETS})
# target_compile_options(kinematic_victor_plugin PRIVATE ${FLAGS})

# add_library(kinematic_val_plugin SHARED plugins/kinematic_val_plugin.cpp plugins/gazebo_plugin_utils.cpp)
# target_include_directories(kinematic_val_plugin PUBLIC include)
# target_include_directories(kinematic_val_plugin SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
# target_link_libraries(kinematic_val_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
# add_dependencies(kinematic_val_plugin ${catkin_EXPORTED_TARGETS})
# target_compile_options(kinematic_val_plugin PRIVATE ${FLAGS})

add_library(gazebo_ros_moveit_planning_scene plugins/gazebo_ros_moveit_planning_scene.cpp)
target_include_directories(gazebo_ros_moveit_planning_scene PUBLIC include)
target_include_directories(gazebo_ros_moveit_planning_scene SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
add_dependencies(gazebo_ros_moveit_planning_scene ${catkin_EXPORTED_TARGETS})
target_link_libraries(gazebo_ros_moveit_planning_scene ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES})
target_compile_options(gazebo_ros_moveit_planning_scene PRIVATE ${FLAGS})

add_library(dual_gripper_plugin plugins/dual_gripper_plugin.cpp plugins/gazebo_plugin_utils.cpp)
target_include_directories(dual_gripper_plugin PUBLIC include)
target_include_directories(dual_gripper_plugin SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
add_dependencies(dual_gripper_plugin ${catkin_EXPORTED_TARGETS})
target_link_libraries(dual_gripper_plugin ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES})
target_compile_options(dual_gripper_plugin PRIVATE ${FLAGS})

add_library(gazebo_ros_tf_plugin plugins/gazebo_ros_tf_plugin.cpp)
target_include_directories(gazebo_ros_tf_plugin PUBLIC include)
target_include_directories(gazebo_ros_tf_plugin SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
add_dependencies(gazebo_ros_tf_plugin ${catkin_EXPORTED_TARGETS})
target_link_libraries(gazebo_ros_tf_plugin ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES})
target_compile_options(gazebo_ros_tf_plugin PRIVATE ${FLAGS})

add_library(rosparam_plugin plugins/rosparam_plugin.cpp)
target_include_directories(rosparam_plugin PUBLIC include)
target_include_directories(rosparam_plugin SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
add_dependencies(rosparam_plugin ${catkin_EXPORTED_TARGETS})
target_link_libraries(rosparam_plugin ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES} stdc++fs)
target_compile_options(rosparam_plugin PRIVATE ${FLAGS})

add_library(arm_video_recorder_plugin plugins/arm_video_recorder_plugin.cpp)
target_include_directories(arm_video_recorder_plugin PUBLIC include)
target_include_directories(arm_video_recorder_plugin SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
add_dependencies(arm_video_recorder_plugin ${catkin_EXPORTED_TARGETS})
set(LIBAV
    avformat
    avcodec
    swresample
    swscale
    avutil
)

target_link_libraries(arm_video_recorder_plugin ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES} CameraPlugin ${LIBAV})
target_compile_options(arm_video_recorder_plugin PRIVATE ${FLAGS})