cmake_minimum_required(VERSION 2.8.3)
project(link_bot_sdf_tools)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs geometry_msgs message_generation)


catkin_python_setup()

add_service_files(
        FILES
        ComputeSDF.srv
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        sdf_tools
)

catkin_package(CATKIN_DEPENDS message_runtime)

