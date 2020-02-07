cmake_minimum_required(VERSION 2.8.3)
project(link_bot_planning)

find_package(catkin REQUIRED COMPONENTS rospy visualization_msgs message_generation)

catkin_python_setup()

add_message_files(
        FILES
        PlannerData.msg
        LinkBotCompoundState.msg
)

generate_messages(
        DEPENDENCIES
        visualization_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)
