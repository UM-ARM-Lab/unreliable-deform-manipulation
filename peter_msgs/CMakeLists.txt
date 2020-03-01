project(peter_msgs)
cmake_minimum_required(VERSION 3.5)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs sensor_msgs geometry_msgs message_generation)

catkin_python_setup()

add_message_files(
        FILES
        IntPoint2D.msg
        LinkBotAction.msg
        ModelPose.msg
        ModelsEnable.msg
        ModelsPoses.msg
        NamedPoint.msg
        NamedPoints.msg
        Objects.msg
        SubspaceDescription.msg
)

add_service_files(
        FILES
        ComputeOccupancy.srv
        ExecuteAction.srv
        GetObject.srv
        GetObjects.srv
        LinkBotReset.srv
        LinkBotState.srv
        LinkBotTrajectory.srv
        WorldControl.srv
        StateSpaceDescription.srv
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        sensor_msgs
        std_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)
