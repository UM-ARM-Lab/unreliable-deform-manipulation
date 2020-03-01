project(peter_msgs)
cmake_minimum_required(VERSION 3.5)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs sensor_msgs geometry_msgs message_generation)

catkin_python_setup()

add_message_files(
        FILES
        LinkBotConfiguration.msg
        LinkBotJointConfiguration.msg
        WriteSDF.msg
        ModelPose.msg
        ModelsEnable.msg
        ModelsPoses.msg
        IntPoint2D.msg
        NamedPoint.msg
        NamedPoints.msg
        Objects.msg
        LinkBotAction.msg
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
        LinkBotTrajectory.srv
        ComputeOccupancy.srv
        GetObjects.srv
        GetObject.srv
        ExecuteAction.srv
        LinkBotReset.srv
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        sensor_msgs
        visualization_msgs
        std_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)
