cmake_minimum_required(VERSION 3.5)
project(peter_msgs)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs sensor_msgs geometry_msgs trajectory_msgs message_generation)

catkin_python_setup()

add_message_files(
        FILES
        Action.msg
        IntPoint2D.msg
        LinkBotAction.msg
        ModelPose.msg
        ModelsEnable.msg
        ModelsPoses.msg
        NamedPoint.msg
        NamedPoints.msg
        Objects.msg
        SubspaceDescription.msg
        WheelSpeeds.msg
        Vector2.msg
        ReconstructionInputs.msg
)

add_service_files(
        FILES
        ComputeOccupancy.srv
        ExecuteAction.srv
        GetObject.srv
        GetObjects.srv
        GetPosition2D.srv
        LinkBotReset.srv
        LinkBotState.srv
        LinkBotTrajectory.srv
        ListObjects.srv
        Position2DEnable.srv
        Position2DAction.srv
        SetRopeConfiguration.srv
        StateSpaceDescription.srv
        Reconstruction.srv
        WorldControl.srv
        GetPosition3D.srv
        Position3DEnable.srv
        Position3DAction.srv
        ActionSpaceDescription.srv
        GetFloat32.srv
        JointTraj.srv
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        sensor_msgs
        std_msgs
        trajectory_msgs
)

catkin_package(CATKIN_DEPENDS geometry_msgs sensor_msgs std_msgs trajectory_msgs message_runtime)
