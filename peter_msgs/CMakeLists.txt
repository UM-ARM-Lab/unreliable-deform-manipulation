cmake_minimum_required(VERSION 3.5)
project(peter_msgs)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs sensor_msgs geometry_msgs trajectory_msgs message_generation)

catkin_python_setup()

add_message_files(
        FILES
        Action.msg
        Points.msg
        IntPoint2D.msg
        LabelStatus.msg
        LinkBotAction.msg
        ModelPose.msg
        ModelsEnable.msg
        ModelsPoses.msg
        NamedPoint.msg
        NamedPoints.msg
        ReconstructionInputs.msg
        SubspaceDescription.msg
        Vector2.msg
        WheelSpeeds.msg
        AnimationControl.msg
)

add_service_files(
        FILES
        ComputeOccupancy.srv
        GrippersTrajectory.srv
        ExecuteAction.srv
        GetDualGripperPoints.srv
        GetBool.srv
        GetFloat32.srv
        GetPosition3D.srv
        GetRopeState.srv
        GetJointState.srv
        LinkBotReset.srv
        LinkBotState.srv
        LinkBotTrajectory.srv
        Position3DAction.srv
        Position3DDeltaAction.srv
        Position3DEnable.srv
        Reconstruction.srv
        SetRopeState.srv
        SetDualGripperPoints.srv
        StateSpaceDescription.srv
        WorldControl.srv
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        sensor_msgs
        std_msgs
        trajectory_msgs)


catkin_package(CATKIN_DEPENDS geometry_msgs sensor_msgs std_msgs trajectory_msgs message_runtime)
