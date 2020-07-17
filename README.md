For running val experiments you need the following things running:

NOVA:
    roscore
    roslaunch arm_video_recorder arm_video_recorder.launch
    roslaunch physical_robot_3d_rope_shim vicon.launch use_val:=true live:=true
    rosrun cdcpd_ros node _kinect_name:=kinect2_victor_head _left_tf_name:=left_gripper_tool _right_tf_name:=right_gripper_tool
    roslaunch link_bot_gazebo real_val_scene.launch use_val:=true
    roslaunch physical_robot_3d_rope_shim robot_shim.launch use_val:=true
    rviz

NOVA VM:
    rcnova && rosrun arm_or_robots ros_trajectory_forwarder.py _world_frame:="robot_root" _robot:="val"

Val:
    rcnova && roslaunch hdt_michigan_control joint_control_filter.launch fake_val:=false --screen

Loki:
    rcnova && roslaunch kinect2_calibration_files kinect2_bridge_victor_head.launch
