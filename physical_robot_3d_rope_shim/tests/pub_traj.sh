#!/usr/bin/env bash
rostopic pub /dual_arm_controller/command trajectory_msgs/JointTrajectory -f traj.msg
