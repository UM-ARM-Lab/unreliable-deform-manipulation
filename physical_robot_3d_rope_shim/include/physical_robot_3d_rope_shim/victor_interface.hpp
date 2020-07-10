#ifndef VICTOR_INTERFACE_HPP
#define VICTOR_INTERFACE_HPP

#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <peter_msgs/DualGripperTrajectory.h>
#include <ros/ros.h>
#include <std_srvs/EmptyRequest.h>
#include <std_srvs/EmptyResponse.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Dense>
#include <mutex>
#include <string>

#include "physical_robot_3d_rope_shim/victor_manipulator.hpp"

class VictorInterface
{
public:
  using TrajectoryClient = actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>;

  enum
  {
    NeedsToAlign = ((sizeof(Pose) % 16) == 0)
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)
  ros::NodeHandle nh_;
  ros::NodeHandle ph_;

  // Debugging
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  ros::Publisher vis_pub_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::string const world_frame_;
  std::string const robot_frame_;
  std::string const table_frame_;
  std::string const left_tool_frame_;
  std::string const right_tool_frame_;
  Pose const worldTrobot;
  Pose const robotTworld;
  Pose const worldTtable;
  Pose const tableTworld;
  Pose const robotTtable;
  Pose const tableTrobot;
  Pose left_tool_offset_;   // TODO: should be const
  Pose right_tool_offset_;  // TODO: should be const

  std::unique_ptr<robot_model_loader::RobotModelLoader> model_loader_;
  robot_model::RobotModelPtr robot_model_;
  std::shared_ptr<VictorManipulator> left_arm_;
  std::shared_ptr<VictorManipulator> right_arm_;
  ros::Publisher talker_;

  std::shared_ptr<Listener<sensor_msgs::JointState>> joint_states_listener_;
  robot_state::RobotState home_state_;
  std::pair<Pose, Pose> home_state_tool_poses_world_frame_;
  std::unique_ptr<TrajectoryClient> trajectory_client_;
  ros::Duration const traj_goal_time_tolerance_;
  ros::ServiceClient obstacles_client_;
  ros::ServiceClient set_grasping_rope_client_;
  ros::ServiceClient world_control_client_;

  double const translation_step_size_;

  VictorInterface(ros::NodeHandle nh, ros::NodeHandle ph, std::shared_ptr<tf2_ros::Buffer> tf_buffer);

  void settle();

  void test();

  robot_state::RobotState getCurrentRobotState() const;
  std::pair<Pose, Pose> getToolTransforms() const;
  std::pair<Pose, Pose> getToolTransforms(robot_state::RobotState const& state) const;

  trajectory_msgs::JointTrajectory plan(robot_state::RobotState const& start_state,
                                        robot_state::RobotState const& goal_state);
  void followTrajectory(trajectory_msgs::JointTrajectory const& traj);
  void waitForNewState();
  void gotoHome();
  bool moveInRobotFrame(std::pair<Eigen::Translation3d, Eigen::Translation3d> const& target_gripper_positions);
  bool moveInWorldFrame(std::pair<Eigen::Translation3d, Eigen::Translation3d> const& target_gripper_positions);
};

#endif
