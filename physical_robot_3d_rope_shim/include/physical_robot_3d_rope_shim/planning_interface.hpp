#ifndef LBV_PLANNING_INTERFACE_HPP
#define LBV_PLANNING_INTERFACE_HPP

#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <arc_utilities/eigen_typedefs.hpp>
#include <memory>

#include "physical_robot_3d_rope_shim/moveit_pose_type.hpp"

using Matrix6Xd = Eigen::Matrix<double, 6, Eigen::Dynamic>;

class PlanningInterace
{
public:
  enum
  {
    NeedsToAlign = ((sizeof(Pose) % 16) == 0)
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)
  ros::NodeHandle nh_;
  ros::NodeHandle ph_;

  std::unique_ptr<robot_model_loader::RobotModelLoader> model_loader_;
  robot_model::RobotModelPtr model_;
  std::string const planning_group_;
  moveit::core::JointModelGroup const* const jmg_;
  size_t const num_ees_;
  std::vector<std::string> tool_names_;
  PoseSequence tool_offsets_;

  // Debugging
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  ros::Publisher vis_pub_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::string const world_frame_;
  std::string const robot_frame_;
  Pose const worldTrobot;
  Pose const robotTworld;

  // Home state data
  Eigen::VectorXd q_home_;
  robot_state::RobotState home_state_;
  PoseSequence home_state_tool_poses_;
  EigenHelpers::VectorQuaterniond robot_nominal_tool_orientations_;

  // For use when moving the EE positions using moveIn[Robot/World]Frame
  double const translation_step_size_;

  PlanningInterace(ros::NodeHandle nh, ros::NodeHandle ph, std::shared_ptr<tf2_ros::Buffer> tf_buffer,
                   std::string const& group);

  virtual Eigen::VectorXd lookupQHome() = 0;

  virtual void updateAllowedCollisionMatrix(collision_detection::AllowedCollisionMatrix& acm) = 0;

  void configureHomeState();

  PoseSequence getToolTransforms(robot_state::RobotState const& state) const;

  trajectory_msgs::JointTrajectory plan(planning_scene::PlanningScenePtr planning_scene,
                                        robot_state::RobotState const& goal_state);

  trajectory_msgs::JointTrajectory moveInRobotFrame(planning_scene::PlanningScenePtr planning_scene,
                                                    PointSequence const& target_tool_positions);

  trajectory_msgs::JointTrajectory moveInWorldFrame(planning_scene::PlanningScenePtr planning_scene,
                                                    PointSequence const& target_tool_positions);

  trajectory_msgs::JointTrajectory jacobianPath3d(planning_scene::PlanningScenePtr planning_scene,
                                                  std::vector<PointSequence> const& tool_paths);

  // Note that robot_goal_points is the target points for the tools, measured in robot frame
  bool jacobianIK(planning_scene::PlanningScenePtr planning_scene, PoseSequence const& robotTtargets);

  Eigen::MatrixXd getJacobianServoFrame(robot_state::RobotState const& state, PoseSequence const& robotTservo);

  Matrix6Xd getJacobianServoFrame(robot_state::RobotState const& state, robot_model::LinkModel const* link,
                                  Pose const& robotTservo);

protected:
  ////////////////////////////////////////////////////////////////////
  // Destructor that prevents "delete pointer to base object"
  ////////////////////////////////////////////////////////////////////

  ~PlanningInterace()
  {
  }
};

#endif // LBV_PLANNING_INTERFACE_HPP
