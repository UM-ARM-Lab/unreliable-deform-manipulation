#pragma once

#include <actionlib/server/simple_action_server.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <tf2_ros/transform_listener.h>
#include <victor_hardware_interface_msgs/MotionCommand.h>
#include <victor_hardware_interface_msgs/MotionStatus.h>
#include <victor_hardware_interface_msgs/Robotiq3FingerStatus.h>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <thread>

namespace gazebo
{
using TrajServer = actionlib::SimpleActionServer<control_msgs::FollowJointTrajectoryAction>;

class KinematicVictorPlugin : public ModelPlugin
{
public:
  KinematicVictorPlugin();
  ~KinematicVictorPlugin() override;

  void Load(physics::ModelPtr parent, sdf::ElementPtr sdf) override;

  sensor_msgs::JointState GetJointStates();
  void PublishJointStates();
  void PublishLeftArmMotionStatus();
  void PublishRightArmMotionStatus();
  void PublishLeftGripperStatus();
  void PublishRightGripperStatus();

  void OnLeftArmMotionCommand(const victor_hardware_interface_msgs::MotionCommandConstPtr &msg);
  void OnRightArmMotionCommand(const victor_hardware_interface_msgs::MotionCommandConstPtr &msg);
  void FollowJointTrajectory(const TrajServer::GoalConstPtr &goal);

  void TeleportGrippers();

private:
  void QueueThread();
  void PrivateQueueThread();
  void PeriodicUpdate();

  event::ConnectionPtr update_connection_;
  physics::ModelPtr model_;
  physics::WorldPtr world_;

  // Protects against multiple ROS callbacks or publishers accessing/changing data out of order
  std::mutex ros_mutex_;
  std::unique_ptr<ros::NodeHandle> private_ros_node_;
  ros::NodeHandle ros_node_;
  ros::CallbackQueue queue_;
  ros::CallbackQueue private_queue_;
  std::thread ros_queue_thread_;
  std::thread private_ros_queue_thread_;
  std::thread periodic_event_thread_;
  std::vector<std::string> joint_names_;

  // pretty generic robot stuff
  ros::Publisher joint_states_pub_;
  ros::ServiceClient set_dual_gripper_points_srv_;
  ros::ServiceClient rope_overstretched_srv_;
  ros::ServiceServer joint_state_server_;
  std::unique_ptr<TrajServer> follow_traj_server_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // A task specific thing, but many robots could implement this.
  // here this just sets a flag so we don't teleport the rope to match the grippers
  ros::ServiceServer grasping_rope_server_;
  ros::ServiceServer ignore_overstretching_server_;

  // mocking Victor specific things, so that we can use the ARM Command Gui
  ros::Publisher left_arm_motion_status_pub_;
  ros::Publisher right_arm_motion_status_pub_;
  ros::Publisher left_gripper_status_pub_;
  ros::Publisher right_gripper_status_pub_;
  ros::Subscriber left_arm_motion_command_sub_;
  ros::Subscriber right_arm_motion_command_sub_;

  // Rope-overstretching-detection
  std::string const left_flange_name_{ "victor::victor_left_arm_link_7" };
  std::string const right_flange_name_{ "victor::victor_right_arm_link_7" };
  std::string const left_flange_tf_name_{ "victor_left_arm_link_7" };
  std::string const right_flange_tf_name_{ "victor_right_arm_link_7" };
  std::string const gripper1_tf_name_{ "left_ee_tool" };
  std::string const gripper2_tf_name_{ "right_ee_tool" };
  physics::LinkPtr left_flange_;
  physics::LinkPtr right_flange_;
  physics::LinkPtr gripper1_;
  physics::LinkPtr gripper2_;
  physics::LinkPtr gripper1_rope_link_;
  physics::LinkPtr gripper2_rope_link_;
  double max_dist_between_gripper_and_link_;
  ignition::math::Pose3d left_flange_to_gripper1_;
  ignition::math::Pose3d right_flange_to_gripper2_;
  bool grasping_rope_{ true };
  bool ignore_overstretching_{ false };
};

}  // namespace gazebo
