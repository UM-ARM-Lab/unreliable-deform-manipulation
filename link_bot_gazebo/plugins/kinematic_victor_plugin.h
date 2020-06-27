#pragma once

#include <actionlib/server/simple_action_server.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <victor_hardware_interface/MotionCommand.h>
#include <victor_hardware_interface/MotionStatus.h>
#include <victor_hardware_interface/Robotiq3FingerStatus.h>

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
  ~KinematicVictorPlugin() override;

  void Load(physics::ModelPtr parent, sdf::ElementPtr sdf) override;

  void FollowJointTrajectory(const TrajServer::GoalConstPtr &goal);

  void OnLeftArmMotionCommand(const victor_hardware_interface::MotionCommandConstPtr &msg);

  void OnRightArmMotionCommand(const victor_hardware_interface::MotionCommandConstPtr &msg);

  void PublishLeftArmMotionStatus();
  void PublishRightArmMotionStatus();

  void PublishLeftGripperStatus();
  void PublishRightGripperStatus();

private:
  void QueueThread();

  void PrivateQueueThread();

  void PeriodicUpdate();

  event::ConnectionPtr update_connection_;
  physics::ModelPtr model_;
  physics::WorldPtr world_;

  std::unique_ptr<ros::NodeHandle> private_ros_node_;
  ros::NodeHandle ros_node_;
  ros::CallbackQueue queue_;
  ros::CallbackQueue private_queue_;
  std::thread ros_queue_thread_;
  std::thread private_ros_queue_thread_;
  std::thread periodic_event_thread_;
  ros::Publisher joint_states_pub_;
  ros::Publisher left_arm_motion_status_pub_;
  ros::Publisher right_arm_motion_status_pub_;
  ros::Publisher left_gripper_status_pub_;
  ros::Publisher right_gripper_status_pub_;
  ros::Subscriber left_arm_motion_command_sub_;
  ros::Subscriber right_arm_motion_command_sub_;

  std::unique_ptr<TrajServer> follow_traj_server_;
  std::string left_flange_name_{ "victor::victor_left_arm_link_7" };
  std::string right_flange_name_{ "victor::victor_right_arm_link_7" };
  std::string gripper1_name_{ "link_bot::gripper1" };
  std::string gripper2_name_{ "link_bot::gripper2" };
  physics::LinkPtr left_flange_;
  physics::LinkPtr right_flange_;
  physics::LinkPtr gripper1_;
  physics::LinkPtr gripper2_;
  ignition::math::Pose3d left_flange_to_gripper1_;
  ignition::math::Pose3d right_flange_to_gripper2_;
};

}  // namespace gazebo
