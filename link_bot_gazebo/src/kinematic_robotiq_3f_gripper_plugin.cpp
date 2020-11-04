#include <ros/console.h>

#include <link_bot_gazebo/kinematic_robotiq_3f_gripper_plugin.h>
#include <link_bot_gazebo/mymath.hpp>
#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include <sensor_msgs/JointState.h>

constexpr static auto const PLUGIN_NAME = "kinematic_robotiq_3f_gripper";

namespace gazebo
{
GZ_REGISTER_MODEL_PLUGIN(KinematicRobotiq3fGripperPlugin)

KinematicRobotiq3fGripperPlugin::~KinematicRobotiq3fGripperPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  private_ros_node_->shutdown();
  ros_queue_thread_.join();
  private_ros_queue_thread_.join();
}

void KinematicRobotiq3fGripperPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
{
  model_ = model;

  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. ");
    return;
  }

  prefix_ = sdf->GetElement("prefix")->Get<std::string>();
  robot_namespace_ = sdf->GetElement("robotNamespace")->Get<std::string>();
  arm_name_ = sdf->GetElement("armName")->Get<std::string>();
  if (sdf->HasElement("rate"))
  {
    status_rate_ = sdf->GetElement("rate")->Get<double>();
  }

  finger_a_joint_ = GetJoint(PLUGIN_NAME, model_, prefix_ + "finger_1_joint_1");
  finger_b_joint_ = GetJoint(PLUGIN_NAME, model_, prefix_ + "finger_2_joint_1");
  finger_c_joint_ = GetJoint(PLUGIN_NAME, model_, prefix_ + "finger_middle_joint_1");

  CreateServices();

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
  private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

  auto update = [this](common::UpdateInfo const & /*info*/) { OnUpdate(); };
  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(update);

  auto periodic_update = [this]()
  {
    while (ros::ok())
    {
      ros::Rate(status_rate_).sleep();
      PeriodicUpdate();
    }
  };
  periodic_event_thread_ = std::thread(periodic_update);

  ROS_INFO("Finished loading kinematic robotiq 3f gripper plugin!");
}

void KinematicRobotiq3fGripperPlugin::CreateServices()
{
  private_ros_node_ = std::make_unique<ros::NodeHandle>("position_3d_plugin");

  auto gripper_command_topic_name = ros::names::append(robot_namespace_,
                                                       ros::names::append(arm_name_, "gripper_command"));
  command_sub_ = ros_node_.subscribe(gripper_command_topic_name, 10, &KinematicRobotiq3fGripperPlugin::OnCommand, this);
  auto gripper_status_topic_name = ros::names::append(robot_namespace_,
                                                      ros::names::append(arm_name_, "gripper_status"));
  status_pub_ = ros_node_.advertise<victor_hardware_interface_msgs::Robotiq3FingerStatus>(
      gripper_status_topic_name, 10);

  auto const joint_state_topic_name = ros::names::append(robot_namespace_, prefix_ + "gripper_joint_states");
  joint_state_pub_ = ros_node_.advertise<sensor_msgs::JointState>(joint_state_topic_name, 10);
}


void KinematicRobotiq3fGripperPlugin::OnUpdate()
{
}

void KinematicRobotiq3fGripperPlugin::PeriodicUpdate()
{
  // publish status msg
  victor_hardware_interface_msgs::Robotiq3FingerStatus status_msg;
  status_msg.header.stamp = ros::Time::now();
  status_msg.finger_a_status.header.stamp = ros::Time::now();
  // set position
  status_msg.finger_a_status.position = finger_a_joint_->Position(0);
  status_msg.finger_b_status.position = finger_b_joint_->Position(0);
  status_msg.finger_c_status.position = finger_c_joint_->Position(0);
  status_pub_.publish(status_msg);

  // publish joint states
  sensor_msgs::JointState joint_state_msg;
  joint_state_msg.header.stamp = ros::Time::now();
  for (auto const &j : model_->GetJoints())
  {
    if (j->GetName().rfind(prefix_, 0) == 0)
    {
      // joint name starts with the prefix
      ROS_DEBUG_STREAM_THROTTLE_NAMED(10, PLUGIN_NAME, "gripper " << prefix_
                                                              << " publishing joint " << j->GetName()
                                                              << " with position "
                                                              << j->Position(0));
      joint_state_msg.name.emplace_back(j->GetName());
      joint_state_msg.position.emplace_back(j->Position(0));
      joint_state_msg.velocity.emplace_back(j->GetVelocity(0));
    }
  }
  joint_state_pub_.publish(joint_state_msg);
}

void
KinematicRobotiq3fGripperPlugin::OnCommand(victor_hardware_interface_msgs::Robotiq3FingerCommandConstPtr const &msg)
{
  finger_a_joint_->SetPosition(0, msg->finger_a_command.position);
  finger_b_joint_->SetPosition(0, msg->finger_b_command.position);
  finger_c_joint_->SetPosition(0, msg->finger_c_command.position);
}


void KinematicRobotiq3fGripperPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void KinematicRobotiq3fGripperPlugin::PrivateQueueThread()
{
  double constexpr timeout = 0.01;
  while (private_ros_node_->ok())
  {
    private_queue_.callAvailable(ros::WallDuration(timeout));
  }
}

}  // namespace gazebo
