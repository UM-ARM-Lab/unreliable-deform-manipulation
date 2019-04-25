#include "linkbot_model_plugin.h"

#include <ignition/math/Vector3.hh>
#include <memory>

namespace gazebo {

void LinkBotModelPlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf)
{
  // Make sure the ROS node for Gazebo has already been initalized
  // Initialize ros, if it has not already bee initialized.
  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "linkbot_model_plugin", ros::init_options::NoSigintHandler);
  }

  ros_node_ = std::make_unique<ros::NodeHandle>("linkbot_model_plugin");

  auto joy_bind = boost::bind(&LinkBotModelPlugin::OnJoy, this, _1);
  auto joy_so = ros::SubscribeOptions::create<sensor_msgs::Joy>("/joy", 1, joy_bind, ros::VoidPtr(), &queue_);
  auto vel_action_bind = boost::bind(&LinkBotModelPlugin::OnVelocityAction, this, _1);
  auto vel_action_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotVelocityAction>(
      "/link_bot_velocity_action", 1, vel_action_bind, ros::VoidPtr(), &queue_);
  auto force_action_bind = boost::bind(&LinkBotModelPlugin::OnForceAction, this, _1);
  auto force_action_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotForceAction>(
      "/link_bot_force_action", 1, force_action_bind, ros::VoidPtr(), &queue_);
  auto config_bind = boost::bind(&LinkBotModelPlugin::OnConfiguration, this, _1);
  auto config_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotConfiguration>(
      "/link_bot_configuration", 1, config_bind, ros::VoidPtr(), &queue_);
  joy_sub_ = ros_node_->subscribe(joy_so);
  vel_cmd_sub_ = ros_node_->subscribe(vel_action_so);
  force_cmd_sub_ = ros_node_->subscribe(force_action_so);
  config_sub_ = ros_node_->subscribe(config_so);
  ros_queue_thread_ = std::thread(std::bind(&LinkBotModelPlugin::QueueThread, this));

  if (!sdf->HasElement("kP")) {
    printf("using default kP=%f\n", kP_);
  }
  else {
    kP_ = sdf->GetElement("kP")->Get<double>();
  }

  if (!sdf->HasElement("kI")) {
    printf("using default kI=%f\n", kI_);
  }
  else {
    kI_ = sdf->GetElement("kI")->Get<double>();
  }

  if (!sdf->HasElement("kD")) {
    printf("using default kD=%f\n", kD_);
  }
  else {
    kD_ = sdf->GetElement("kD")->Get<double>();
  }

  if (!sdf->HasElement("action_scale")) {
    printf("using default action_scale=%f\n", action_scale);
  }
  else {
    action_scale = sdf->GetElement("action_scale")->Get<double>();
  }

  ROS_INFO("kP=%f, kI=%f, kD=%f", kP_, kI_, kD_);

  model_ = parent;

  updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&LinkBotModelPlugin::OnUpdate, this));
  x_vel_pid_ = common::PID(kP_, kI_, kD_, 100, -100, 800, -800);
  y_vel_pid_ = common::PID(kP_, kI_, kD_, 100, -100, 800, -800);
}

void LinkBotModelPlugin::OnUpdate()
{
  if (use_force_) {
    auto i{0u};
    auto const &links = model_->GetLinks();
    for (auto &link : links) {
      ignition::math::Vector3d force{};
      force.X(wrenches_[i].force.x);
      force.Y(wrenches_[i].force.y);
      link->AddForce(force);
      ++i;
    }
  }
  else {
    if (velocity_control_link_) {
      auto const current_linear_vel = velocity_control_link_->WorldLinearVel();
      auto const error = current_linear_vel - target_linear_vel_;
      ignition::math::Vector3d force{};
      force.X(x_vel_pid_.Update(error.X(), 0.001));
      force.Y(y_vel_pid_.Update(error.Y(), 0.001));
      velocity_control_link_->AddForce(force);
    }
  }
}

void LinkBotModelPlugin::OnJoy(sensor_msgs::JoyConstPtr const msg)
{
  use_force_ = false;
  velocity_control_link_ = model_->GetLink("head");
  if (not velocity_control_link_) {
    std::cout << "invalid link pointer. Link name "
              << "head"
              << " is not one of:\n";
    for (auto const &link : model_->GetLinks()) {
      std::cout << link->GetName() << "\n";
    }
    return;
  }
  target_linear_vel_.X(-msg->axes[0] * action_scale);
  target_linear_vel_.Y(msg->axes[1] * action_scale);
}

void LinkBotModelPlugin::OnVelocityAction(link_bot_gazebo::LinkBotVelocityActionConstPtr const msg)
{
  use_force_ = false;
  velocity_control_link_ = model_->GetLink(msg->control_link_name);
  if (not velocity_control_link_) {
    std::cout << "invalid link pointer. Link name " << msg->control_link_name << " is not one of:\n";
    for (auto const &link : model_->GetLinks()) {
      std::cout << link->GetName() << "\n";
    }
    return;
  }
  target_linear_vel_.X(msg->vx * action_scale);
  target_linear_vel_.Y(msg->vy * action_scale);
}

void LinkBotModelPlugin::OnForceAction(link_bot_gazebo::LinkBotForceActionConstPtr const msg)
{
  use_force_ = true;
  auto const &joints = model_->GetJoints();
  if (msg->wrenches.size() != joints.size()) {
    ROS_ERROR("Model as %lu joints config message had %lu", joints.size(), msg->wrenches.size());
    return;
  }

  wrenches_ = msg->wrenches;
}

void LinkBotModelPlugin::OnConfiguration(link_bot_gazebo::LinkBotConfigurationConstPtr msg)
{
  auto const &joints = model_->GetJoints();

  if (joints.size() != msg->joint_angles_rad.size()) {
    ROS_ERROR("Model as %lu joints config message had %lu", joints.size(), msg->joint_angles_rad.size());
    return;
  }

  ignition::math::Pose3d pose{};
  pose.Pos().X(msg->tail_pose.x);
  pose.Pos().Y(msg->tail_pose.y);
  pose.Pos().Z(0.05);
  pose.Rot() = ignition::math::Quaterniond::EulerToQuaternion(0, 0, msg->tail_pose.theta);
  model_->SetWorldPose(pose);
  model_->SetWorldTwist({0, 0, 0}, {0, 0, 0});

  for (size_t i = 0; i < joints.size(); ++i) {
    auto const &joint = joints[i];
    joint->SetPosition(0, msg->joint_angles_rad[i]);
    joint->SetVelocity(0, 0);
  }
}

void LinkBotModelPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_->ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

LinkBotModelPlugin::~LinkBotModelPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_->shutdown();
  ros_queue_thread_.join();
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(LinkBotModelPlugin)
}  // namespace gazebo
