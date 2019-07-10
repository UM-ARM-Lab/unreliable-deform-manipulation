#include "multi_link_bot_model_plugin.h"

#include <exception>
#include <memory>

#include <geometry_msgs/Point.h>
#include <ignition/math/Vector3.hh>

namespace gazebo {

bool in_contact(msgs::Contacts const &contacts)
{
  for (auto i{0u}; i < contacts.contact_size(); ++i) {
    auto const &contact = contacts.contact(i);
    std::cout << contact.collision1() << " " << contact.collision2() << '\n';
    if (contact.collision1() == "link_bot::head::head_collision" and
        contact.collision2() != "ground_plane::link::collision") {
      return true;
    }
    if (contact.collision2() == "link_bot::head::head_collision" and
        contact.collision1() != "ground_plane::link::collision") {
      return true;
    }
  }
  return false;
}

void MultiLinkBotModelPlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf)
{
  // Make sure the ROS node for Gazebo has already been initalized
  // Initialize ros, if it has not already bee initialized.
  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "multi_link_bot_model_plugin", ros::init_options::NoSigintHandler);
  }

  ros_node_ = std::make_unique<ros::NodeHandle>("multi_link_bot_model_plugin");

  auto joy_bind = boost::bind(&MultiLinkBotModelPlugin::OnJoy, this, _1);
  auto joy_so = ros::SubscribeOptions::create<sensor_msgs::Joy>("/joy", 1, joy_bind, ros::VoidPtr(), &queue_);
  auto vel_action_bind = boost::bind(&MultiLinkBotModelPlugin::OnAction, this, _1);
  auto vel_action_so = ros::SubscribeOptions::create<link_bot_gazebo::MultiLinkBotPositionAction>(
      "/multi_link_bot_position_action", 1, vel_action_bind, ros::VoidPtr(), &queue_);
  auto config_bind = boost::bind(&MultiLinkBotModelPlugin::OnConfiguration, this, _1);
  auto config_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotConfiguration>(
      "/link_bot_configuration", 1, config_bind, ros::VoidPtr(), &queue_);
  auto state_bind = boost::bind(&MultiLinkBotModelPlugin::StateServiceCallback, this, _1, _2);
  auto service_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::LinkBotState>("/link_bot_state", state_bind,
                                                                                        ros::VoidPtr(), &queue_);

  joy_sub_ = ros_node_->subscribe(joy_so);
  action_sub_ = ros_node_->subscribe(vel_action_so);
  config_sub_ = ros_node_->subscribe(config_so);
  state_service_ = ros_node_->advertiseService(service_so);

  ros_queue_thread_ = std::thread(std::bind(&MultiLinkBotModelPlugin::QueueThread, this));

  model_ = parent;

  if (!sdf->HasElement("kP_pos")) {
    printf("using default kP_pos=%f\n", kP_pos_);
  }
  else {
    kP_pos_ = sdf->GetElement("kP_pos")->Get<double>();
  }

  if (!sdf->HasElement("kI_pos")) {
    printf("using default kI_pos=%f\n", kI_pos_);
  }
  else {
    kI_pos_ = sdf->GetElement("kI_pos")->Get<double>();
  }

  if (!sdf->HasElement("kD_pos")) {
    printf("using default kD_pos=%f\n", kD_pos_);
  }
  else {
    kD_pos_ = sdf->GetElement("kD_pos")->Get<double>();
  }

  if (!sdf->HasElement("kP_vel")) {
    printf("using default kP_vel=%f\n", kP_vel_);
  }
  else {
    kP_vel_ = sdf->GetElement("kP_vel")->Get<double>();
  }

  if (!sdf->HasElement("kI_vel")) {
    printf("using default kI_vel=%f\n", kI_vel_);
  }
  else {
    kI_vel_ = sdf->GetElement("kI_vel")->Get<double>();
  }

  if (!sdf->HasElement("kD_vel")) {
    printf("using default kD_vel=%f\n", kD_vel_);
  }
  else {
    kD_vel_ = sdf->GetElement("kD_vel")->Get<double>();
  }

  if (!sdf->HasElement("gripper1_link")) {
    throw std::invalid_argument("no gripper1_link tag provided");
  }

  auto const &gripper1_link_name = sdf->GetElement("gripper1_link")->Get<std::string>();
  gripper1_link_ = model_->GetLink(gripper1_link_name);

  if (sdf->HasElement("gripper2_link")) {
    auto const &gripper2_link_name = sdf->GetElement("gripper2_link")->Get<std::string>();
    gripper2_link_ = model_->GetLink(gripper2_link_name);
  }

  updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&MultiLinkBotModelPlugin::OnUpdate, this));
  constexpr auto max_force{200};
  constexpr auto max_vel{0.30};
  constexpr auto max_vel_integral{1};
  constexpr auto max_integral{100};
  gripper1_x_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel, -max_vel);
  gripper1_y_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel, -max_vel);
  gripper2_x_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel, -max_vel);
  gripper2_y_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel, -max_vel);

  gripper1_x_vel_pid_ = common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force, -max_force);
  gripper1_y_vel_pid_ = common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force, -max_force);
  gripper2_x_vel_pid_ = common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force, -max_force);
  gripper2_y_vel_pid_ = common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force, -max_force);

  std::cout << "MultiLinkBot Model Plugin finished initializing!\n";
}

void MultiLinkBotModelPlugin::OnUpdate()
{
  constexpr auto dt{0.001};

  ignition::math::Vector3d force{};
  auto const gripper1_pos = gripper1_link_->WorldPose().Pos();
  auto const gripper1_pos_error = gripper1_pos - gripper1_target_position_;

  gripper1_target_velocity_.X(gripper1_x_pos_pid_.Update(gripper1_pos_error.X(), dt));
  gripper1_target_velocity_.Y(gripper1_y_pos_pid_.Update(gripper1_pos_error.Y(), dt));

  auto const gripper1_vel = gripper1_link_->WorldLinearVel();
  auto const gripper1_vel_error = gripper1_vel - gripper1_target_velocity_;
  force.X(gripper1_x_vel_pid_.Update(gripper1_vel_error.X(), dt));
  force.Y(gripper1_y_vel_pid_.Update(gripper1_vel_error.Y(), dt));
  gripper1_link_->AddForce(force);

  if (gripper2_link_) {
    auto const gripper2_pos = gripper2_link_->WorldPose().Pos();
    auto const gripper2_pos_error = gripper2_pos - gripper2_target_position_;

    gripper2_target_velocity_.X(gripper2_x_pos_pid_.Update(gripper2_pos_error.X(), dt));
    gripper2_target_velocity_.Y(gripper2_y_pos_pid_.Update(gripper2_pos_error.Y(), dt));

    auto const gripper2_vel = gripper2_link_->WorldLinearVel();
    auto const gripper2_vel_error = gripper2_vel - gripper2_target_velocity_;
    force.X(gripper2_x_vel_pid_.Update(gripper2_vel_error.X(), dt));
    force.Y(gripper2_y_vel_pid_.Update(gripper2_vel_error.Y(), dt));
    gripper2_link_->AddForce(force);

  }
}

void MultiLinkBotModelPlugin::OnJoy(sensor_msgs::JoyConstPtr const msg)
{
  constexpr auto scale{2000.0 / 32768.0};
  gripper1_target_position_.X(gripper1_target_position_.X() - msg->axes[0] * scale);
  gripper1_target_position_.Y(gripper1_target_position_.Y() + msg->axes[1] * scale);
  gripper2_target_position_.X(gripper2_target_position_.X() + msg->axes[3] * scale);
  gripper2_target_position_.Y(gripper2_target_position_.Y() -msg->axes[4] * scale);
}

void MultiLinkBotModelPlugin::OnAction(link_bot_gazebo::MultiLinkBotPositionActionConstPtr const msg)
{
  gripper1_target_position_.X(msg->gripper1_pos.x);
  gripper1_target_position_.Y(msg->gripper1_pos.y);
  gripper2_target_position_.X(msg->gripper2_pos.x);
  gripper2_target_position_.Y(msg->gripper2_pos.y);
}

void MultiLinkBotModelPlugin::OnConfiguration(link_bot_gazebo::LinkBotConfigurationConstPtr msg)
{
  auto const &joints = model_->GetJoints();

  if (joints.size() != msg->joint_angles_rad.size()) {
    ROS_ERROR("Model as %lu joints config message had %lu", joints.size(), msg->joint_angles_rad.size());
    return;
  }

  ignition::math::Pose3d pose{};
  pose.Pos().X(msg->tail_pose.x);
  pose.Pos().Y(msg->tail_pose.y);
  pose.Pos().Z(0.01);
  pose.Rot() = ignition::math::Quaterniond::EulerToQuaternion(0, 0, msg->tail_pose.theta);
  model_->SetWorldPose(pose);
  model_->SetWorldTwist({0, 0, 0}, {0, 0, 0});

  for (size_t i = 0; i < joints.size(); ++i) {
    auto const &joint = joints[i];
    joint->SetPosition(0, msg->joint_angles_rad[i]);
    joint->SetVelocity(0, 0);
  }
}

bool MultiLinkBotModelPlugin::StateServiceCallback(link_bot_gazebo::LinkBotStateRequest &req,
                                                   link_bot_gazebo::LinkBotStateResponse &res)
{
  for (auto const &link : model_->GetLinks())
  {
    geometry_msgs::Point pt;
    pt.x = link->WorldPose().Pos().X();
    pt.y = link->WorldPose().Pos().Y();
    res.points.emplace_back(pt);
  }

  res.gripper1_force.x = gripper1_x_vel_pid_.GetCmd();
  res.gripper1_force.y = gripper1_y_vel_pid_.GetCmd();
  res.gripper1_force.z = 0;

  auto const gripper1_velocity = gripper1_link_->WorldLinearVel();
  res.gripper1_velocity.x = gripper1_velocity.X();
  res.gripper1_velocity.y = gripper1_velocity.Y();
  res.gripper1_velocity.z = 0;

  res.gripper1_target_velocity.x = gripper1_target_velocity_.X();
  res.gripper1_target_velocity.y = gripper1_target_velocity_.Y();
  res.gripper1_target_velocity.z = gripper1_target_velocity_.Z();

  if (gripper2_link_) {
    auto const gripper2_velocity = gripper2_link_->WorldLinearVel();
    res.gripper2_velocity.x = gripper2_velocity.X();
    res.gripper2_velocity.y = gripper2_velocity.Y();
    res.gripper2_velocity.z = 0;

    res.gripper2_force.x = gripper2_x_vel_pid_.GetCmd();
    res.gripper2_force.y = gripper2_y_vel_pid_.GetCmd();
    res.gripper2_force.z = 0;

    res.gripper2_target_velocity.x = gripper2_target_velocity_.X();
    res.gripper2_target_velocity.y = gripper2_target_velocity_.Y();
    res.gripper2_target_velocity.z = gripper2_target_velocity_.Z();
  }

  return true;
}

void MultiLinkBotModelPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_->ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

MultiLinkBotModelPlugin::~MultiLinkBotModelPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_->shutdown();
  ros_queue_thread_.join();
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(MultiLinkBotModelPlugin)
}  // namespace gazebo
