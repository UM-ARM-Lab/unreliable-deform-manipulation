#include "multi_link_bot_model_plugin.h"

#include <memory>

#include <geometry_msgs/Point.h>
#include <ignition/math/Vector3.hh>

namespace gazebo {

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
  auto pos_action_bind = boost::bind(&MultiLinkBotModelPlugin::OnAction, this, _1);
  auto pos_action_so = ros::SubscribeOptions::create<link_bot_gazebo::MultiLinkBotPositionAction>(
      "/multi_link_bot_position_action", 1, pos_action_bind, ros::VoidPtr(), &queue_);
  auto vel_action_bind = boost::bind(&MultiLinkBotModelPlugin::OnVelocityAction, this, _1);
  auto vel_action_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotVelocityAction>(
      "/link_bot_velocity_action", 1, vel_action_bind, ros::VoidPtr(), &queue_);
  auto action_mode_bind = boost::bind(&MultiLinkBotModelPlugin::OnActionMode, this, _1);
  auto action_mode_so = ros::SubscribeOptions::create<std_msgs::String>("/link_bot_action_mode", 1, action_mode_bind,
                                                                        ros::VoidPtr(), &queue_);
  auto config_bind = boost::bind(&MultiLinkBotModelPlugin::OnConfiguration, this, _1);
  auto config_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotConfiguration>(
      "/link_bot_configuration", 1, config_bind, ros::VoidPtr(), &queue_);
  auto state_bind = boost::bind(&MultiLinkBotModelPlugin::StateServiceCallback, this, _1, _2);
  auto service_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::LinkBotState>("/link_bot_state", state_bind,
                                                                                        ros::VoidPtr(), &queue_);

  joy_sub_ = ros_node_->subscribe(joy_so);
  action_sub_ = ros_node_->subscribe(pos_action_so);
  velocity_action_sub_ = ros_node_->subscribe(vel_action_so);
  action_mode_sub_ = ros_node_->subscribe(action_mode_so);
  config_sub_ = ros_node_->subscribe(config_so);
  state_service_ = ros_node_->advertiseService(service_so);

  ros_queue_thread_ = std::thread(std::bind(&MultiLinkBotModelPlugin::QueueThread, this));

  model_ = parent;

  {
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
  }

  auto const &gripper1_link_name = sdf->GetElement("gripper1_link")->Get<std::string>();
  gripper1_link_ = model_->GetLink(gripper1_link_name);

  if (sdf->HasElement("gripper2_link")) {
    auto const &gripper2_link_name = sdf->GetElement("gripper2_link")->Get<std::string>();
    gripper2_link_ = model_->GetLink(gripper2_link_name);
  }

  // TODO: make this a sdformat tag
  auto constexpr camera_name{"default::my_camera::link::my_camera"};
  auto const &sensor = sensors::get_sensor(camera_name);
  camera_sensor = std::dynamic_pointer_cast<sensors::CameraSensor>(sensor);
  if (!camera_sensor) {
    gzerr << "Failed to load camera: " << camera_name << '\n';
  }
  updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&MultiLinkBotModelPlugin::OnUpdate, this));
  postRenderConnection_ = event::Events::ConnectPostRender(std::bind(&MultiLinkBotModelPlugin::OnPostRender, this));
  constexpr auto max_integral{0};
  constexpr auto max_vel{0.15};
  gripper1_x_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel, -max_vel);
  gripper1_y_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel, -max_vel);
  gripper2_x_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel, -max_vel);
  gripper2_y_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel, -max_vel);

  constexpr auto max_vel_integral{1};
  constexpr auto max_force{50};
  gripper1_x_vel_pid_ =
      common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force, -max_force);
  gripper1_y_vel_pid_ =
      common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force, -max_force);
  gripper2_x_vel_pid_ =
      common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force, -max_force);
  gripper2_y_vel_pid_ =
      common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force, -max_force);

  // this won't be changing
  latest_image_.encoding = "rgb8";

  gzlog << "MultiLinkBot Model Plugin finished initializing!\n";
}

void MultiLinkBotModelPlugin::OnPostRender()
{
  if (camera_sensor and camera_sensor->LastUpdateTime() > common::Time::Zero) {
    // one byte per channel
    auto constexpr byte_depth = 1;
    auto constexpr num_channels = 3;
    auto const w = camera_sensor->ImageWidth();
    auto const h = camera_sensor->ImageHeight();
    auto const total_size_bytes = w * h * byte_depth * num_channels;
    auto const &sensor_image = camera_sensor->ImageData();
    latest_image_.width = w;
    latest_image_.height = h;
    latest_image_.step = w * byte_depth * num_channels;
    latest_image_.header.seq = image_sequence_number;
    auto const stamp = camera_sensor->LastUpdateTime();
    latest_image_.header.stamp.sec = stamp.sec;
    latest_image_.header.stamp.nsec = stamp.nsec;
    latest_image_.data.assign(sensor_image, sensor_image + total_size_bytes);
    image_sequence_number += 1;
    ready_ = true;
  }
  else {
    gzwarn << "no camera image available" << std::endl;
  }
}

void MultiLinkBotModelPlugin::OnUpdate()
{
  constexpr auto dt{0.001};

  if (mode == "velocity") {
    ignition::math::Vector3d force{};
    auto const gripper1_vel = gripper1_link_->WorldLinearVel();
    auto const gripper1_vel_error = gripper1_vel - gripper1_target_velocity_;
    force.X(gripper1_x_vel_pid_.Update(gripper1_vel_error.X(), dt));
    force.Y(gripper1_y_vel_pid_.Update(gripper1_vel_error.Y(), dt));
    gripper1_link_->AddForce(force);

    if (gripper2_link_) {
      auto const gripper2_vel = gripper2_link_->WorldLinearVel();
      auto const gripper2_vel_error = gripper2_vel - gripper2_target_velocity_;
      force.X(gripper2_x_vel_pid_.Update(gripper2_vel_error.X(), dt));
      force.Y(gripper2_y_vel_pid_.Update(gripper2_vel_error.Y(), dt));
      gripper2_link_->AddForce(force);
    }
  }
  else if (mode == "position") {
    ignition::math::Vector3d force{};
    // zero the Z component
    auto const gripper1_pos = [&]() {
      auto p = gripper1_link_->WorldPose().Pos();
      p.Z(0);
      return p;
    }();
    auto const gripper1_pos_error = gripper1_pos - gripper1_target_position_;

    gripper1_target_velocity_.X(gripper1_x_pos_pid_.Update(gripper1_pos_error.X(), dt));
    gripper1_target_velocity_.Y(gripper1_y_pos_pid_.Update(gripper1_pos_error.Y(), dt));

    auto const gripper1_vel = [&]() {
      auto v = gripper1_link_->WorldLinearVel();
      v.Z(0);
      return v;
    }();
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
  else if (mode == "disabled") {
    // do nothing!
  }
}

void MultiLinkBotModelPlugin::OnJoy(sensor_msgs::JoyConstPtr const msg)
{
  constexpr auto scale{2000.0 / 32768.0};
  gripper1_target_position_.X(gripper1_target_position_.X() - msg->axes[0] * scale);
  gripper1_target_position_.Y(gripper1_target_position_.Y() + msg->axes[1] * scale);
  gripper2_target_position_.X(gripper2_target_position_.X() + msg->axes[3] * scale);
  gripper2_target_position_.Y(gripper2_target_position_.Y() - msg->axes[4] * scale);
}

void MultiLinkBotModelPlugin::OnAction(link_bot_gazebo::MultiLinkBotPositionActionConstPtr const msg)
{
  gripper1_target_position_.X(msg->gripper1_pos.x);
  gripper1_target_position_.Y(msg->gripper1_pos.y);
  gripper2_target_position_.X(msg->gripper2_pos.x);
  gripper2_target_position_.Y(msg->gripper2_pos.y);
}

void MultiLinkBotModelPlugin::OnVelocityAction(link_bot_gazebo::LinkBotVelocityActionConstPtr msg)
{
  gripper1_target_velocity_.X(msg->gripper1_velocity.x);
  gripper1_target_velocity_.Y(msg->gripper1_velocity.y);
  gripper2_target_velocity_.X(msg->gripper2_velocity.x);
  gripper2_target_velocity_.Y(msg->gripper2_velocity.y);
}

void MultiLinkBotModelPlugin::OnActionMode(std_msgs::StringConstPtr msg) { mode = msg->data; }

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
  for (auto const &link : model_->GetLinks()) {
    geometry_msgs::Point pt;
    pt.x = link->WorldPose().Pos().X();
    pt.y = link->WorldPose().Pos().Y();
    res.points.emplace_back(pt);
    res.link_names.emplace_back(link->GetName());
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

  while (!ready_)
    ;

  // one byte per channel
  auto constexpr byte_depth = 1;
  auto constexpr num_channels = 3;
  auto const w = camera_sensor->ImageWidth();
  auto const h = camera_sensor->ImageHeight();
  auto const total_size_bytes = w * h * byte_depth * num_channels;
  res.camera_image = latest_image_;
  res.header.stamp = ros::Time::now();
  image_sequence_number += 1;

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
