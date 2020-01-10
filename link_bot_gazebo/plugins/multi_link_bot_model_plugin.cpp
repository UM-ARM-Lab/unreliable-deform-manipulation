#include "multi_link_bot_model_plugin.h"

#include <memory>
#include <sstream>

#include <geometry_msgs/Point.h>
#include <gazebo/common/Time.hh>
#include <gazebo/common/Timer.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo {

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-vararg"
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
  auto execute_path_bind = boost::bind(&MultiLinkBotModelPlugin::ExecutePathCallback, this, _1, _2);
  auto execute_path_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::LinkBotPath>(
      "/link_bot_execute_path", execute_path_bind, ros::VoidPtr(), &queue_);
  auto execute_trajectory_bind = boost::bind(&MultiLinkBotModelPlugin::ExecuteTrajectoryCallback, this, _1, _2);
  auto execute_trajectory_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::LinkBotTrajectory>(
      "/link_bot_execute_trajectory", execute_trajectory_bind, ros::VoidPtr(), &queue_);
  auto pos_action_bind = boost::bind(&MultiLinkBotModelPlugin::OnPositionAction, this, _1, _2);
  auto pos_action_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::LinkBotPositionAction>(
      "/link_bot_position_action", pos_action_bind, ros::VoidPtr(), &queue_);
  auto vel_action_bind = boost::bind(&MultiLinkBotModelPlugin::OnVelocityAction, this, _1);
  auto vel_action_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotVelocityAction>(
      "/link_bot_velocity_action", 1, vel_action_bind, ros::VoidPtr(), &queue_);
  auto action_mode_bind = boost::bind(&MultiLinkBotModelPlugin::OnActionMode, this, _1);
  auto action_mode_so = ros::SubscribeOptions::create<std_msgs::String>("/link_bot_action_mode", 1, action_mode_bind,
                                                                        ros::VoidPtr(), &queue_);
  auto config_bind = boost::bind(&MultiLinkBotModelPlugin::OnConfiguration, this, _1);
  auto config_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotJointConfiguration>(
      "/link_bot_configuration", 1, config_bind, ros::VoidPtr(), &queue_);
  auto state_bind = boost::bind(&MultiLinkBotModelPlugin::StateServiceCallback, this, _1, _2);
  auto service_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::LinkBotState>("/link_bot_state", state_bind,
                                                                                        ros::VoidPtr(), &queue_);

  joy_sub_ = ros_node_->subscribe(joy_so);
  velocity_action_sub_ = ros_node_->subscribe(vel_action_so);
  action_mode_sub_ = ros_node_->subscribe(action_mode_so);
  config_sub_ = ros_node_->subscribe(config_so);
  state_service_ = ros_node_->advertiseService(service_so);
  pos_action_service_ = ros_node_->advertiseService(pos_action_so);
  execute_path_service_ = ros_node_->advertiseService(execute_path_so);
  execute_traj_service_ = ros_node_->advertiseService(execute_trajectory_so);

  ros_queue_thread_ = std::thread(std::bind(&MultiLinkBotModelPlugin::QueueThread, this));

  model_ = parent;

  {
    if (!sdf->HasElement("length")) {
      printf("using default length=%f\n", length_);
    }
    else {
      length_ = sdf->GetElement("length")->Get<double>();
    }

    if (!sdf->HasElement("num_links")) {
      printf("using default num_links=%u\n", num_links_);
    }
    else {
      num_links_ = sdf->GetElement("num_links")->Get<unsigned int>();
    }

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

    if (!sdf->HasElement("max_force")) {
      printf("using default max_force=%f\n", max_force_);
    }
    else {
      max_force_ = sdf->GetElement("max_force")->Get<double>();
    }

    if (!sdf->HasElement("max_vel")) {
      printf("using default max_vel=%f\n", max_vel_);
    }
    else {
      max_vel_ = sdf->GetElement("max_vel")->Get<double>();
    }

    if (!sdf->HasElement("max_vel_acc")) {
      printf("using default max_vel_acc=%f\n", max_vel_acc_);
    }
    else {
      max_vel_acc_ = sdf->GetElement("max_vel_acc")->Get<double>();
    }
  }

  // plus 1 because we want both end points inclusive
  ros_node_->setParam("/link_bot/n_state", static_cast<int>((num_links_ + 1) * 2));
  ros_node_->setParam("/link_bot/rope_length", length_);

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
  gripper1_x_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);
  gripper1_y_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);
  gripper2_x_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);
  gripper2_y_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);

  constexpr auto max_vel_integral{1};
  gripper1_x_vel_pid_ =
      common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force_, -max_force_);
  gripper1_y_vel_pid_ =
      common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force_, -max_force_);
  gripper2_x_vel_pid_ =
      common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force_, -max_force_);
  gripper2_y_vel_pid_ =
      common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force_, -max_force_);

  // this won't be changing
  latest_image_.encoding = "rgb8";

  gzlog << "MultiLinkBot Model Plugin finished initializing!\n";
}
#pragma clang diagnostic pop

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
    latest_image_.data.assign(
        sensor_image, sensor_image + total_size_bytes);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    image_sequence_number += 1;
    ready_ = true;
  }
  else {
    //    gzwarn << "no camera image available" << std::endl;
  }
}

double update_target(double const current_target, double const target, double const max_acc)
{
  auto const delta = target - current_target;
  auto const clamped_acc = std::max(std::min(max_acc, delta), -max_acc);
  auto const new_current_target = current_target + clamped_acc;
  return new_current_target;
}

link_bot_gazebo::LinkBotConfiguration MultiLinkBotModelPlugin::GetConfiguration()
{
  link_bot_gazebo::LinkBotConfiguration configuration;
  for (auto link_idx{1U}; link_idx <= num_links_; ++link_idx)
  {
    std::stringstream ss;
    ss << "link_" << link_idx;
    auto link_name = ss.str();
    auto const tail = model_->GetLink(link_name);
    geometry_msgs::Point point;
    point.x = tail->WorldPose().Pos().X();
    point.y = tail->WorldPose().Pos().Y();
    configuration.points.emplace_back(point);
  }
  return configuration;
}

ControlResult MultiLinkBotModelPlugin::UpdateControl()
{
  std::lock_guard<std::mutex> guard(control_mutex_);
  constexpr auto dt{0.001};
  ControlResult control_result{};

  // zero the Z component
  auto const gripper1_pos = [&]() {
    auto p = gripper1_link_->WorldPose().Pos();
    p.Z(0);
    return p;
  }();

  auto const gripper1_vel = [&]() {
    auto v = gripper1_link_->WorldLinearVel();
    v.Z(0);
    return v;
  }();

  control_result.configuration = GetConfiguration();

  // Gripper 1
  {
    if (mode_ == "velocity") {
      gripper1_current_target_velocity_.X(
          update_target(gripper1_current_target_velocity_.X(), gripper1_target_velocity_.X(), max_vel_acc_));
      gripper1_current_target_velocity_.Y(
          update_target(gripper1_current_target_velocity_.Y(), gripper1_target_velocity_.Y(), max_vel_acc_));

      auto const gripper1_vel_error = gripper1_vel - gripper1_current_target_velocity_;
      control_result.gripper1_force.X(gripper1_x_vel_pid_.Update(gripper1_vel_error.X(), dt));
      control_result.gripper1_force.Y(gripper1_y_vel_pid_.Update(gripper1_vel_error.Y(), dt));
    }
    else if (mode_ == "position") {
      gripper1_pos_error_ = gripper1_pos - gripper1_target_position_;

      gripper1_target_velocity_.X(gripper1_x_pos_pid_.Update(gripper1_pos_error_.X(), dt));
      gripper1_target_velocity_.Y(gripper1_y_pos_pid_.Update(gripper1_pos_error_.Y(), dt));

      gripper1_current_target_velocity_.X(
          update_target(gripper1_current_target_velocity_.X(), gripper1_target_velocity_.X(), max_vel_acc_));
      gripper1_current_target_velocity_.Y(
          update_target(gripper1_current_target_velocity_.Y(), gripper1_target_velocity_.Y(), max_vel_acc_));
      control_result.gripper1_vel = gripper1_current_target_velocity_;

      auto const gripper1_vel_error = gripper1_vel - gripper1_current_target_velocity_;
      control_result.gripper1_force.X(gripper1_x_vel_pid_.Update(gripper1_vel_error.X(), dt));
      control_result.gripper1_force.Y(gripper1_y_vel_pid_.Update(gripper1_vel_error.Y(), dt));
    }
    else if (mode_ == "disabled") {
      // do nothing!
    }
  }

  // Gripper 2
  if (gripper2_link_) {
    // zero the Z component
    auto const gripper2_pos = [&]() {
      auto p = gripper2_link_->WorldPose().Pos();
      p.Z(0);
      return p;
    }();

    auto const gripper2_vel = [&]() {
      auto v = gripper2_link_->WorldLinearVel();
      v.Z(0);
      return v;
    }();

    if (mode_ == "velocity") {
      auto const gripper2_vel_error = gripper2_vel - gripper2_target_velocity_;
      control_result.gripper2_force.X(gripper2_x_vel_pid_.Update(gripper2_vel_error.X(), dt));
      control_result.gripper2_force.Y(gripper2_y_vel_pid_.Update(gripper2_vel_error.Y(), dt));
    }
    else if (mode_ == "position") {
      gripper2_pos_error_ = gripper2_pos - gripper2_target_position_;

      gripper2_target_velocity_.X(gripper2_x_pos_pid_.Update(gripper2_pos_error_.X(), dt));
      gripper2_target_velocity_.Y(gripper2_y_pos_pid_.Update(gripper2_pos_error_.Y(), dt));
      control_result.gripper2_vel = gripper2_target_velocity_;

      auto const gripper2_vel_error = gripper2_vel - gripper2_target_velocity_;
      control_result.gripper2_force.X(gripper2_x_vel_pid_.Update(gripper2_vel_error.X(), dt));
      control_result.gripper2_force.Y(gripper2_y_vel_pid_.Update(gripper2_vel_error.Y(), dt));
    }
    else if (mode_ == "disabled") {
      // do nothing!
    }
  }

  return control_result;
}

void MultiLinkBotModelPlugin::OnUpdate()
{
  ControlResult control = UpdateControl();

  gripper1_link_->AddForce(control.gripper1_force);
  if (gripper2_link_) {
    gripper2_link_->AddForce(control.gripper2_force);
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

bool MultiLinkBotModelPlugin::OnPositionAction(link_bot_gazebo::LinkBotPositionActionRequest &req,
                                               link_bot_gazebo::LinkBotPositionActionResponse &res)
{
  gripper1_target_position_.X(req.gripper1_pos.x);
  gripper1_target_position_.Y(req.gripper1_pos.y);
  gripper2_target_position_.X(req.gripper2_pos.x);
  gripper2_target_position_.Y(req.gripper2_pos.y);

  gazebo::ControlResult control = UpdateControl();

  res.gripper1_target_velocity.x = control.gripper1_vel.X();
  res.gripper1_target_velocity.y = control.gripper1_vel.Y();
  res.gripper1_target_velocity.z = control.gripper1_vel.Z();
  res.gripper2_target_velocity.x = control.gripper2_vel.X();
  res.gripper2_target_velocity.y = control.gripper2_vel.Y();
  res.gripper2_target_velocity.z = control.gripper2_vel.Z();

  return true;
}

void MultiLinkBotModelPlugin::OnVelocityAction(link_bot_gazebo::LinkBotVelocityActionConstPtr msg)
{
  gripper1_target_velocity_.X(msg->gripper1_velocity.x);
  gripper1_target_velocity_.Y(msg->gripper1_velocity.y);
  gripper2_target_velocity_.X(msg->gripper2_velocity.x);
  gripper2_target_velocity_.Y(msg->gripper2_velocity.y);
}

void MultiLinkBotModelPlugin::OnActionMode(std_msgs::StringConstPtr msg) { mode_ = msg->data; }

void MultiLinkBotModelPlugin::OnConfiguration(link_bot_gazebo::LinkBotJointConfigurationConstPtr msg)
{
  auto const &joints = model_->GetJoints();

  if (joints.size() != msg->joint_angles_rad.size()) {
    ROS_ERROR("Model as %lu joints config message had %lu", joints.size(), msg->joint_angles_rad.size());
    return;
  }

  ignition::math::Pose3d pose{};
  pose.Pos().X(msg->tail_pose.x);
  pose.Pos().Y(msg->tail_pose.y);
  pose.Pos().Z(HEAD_Z);
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

  if (gripper2_link_) {
    auto const gripper2_velocity = gripper2_link_->WorldLinearVel();
    res.gripper2_velocity.x = gripper2_velocity.X();
    res.gripper2_velocity.y = gripper2_velocity.Y();
    res.gripper2_velocity.z = 0;

    res.gripper2_force.x = gripper2_x_vel_pid_.GetCmd();
    res.gripper2_force.y = gripper2_y_vel_pid_.GetCmd();
    res.gripper2_force.z = 0;
  }

  while (!ready_) {
  }

  // one byte per channel
  auto constexpr byte_depth = 1;
  auto constexpr num_channels = 3;
  auto const w = camera_sensor->ImageWidth();
  auto const h = camera_sensor->ImageHeight();
  res.camera_image = latest_image_;
  res.header.stamp = ros::Time::now();
  image_sequence_number += 1;

  return true;
}

bool MultiLinkBotModelPlugin::ExecuteTrajectoryCallback(link_bot_gazebo::LinkBotTrajectoryRequest &req,
                                                        link_bot_gazebo::LinkBotTrajectoryResponse &res)
{
  // TODO: Implement gripper1 path in parallel
  for (auto const &target_velocity : req.gripper1_traj) {
    mode_ = "velocity";
    gripper1_target_velocity_.X(target_velocity.gripper1_velocity.x);
    gripper1_target_velocity_.Y(target_velocity.gripper1_velocity.y);

    auto control_result = UpdateControl();
    res.actual_path.emplace_back(control_result.configuration);

    auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
    auto const steps = static_cast<unsigned int>(req.dt / seconds_per_step);
    // Wait until the setpoint is reached
    model_->GetWorld()->Step(steps);
  }

  auto const final_configuration = GetConfiguration();
  res.actual_path.emplace_back(final_configuration);

  return true;
}

bool MultiLinkBotModelPlugin::ExecutePathCallback(link_bot_gazebo::LinkBotPathRequest &req,
                                                  link_bot_gazebo::LinkBotPathResponse &res)
{
  // TODO: Implement gripper1 path in parallel
  for (auto const &target_position : req.gripper1_path) {
    mode_ = "position";
    gripper1_target_position_.X(target_position.x);
    gripper1_target_position_.Y(target_position.y);

    auto control_result = UpdateControl();
    res.actual_path.emplace_back(control_result.configuration);

    // Wait until the setpoint is reached
    gazebo::common::Time::Sleep(common::Time(req.execute_path_dt));
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
