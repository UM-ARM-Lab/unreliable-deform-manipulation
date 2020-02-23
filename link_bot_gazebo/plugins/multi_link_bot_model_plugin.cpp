#include "multi_link_bot_model_plugin.h"

#include <link_bot_gazebo/GetObjects.h>
#include <std_srvs/EmptyRequest.h>

#include <cstdio>
#include <gazebo/common/Time.hh>
#include <gazebo/common/Timer.hh>
#include <ignition/math/Vector3.hh>
#include <iterator>
#include <memory>
#include <sstream>

namespace gazebo {

constexpr auto close_enough{0.001};
constexpr auto stopped_threshold{0.01};

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
  auto execute_trajectory_bind = boost::bind(&MultiLinkBotModelPlugin::ExecuteTrajectoryCallback, this, _1, _2);
  auto execute_trajectory_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::LinkBotTrajectory>(
      "/link_bot_execute_trajectory", execute_trajectory_bind, ros::VoidPtr(), &queue_);
  auto execute_abs_action_bind = boost::bind(&MultiLinkBotModelPlugin::ExecuteAbsoluteAction, this, _1, _2);
  auto execute_abs_action_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::ExecuteAction>(
      "/execute_absolute_action", execute_abs_action_bind, ros::VoidPtr(), &queue_);
  auto action_bind = boost::bind(&MultiLinkBotModelPlugin::ExecuteAction, this, _1, _2);
  auto action_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::ExecuteAction>("/execute_action", action_bind,
                                                                                        ros::VoidPtr(), &queue_);
  auto action_mode_bind = boost::bind(&MultiLinkBotModelPlugin::OnActionMode, this, _1);
  auto action_mode_so = ros::SubscribeOptions::create<std_msgs::String>("/link_bot_action_mode", 1, action_mode_bind,
                                                                        ros::VoidPtr(), &queue_);
  auto config_bind = boost::bind(&MultiLinkBotModelPlugin::OnConfiguration, this, _1);
  auto config_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotJointConfiguration>(
      "/link_bot_configuration", 1, config_bind, ros::VoidPtr(), &queue_);
  auto state_bind = boost::bind(&MultiLinkBotModelPlugin::StateServiceCallback, this, _1, _2);
  auto service_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::LinkBotState>("/link_bot_state", state_bind,
                                                                                        ros::VoidPtr(), &queue_);
  auto reset_bind = boost::bind(&MultiLinkBotModelPlugin::LinkBotReset, this, _1, _2);
  auto reset_so =
      ros::AdvertiseServiceOptions::create<link_bot_gazebo::LinkBotReset>("/link_bot_reset", reset_bind, ros::VoidPtr(), &queue_);

  joy_sub_ = ros_node_->subscribe(joy_so);
  execute_action_service_ = ros_node_->advertiseService(action_so);
  execute_absolute_action_service_ = ros_node_->advertiseService(execute_abs_action_so);
  reset_service_ = ros_node_->advertiseService(reset_so);
  action_mode_sub_ = ros_node_->subscribe(action_mode_so);
  config_sub_ = ros_node_->subscribe(config_so);
  state_service_ = ros_node_->advertiseService(service_so);
  execute_traj_service_ = ros_node_->advertiseService(execute_trajectory_so);
  objects_service_ = ros_node_->serviceClient<link_bot_gazebo::GetObjects>("/objects");

  ros_queue_thread_ = std::thread(std::bind(&MultiLinkBotModelPlugin::QueueThread, this));

  model_ = parent;

  {
    if (!sdf->HasElement("rope_length")) {
      printf("using default rope length=%f\n", length_);
    }
    else {
      length_ = sdf->GetElement("rope_length")->Get<double>();
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
  }

  // plus 1 because we want both end points inclusive
  ros_node_->setParam("/link_bot/n_state", static_cast<int>((num_links_ + 1) * 2));
  ros_node_->setParam("/link_bot/rope_length", length_);
  ros_node_->setParam("/link_bot/max_speed", max_vel_);

  auto const &gripper1_link_name = sdf->GetElement("gripper1_link")->Get<std::string>();
  gripper1_link_ = model_->GetLink(gripper1_link_name);

  updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&MultiLinkBotModelPlugin::OnUpdate, this));
  constexpr auto max_integral{0};
  gripper1_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);

  constexpr auto max_vel_integral{1};
  gripper1_vel_pid_ =
      common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force_, -max_force_);

  gzlog << "MultiLinkBot Model Plugin finished initializing!\n";
}
#pragma clang diagnostic pop

link_bot_gazebo::NamedPoints MultiLinkBotModelPlugin::GetConfiguration()
{
  link_bot_gazebo::NamedPoints configuration;
  configuration.name = "link_bot";
  for (auto link_idx{1U}; link_idx <= num_links_; ++link_idx) {
    std::stringstream ss;
    ss << "link_" << link_idx;
    auto link_name = ss.str();
    auto const link = model_->GetLink(link_name);
    link_bot_gazebo::NamedPoint named_point;
    named_point.point.x = link->WorldPose().Pos().X();
    named_point.point.y = link->WorldPose().Pos().Y();
    named_point.name = link_name;
    configuration.points.emplace_back(named_point);
  }

  auto const head = model_->GetLink("head");
  link_bot_gazebo::NamedPoint named_point;
  named_point.point.x = head->WorldPose().Pos().X();
  named_point.point.y = head->WorldPose().Pos().Y();
  named_point.name = "head";
  configuration.points.emplace_back(named_point);

  return configuration;
}

auto MultiLinkBotModelPlugin::GetGripper1Pos() -> ignition::math::Vector3d const
{
  auto p = gripper1_link_->WorldPose().Pos();
  // zero the Z component
  p.Z(0);
  return p;
}

auto MultiLinkBotModelPlugin::GetGripper1Vel() -> ignition::math::Vector3d const
{
  auto v = gripper1_link_->WorldLinearVel();
  v.Z(0);
  return v;
}

ControlResult MultiLinkBotModelPlugin::UpdateControl()
{
  std::lock_guard<std::mutex> guard(control_mutex_);
  auto const dt = model_->GetWorld()->Physics()->GetMaxStepSize();
  ControlResult control_result{};

  control_result.link_bot_config = GetConfiguration();

  auto const gripper1_pos = GetGripper1Pos();
  auto const gripper1_vel_ = GetGripper1Vel();

  // Gripper 1
  {
    if (mode_ == "position") {
      gripper1_pos_error_ = gripper1_pos - gripper1_target_position_;
      auto const target_vel = gripper1_pos_pid_.Update(gripper1_pos_error_.Length(), dt);
      auto const gripper1_target_velocity = gripper1_pos_error_.Normalized() * target_vel;
      control_result.gripper1_vel = gripper1_target_velocity;

      auto const gripper1_vel_error = gripper1_vel_ - gripper1_target_velocity;
      auto const force_mag = gripper1_vel_pid_.Update(gripper1_vel_error.Length(), dt);
      control_result.gripper1_force = gripper1_vel_error.Normalized() * force_mag;
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
}

void MultiLinkBotModelPlugin::OnJoy(sensor_msgs::JoyConstPtr const msg)
{
  constexpr auto scale{2000.0 / 32768.0};
  gripper1_target_position_.X(gripper1_target_position_.X() - msg->axes[0] * scale);
  gripper1_target_position_.Y(gripper1_target_position_.Y() + msg->axes[1] * scale);
}

bool MultiLinkBotModelPlugin::ExecuteAbsoluteAction(link_bot_gazebo::ExecuteActionRequest &req,
                                                    link_bot_gazebo::ExecuteActionResponse &res)
{
  mode_ = "position";

  ignition::math::Vector3d position{req.action.gripper1_delta_pos.x, req.action.gripper1_delta_pos.y,
                                    req.action.gripper1_delta_pos.z};
  gripper1_target_position_ = position;

  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const steps = static_cast<unsigned int>(req.action.max_time_per_step / seconds_per_step);
  // Wait until the setpoint is reached
  model_->GetWorld()->Step(steps);

  // TODO: fill out state here
  res.needs_reset = false;

  // stop by setting the current position as the target
  gripper1_target_position_ = GetGripper1Pos();

  return true;
}

bool MultiLinkBotModelPlugin::ExecuteAction(link_bot_gazebo::ExecuteActionRequest &req,
                                            link_bot_gazebo::ExecuteActionResponse &res)
{
  mode_ = "position";

  ignition::math::Vector3d delta_position{req.action.gripper1_delta_pos.x, req.action.gripper1_delta_pos.y,
                                          req.action.gripper1_delta_pos.z};
  gripper1_target_position_ += delta_position;

  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const steps = static_cast<unsigned int>(req.action.max_time_per_step / seconds_per_step);
  // Wait until the setpoint is reached
  model_->GetWorld()->Step(steps);

  // TODO: fill out state here
  res.needs_reset = false;

  // stop by setting the current position as the target
  gripper1_target_position_ = GetGripper1Pos();

  return true;
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
  // get all links named "link_%d" where d is in [1, num_links)
  for (auto const &link : model_->GetLinks()) {
    auto const name = link->GetName();
    int link_idx;
    auto const n_matches = sscanf(name.c_str(), "link_%d", &link_idx);
    if (n_matches == 1 and link_idx >= 1 and link_idx <= num_links_) {
      geometry_msgs::Point pt;
      pt.x = link->WorldPose().Pos().X();
      pt.y = link->WorldPose().Pos().Y();
      res.points.emplace_back(pt);
      res.link_names.emplace_back(name);
    }
  }

  auto const link = model_->GetLink("head");
  geometry_msgs::Point pt;
  pt.x = link->WorldPose().Pos().X();
  pt.y = link->WorldPose().Pos().Y();
  res.points.emplace_back(pt);
  res.link_names.emplace_back("head");

  res.gripper1_force.x = gripper1_vel_pid_.GetCmd();
  res.gripper1_force.z = 0;

  auto const gripper1_velocity = gripper1_link_->WorldLinearVel();
  res.gripper1_velocity.x = gripper1_velocity.X();
  res.gripper1_velocity.y = gripper1_velocity.Y();
  res.gripper1_velocity.z = 0;

  res.header.stamp = ros::Time::now();

  return true;
}

bool MultiLinkBotModelPlugin::ExecuteTrajectoryCallback(link_bot_gazebo::LinkBotTrajectoryRequest &req,
                                                        link_bot_gazebo::LinkBotTrajectoryResponse &res)
{
  // TODO: Implement gripper1 path in parallel
  for (auto const &action : req.gripper1_traj) {
    mode_ = "position";
    ignition::math::Vector3d delta_position{action.gripper1_delta_pos.x, action.gripper1_delta_pos.y,
                                            action.gripper1_delta_pos.z};
    gripper1_target_position_ += delta_position;

    auto control_result = UpdateControl();
    auto link_bot_object = control_result.link_bot_config;
    link_bot_object.name = "link_bot";

    // get tether (or other) object configurations
    link_bot_gazebo::GetObjectsRequest get_objects_req;
    link_bot_gazebo::GetObjectsResponse get_objects_res;
    objects_service_.call(get_objects_req, get_objects_res);

    link_bot_gazebo::Objects objects;
    objects.objects.push_back(link_bot_object);
    std::copy(get_objects_res.objects.objects.begin(), get_objects_res.objects.objects.end(),
              std::back_inserter(objects.objects));

    res.actual_path.emplace_back(objects);

    auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
    auto const steps = static_cast<unsigned int>(action.max_time_per_step / seconds_per_step);
    for (auto i{0u}; i < steps; ++i) {
      model_->GetWorld()->Step(1);
      // check if setpoint is reached
      if (gripper1_pos_error_.Length() < close_enough and gripper1_vel_.Length() < stopped_threshold) {
        break;
      }
    }

    // stop by setting the current position as the target
    gripper1_target_position_ = GetGripper1Pos();
  }

  auto link_bot_object = GetConfiguration();

  // get tether (or other) object configurations
  link_bot_gazebo::GetObjectsRequest get_objects_req;
  link_bot_gazebo::GetObjectsResponse get_objects_res;
  objects_service_.call(get_objects_req, get_objects_res);

  link_bot_gazebo::Objects objects;
  objects.objects.push_back(link_bot_object);
  std::copy(get_objects_res.objects.objects.begin(), get_objects_res.objects.objects.end(),
            std::back_inserter(objects.objects));
  res.actual_path.emplace_back(objects);

  return true;
}

bool MultiLinkBotModelPlugin::LinkBotReset(link_bot_gazebo::LinkBotResetRequest &req, link_bot_gazebo::LinkBotResetResponse &res)
{
  gripper1_target_position_.X(req.point.x);
  gripper1_target_position_.Y(req.point.y);

  while (true) {
    auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
    auto const steps = static_cast<unsigned int>(1.0 / seconds_per_step);
    // Wait until the setpoint is reached
    model_->GetWorld()->Step(steps);
    if (gripper1_pos_error_.Length() < close_enough) {
      break;
    }
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
