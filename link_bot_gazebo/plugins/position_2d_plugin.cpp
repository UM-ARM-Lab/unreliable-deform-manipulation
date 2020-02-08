#include "position_2d_plugin.h"
#include <ros/ros.h>

namespace gazebo {
GZ_REGISTER_MODEL_PLUGIN(Position2dPlugin);

Position2dPlugin::~Position2dPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_->shutdown();
  ros_queue_thread_.join();
}

void Position2dPlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf)
{
  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "position_2d_plugin", ros::init_options::NoSigintHandler);
    return;
  }

  // Get sdf parameters
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

    if (!sdf->HasElement("max_vel")) {
      printf("using default max_vel=%f\n", max_vel_);
    }
    else {
      max_vel_ = sdf->GetElement("max_vel")->Get<double>();
    }

    if (!sdf->HasElement("kP_rot")) {
      printf("using default kP_rot=%f\n", kP_rot_);
    }
    else {
      kP_rot_ = sdf->GetElement("kP_rot")->Get<double>();
    }

    if (!sdf->HasElement("kI_rot")) {
      printf("using default kI_rot=%f\n", kI_rot_);
    }
    else {
      kI_rot_ = sdf->GetElement("kI_rot")->Get<double>();
    }

    if (!sdf->HasElement("kD_rot")) {
      printf("using default kD_rot=%f\n", kD_rot_);
    }
    else {
      kD_rot_ = sdf->GetElement("kD_rot")->Get<double>();
    }

    if (!sdf->HasElement("max_torque")) {
      printf("using default max_torque=%f\n", max_torque_);
    }
    else {
      max_torque_ = sdf->GetElement("max_torque")->Get<double>();
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

    if (!sdf->HasElement("max_force")) {
      printf("using default max_force=%f\n", max_force_);
    }
    else {
      max_force_ = sdf->GetElement("max_force")->Get<double>();
    }

    if (sdf->HasElement("linkName")) {
      this->link_name_ = sdf->Get<std::string>("linkName");
    }
    else {
      ROS_FATAL_STREAM("The position 2d plugin requires a `linkName` parameter tag");
      return;
    }
  }

  model_ = parent;

  link_ = model_->GetLink(link_name_);
  if (!link_) {
    ROS_ERROR_NAMED("hand_of_god", "link not found");
    const std::vector<physics::LinkPtr> &links = model_->GetLinks();
    for (unsigned i = 0; i < links.size(); i++) {
      ROS_ERROR_STREAM_NAMED("hand_of_god", " -- Link " << i << ": " << links[i]->GetName());
    }
    return;
  }

  auto stop_bind = boost::bind(&Position2dPlugin::OnStop, this, _1);
  auto stop_so =
      ros::SubscribeOptions::create<std_msgs::Empty>("/position_2d_stop", 1, stop_bind, ros::VoidPtr(), &queue_);
  auto enable_bind = boost::bind(&Position2dPlugin::OnEnable, this, _1);
  auto enable_so = ros::SubscribeOptions::create<link_bot_gazebo::Position2dEnable>(
      "/position_2d_enable", 1, enable_bind, ros::VoidPtr(), &queue_);
  auto pos_action_bind = boost::bind(&Position2dPlugin::OnAction, this, _1);
  auto pos_action_so = ros::SubscribeOptions::create<link_bot_gazebo::Position2dAction>(
      "/position_2d_action", 1, pos_action_bind, ros::VoidPtr(), &queue_);

  ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
  enable_sub_ = ros_node_->subscribe(enable_so);
  action_sub_ = ros_node_->subscribe(pos_action_so);
  stop_sub_ = ros_node_->subscribe(stop_so);

  ros_queue_thread_ = std::thread(std::bind(&Position2dPlugin::QueueThread, this));

  constexpr auto max_vel_integral{0};
  constexpr auto max_integral{0};
  x_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);
  y_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);

  z_rot_pid_ = common::PID(kP_rot_, kI_rot_, kD_rot_, max_integral, -max_integral, max_torque_, -max_torque_);

  x_vel_pid_ = common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force_, -max_force_);
  y_vel_pid_ = common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force_, -max_force_);

  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(boost::bind(&Position2dPlugin::OnUpdate, this));

  target_pose_ = link_->WorldPose();
}

void Position2dPlugin::OnUpdate()
{
  constexpr auto dt{0.001};

  ignition::math::Vector3d force;
  auto const pos = link_->WorldPose().Pos();
  auto const pose_error = pos - target_pose_.Pos();

  target_velocity_.X(x_pos_pid_.Update(pose_error.X(), dt));
  target_velocity_.Y(y_pos_pid_.Update(pose_error.Y(), dt));

  auto const vel = link_->WorldLinearVel();
  auto const vel_error = vel - target_velocity_;
  force.X(x_vel_pid_.Update(vel_error.X(), dt));
  force.Y(y_vel_pid_.Update(vel_error.Y(), dt));

  // FIXME: this assumes the objects are boxes
  auto const i{0u};
  auto const collision = link_->GetCollision(i);
  auto const box = boost::dynamic_pointer_cast<physics::BoxShape>(collision->GetShape());
  auto const z = box->Size().Z();

  ignition::math::Vector3d torque;
  torque.X(0);
  torque.Y(0);
  auto const yaw_error = link_->WorldPose().Rot().Yaw();
  auto const yaw_torque = z_rot_pid_.Update(yaw_error, dt);
  torque.Z(yaw_torque);

  if (enabled_) {
    ignition::math::Vector3d const push_position(0, 0, -z / 2.0 + 0.01);
    link_->AddForceAtRelativePosition(force, push_position);
    link_->AddRelativeTorque(torque);
  }
}

void Position2dPlugin::OnStop(std_msgs::EmptyConstPtr const msg)
{
  target_pose_ = link_->WorldPose();
  std::cout << "stopping!\n";
}

void Position2dPlugin::OnEnable(link_bot_gazebo::Position2dEnableConstPtr const msg)
{
  for (auto const &model_name : msg->model_names) {
    if (model_name == model_->GetScopedName()) {
      enabled_ = msg->enable;
    }
  }
}

void Position2dPlugin::OnAction(link_bot_gazebo::Position2dActionConstPtr const msg)
{
  for (auto const &action : msg->actions) {
    if (action.model_name == model_->GetScopedName()) {
      target_pose_.Pos().X(action.pose.position.x);
      target_pose_.Pos().Y(action.pose.position.y);
      target_pose_.Rot().X(action.pose.orientation.x);
      target_pose_.Rot().Y(action.pose.orientation.y);
      target_pose_.Rot().Z(action.pose.orientation.z);
      target_pose_.Rot().W(action.pose.orientation.w);
    }
  }
}

void Position2dPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_->ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}
}  // namespace gazebo
