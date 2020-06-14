#include "position_3d_plugin.h"

#define create_service_options(type, name, bind) \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

namespace gazebo {
GZ_REGISTER_MODEL_PLUGIN(Position3dPlugin)

Position3dPlugin::~Position3dPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_->shutdown();
  ros_queue_thread_.join();
}

void Position3dPlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf)
{
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

    if (sdf->HasElement("link")) {
      this->link_name_ = sdf->Get<std::string>("link");
    }
    else {
      ROS_FATAL_STREAM("The position 3d plugin requires a `linkName` parameter tag");
      return;
    }
  }

  model_ = parent;

  link_ = model_->GetLink(link_name_);
  if (!link_) {
    ROS_ERROR_NAMED("position_3d", "link not found");
    const std::vector<physics::LinkPtr> &links = model_->GetLinks();
    for (unsigned i = 0; i < links.size(); i++) {
      ROS_ERROR_STREAM_NAMED("position_3d", " -- Link " << i << ": " << links[i]->GetName());
    }
    return;
  }

  // setup ROS stuff
  int argc = 0;
  ros::init(argc, nullptr, model_->GetScopedName(), ros::init_options::NoSigintHandler);

  auto stop_bind = [this](std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res) { return OnStop(req, res); };
  auto stop_so = create_service_options(std_srvs::Empty, "stop", stop_bind);

  auto enable_bind = [this](peter_msgs::Position3DEnableRequest &req, peter_msgs::Position3DEnableResponse &res) {
    return OnEnable(req, res);
  };
  auto enable_so = create_service_options(peter_msgs::Position3DEnable, "enable", enable_bind);

  auto pos_action_bind = [this](peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res) {
    return OnAction(req, res);
  };
  auto pos_action_so = create_service_options(peter_msgs::Position3DAction, "set", pos_action_bind);

  auto get_pos_bind = [this](peter_msgs::GetPosition3DRequest &req, peter_msgs::GetPosition3DResponse &res) {
    return GetPos(req, res);
  };
  auto get_pos_so = create_service_options(peter_msgs::GetPosition3D, "get", get_pos_bind);

  ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
  enable_service_ = ros_node_->advertiseService(enable_so);
  action_service_ = ros_node_->advertiseService(pos_action_so);
  stop_service_ = ros_node_->advertiseService(stop_so);
  get_position_service_ = ros_node_->advertiseService(get_pos_so);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });

  constexpr auto max_vel_integral{0};
  constexpr auto max_integral{0};
  x_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);
  y_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);

  z_rot_pid_ = common::PID(kP_rot_, kI_rot_, kD_rot_, max_integral, -max_integral, max_torque_, -max_torque_);

  x_vel_pid_ = common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force_, -max_force_);
  y_vel_pid_ = common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force_, -max_force_);

  auto update = [this](common::UpdateInfo const &info) { OnUpdate(info); };
  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(update);

  target_pose_ = link_->WorldPose();
}

void Position3dPlugin::OnUpdate(common::UpdateInfo const &info)
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

bool Position3dPlugin::OnStop(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res)
{
  target_pose_ = link_->WorldPose();
  return true;
}

bool Position3dPlugin::OnEnable(peter_msgs::Position3DEnableRequest &req, peter_msgs::Position3DEnableResponse &res)
{
  enabled_ = req.enable;
  return true;
}

bool Position3dPlugin::OnAction(peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res)
{
  target_pose_.Pos().X(req.pose.position.x);
  target_pose_.Pos().Y(req.pose.position.y);
  target_pose_.Rot().X(req.pose.orientation.x);
  target_pose_.Rot().Y(req.pose.orientation.y);
  target_pose_.Rot().Z(req.pose.orientation.z);
  target_pose_.Rot().W(req.pose.orientation.w);
  return true;
}

bool Position3dPlugin::GetPos(peter_msgs::GetPosition3DRequest &req, peter_msgs::GetPosition3DResponse &res)
{
  auto const pos = link_->WorldPose().Pos();
  res.pos.x = pos.X();
  res.pos.y = pos.Y();
  res.pos.z = pos.Z();
  return true;
}

void Position3dPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_->ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}
}  // namespace gazebo
