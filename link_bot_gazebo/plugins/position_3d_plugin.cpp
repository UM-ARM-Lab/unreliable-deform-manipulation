#include "position_3d_plugin.h"

#include <link_bot_gazebo/mymath.hpp>

#define create_service_options(type, name, bind) \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

namespace gazebo {
GZ_REGISTER_MODEL_PLUGIN(Position3dPlugin)

Position3dPlugin::~Position3dPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  private_ros_node_->shutdown();
  ros_queue_thread_.join();
  private_ros_queue_thread_.join();
}

void Position3dPlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf)
{
  model_ = parent;

  // setup ROS stuff
  if (!ros::isInitialized()) {
    int argc = 0;
    ros::init(argc, nullptr, model_->GetScopedName(), ros::init_options::NoSigintHandler);
  }

  // Get sdf parameters
  {
    if (sdf->HasElement("object_name")) {
      name_ = sdf->GetElement("object_name")->Get<std::string>();
    }
    else {
      name_ = model_->GetScopedName();
    }

    if (!sdf->HasElement("kP_pos")) {
      printf("using default kP_pos=%f\n", kP_pos_);
    }
    else {
      kP_pos_ = sdf->GetElement("kP_pos")->Get<double>();
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

    if (!sdf->HasElement("kP_rot")) {
      printf("using default kP_rot=%f\n", kP_rot_);
    }
    else {
      kP_rot_ = sdf->GetElement("kP_rot")->Get<double>();
    }

    if (!sdf->HasElement("kD_rot")) {
      printf("using default kD_rot=%f\n", kD_rot_);
    }
    else {
      kD_rot_ = sdf->GetElement("kD_rot")->Get<double>();
    }

    if (!sdf->HasElement("kP_rot_vel")) {
      printf("using default kP_rot_vel=%f\n", kP_rot_vel_);
    }
    else {
      kP_rot_vel_ = sdf->GetElement("kP_rot_vel")->Get<double>();
    }

    if (!sdf->HasElement("kD_rot_vel")) {
      printf("using default kD_rot_vel=%f\n", kD_rot_vel_);
    }
    else {
      kD_rot_vel_ = sdf->GetElement("kD_rot_vel")->Get<double>();
    }

    if (!sdf->HasElement("max_torque")) {
      printf("using default max_torque=%f\n", max_torque_);
    }
    else {
      max_torque_ = sdf->GetElement("max_torque")->Get<double>();
    }

    if (sdf->HasElement("gravity_compensation")) {
      gravity_compensation_ = sdf->GetElement("gravity_compensation")->Get<bool>();
    }

    if (sdf->HasElement("link")) {
      this->link_name_ = sdf->Get<std::string>("link");
    }
    else {
      ROS_FATAL_STREAM("The position 3d plugin requires a `link` parameter tag");
      return;
    }
  }

  link_ = model_->GetLink(link_name_);
  if (!link_) {
    gzerr << "position_3d link not found\n";
    const std::vector<physics::LinkPtr> &links = model_->GetLinks();
    for (unsigned i = 0; i < links.size(); i++) {
      gzerr << "position_3d -- Link " << i << ": " << links[i]->GetName() << std::endl;
    }
    return;
  }

  // compute total mass
  const std::vector<physics::LinkPtr> &links = model_->GetLinks();
  for (const auto &link : links) {
    total_mass_ += link->GetInertial()->Mass();
  }

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

  auto get_object_bind = [this](auto &&req, auto &&res) { return GetObjectCallback(req, res); };
  auto get_object_so = create_service_options(peter_msgs::GetObject, name_, get_object_bind);

  private_ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
  enable_service_ = private_ros_node_->advertiseService(enable_so);
  action_service_ = private_ros_node_->advertiseService(pos_action_so);
  stop_service_ = private_ros_node_->advertiseService(stop_so);
  get_position_service_ = private_ros_node_->advertiseService(get_pos_so);

  register_object_pub_ = ros_node_.advertise<std_msgs::String>("register_object", 10, true);
  get_object_service_ = ros_node_.advertiseService(get_object_so);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
  private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

  gzwarn << "[" << model_->GetScopedName() << "] Waiting for object server\n";
  while (register_object_pub_.getNumSubscribers() < 1) {
  }

  {
    std_msgs::String register_object;
    register_object.data = name_;
    register_object_pub_.publish(register_object);
  }

  pos_pid_ = common::PID(kP_pos_, 0, kD_pos_, 0, 0, max_vel_, -max_vel_);
  vel_pid_ = common::PID(kP_vel_, 0, kD_vel_, 0, 0, max_force_, -max_force_);
  rot_pid_ = common::PID(kP_rot_, 0, kD_rot_, 0, 0, max_rot_vel_, -max_rot_vel_);
  rot_vel_pid_ = common::PID(kP_rot_vel_, 0, kD_rot_vel_, 0, 0, max_torque_, -max_torque_);

  auto update = [this](common::UpdateInfo const &info) { OnUpdate(info); };
  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(update);

  target_position_ = link_->WorldPose().Pos();
}

void Position3dPlugin::OnUpdate(common::UpdateInfo const &info)
{
  constexpr auto dt{0.001};

  auto const pos = link_->WorldPose().Pos();
  auto const rot = link_->RelativePose().Rot();
  auto const vel_ = link_->WorldLinearVel();
  auto const rot_vel_ = link_->RelativeAngularVel();

  pos_error_ = pos - target_position_;
  auto const target_vel = pos_error_.Normalized() * pos_pid_.Update(pos_error_.Length(), dt);

  auto const vel_error = vel_ - target_vel;
  auto force = vel_error.Normalized() * vel_pid_.Update(vel_error.Length(), dt);

  rot_error_ = angle_error(rot.Z(), 0);  // assume target rotation is 0
  auto const target_rot_vel = rot_pid_.Update(rot_error_, dt);

  auto const rot_vel_error = rot_vel_.X() - target_rot_vel;
  auto const torque = rot_vel_pid_.Update(rot_vel_error, dt);

  if (gravity_compensation_) {
    auto const max_i = total_mass_ * model_->GetWorld()->Gravity().Length();
    auto const z_comp = kI_vel_ * z_integral_;

    if (vel_error.Z() < 0 and z_comp < max_i) {
      z_integral_ += -vel_error.Z();
    }
    else if (vel_error.Z() > 0 and z_comp > 0) {
      z_integral_ += -vel_error.Z();
    }
    force.Z(force.Z() + z_comp);
  }

  if (enabled_) {
    link_->AddForce(force);
//    link_->AddRelativeTorque({torque, 0, 0});
  }
}

bool Position3dPlugin::OnStop(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res)
{
  target_position_ = link_->WorldPose().Pos();
  return true;
}

bool Position3dPlugin::OnEnable(peter_msgs::Position3DEnableRequest &req, peter_msgs::Position3DEnableResponse &res)
{
  enabled_ = req.enable;
  if (req.enable) {
    target_position_ = link_->WorldPose().Pos();
  }
  return true;
}

bool Position3dPlugin::OnAction(peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res)
{
  enabled_ = true;
  target_position_.X(req.position.x);
  target_position_.Y(req.position.y);
  target_position_.Z(req.position.z);

  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const steps = static_cast<unsigned int>(req.timeout / seconds_per_step);
  // Wait until the setpoint is reached
  model_->GetWorld()->Step(steps);

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

bool Position3dPlugin::GetObjectCallback(peter_msgs::GetObjectRequest &req, peter_msgs::GetObjectResponse &res)
{
  std::vector<float> state_vector;
  peter_msgs::NamedPoint link_point;
  geometry_msgs::Point pt;
  float const x = link_->WorldPose().Pos().X();
  float const y = link_->WorldPose().Pos().Y();
  float const z = link_->WorldPose().Pos().Z();
  state_vector.push_back(x);
  state_vector.push_back(y);
  state_vector.push_back(z);
  link_point.point.x = x;
  link_point.point.y = y;
  link_point.point.z = z;
  link_point.name = link_->GetName();

  res.object.name = name_;
  res.object.state_vector = state_vector;
  res.object.points.emplace_back(link_point);

  return true;
}

void Position3dPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void Position3dPlugin::PrivateQueueThread()
{
  double constexpr timeout = 0.01;
  while (private_ros_node_->ok()) {
    private_queue_.callAvailable(ros::WallDuration(timeout));
  }
}

}  // namespace gazebo
