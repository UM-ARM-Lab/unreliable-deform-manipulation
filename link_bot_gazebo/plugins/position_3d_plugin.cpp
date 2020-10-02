#include "position_3d_plugin.h"

#include <link_bot_gazebo/mymath.hpp>

#define create_service_options(type, name, bind)                                                                       \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

namespace gazebo
{
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
  if (!ros::isInitialized())
  {
    int argc = 0;
    ros::init(argc, nullptr, model_->GetScopedName(), ros::init_options::NoSigintHandler);
  }

  // Get sdf parameters
  {
    if (!sdf->HasElement("kP_pos"))
    {
      printf("using default kP_pos=%f\n", kP_pos_);
    } else
    {
      kP_pos_ = sdf->GetElement("kP_pos")->Get<double>();
    }

    if (!sdf->HasElement("kD_pos"))
    {
      printf("using default kD_pos=%f\n", kD_pos_);
    } else
    {
      kD_pos_ = sdf->GetElement("kD_pos")->Get<double>();
    }

    if (!sdf->HasElement("max_vel"))
    {
      printf("using default max_vel=%f\n", max_vel_);
    } else
    {
      max_vel_ = sdf->GetElement("max_vel")->Get<double>();
    }

    if (!sdf->HasElement("kP_vel"))
    {
      printf("using default kP_vel=%f\n", kP_vel_);
    } else
    {
      kP_vel_ = sdf->GetElement("kP_vel")->Get<double>();
    }

    if (!sdf->HasElement("kI_vel"))
    {
      printf("using default kI_vel=%f\n", kI_vel_);
    } else
    {
      kI_vel_ = sdf->GetElement("kI_vel")->Get<double>();
    }

    if (!sdf->HasElement("kD_vel"))
    {
      printf("using default kD_vel=%f\n", kD_vel_);
    } else
    {
      kD_vel_ = sdf->GetElement("kD_vel")->Get<double>();
    }

    if (!sdf->HasElement("max_force"))
    {
      printf("using default max_force=%f\n", max_force_);
    } else
    {
      max_force_ = sdf->GetElement("max_force")->Get<double>();
    }

    if (sdf->HasElement("gravity_compensation"))
    {
      gravity_compensation_ = sdf->GetElement("gravity_compensation")->Get<bool>();
    }

    if (sdf->HasElement("link"))
    {
      this->link_name_ = sdf->Get<std::string>("link");
    } else
    {
      ROS_FATAL_STREAM("The position 3d plugin requires a `link` parameter tag");
      return;
    }
  }

  link_ = model_->GetLink(link_name_);
  if (!link_)
  {
    gzerr << "position_3d link not found\n";
    const std::vector<physics::LinkPtr> &links = model_->GetLinks();
    for (unsigned i = 0; i < links.size(); i++)
    {
      gzerr << "position_3d -- Link " << i << ": " << links[i]->GetName() << std::endl;
    }
    return;
  }

  // compute total mass
  const std::vector<physics::LinkPtr> &links = model_->GetLinks();
  for (const auto &link : links)
  {
    total_mass_ += link->GetInertial()->Mass();
  }

  {
    private_ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());

    {
      auto stop_bind = [this](std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res)
      { return OnStop(req, res); };
      auto stop_so = create_service_options(std_srvs::Empty, "stop", stop_bind);
      stop_service_ = private_ros_node_->advertiseService(stop_so);
    }

    {
      auto enable_bind = [this](peter_msgs::Position3DEnableRequest &req, peter_msgs::Position3DEnableResponse &res)
      {
        return OnEnable(req, res);
      };
      auto enable_so = create_service_options(peter_msgs::Position3DEnable, "enable", enable_bind);
      enable_service_ = private_ros_node_->advertiseService(enable_so);
    }

    {
      auto pos_move_bind = [this](peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res)
      {
        return OnMove(req, res);
      };
      auto pos_move_so = create_service_options(peter_msgs::Position3DAction, "move", pos_move_bind);
      move_service_ = private_ros_node_->advertiseService(pos_move_so);
    }

    {
      auto pos_set_bind = [this](peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res)
      {
        return OnSet(req, res);
      };
      auto pos_set_so = create_service_options(peter_msgs::Position3DAction, "set", pos_set_bind);
      set_service_ = private_ros_node_->advertiseService(pos_set_so);
    }

    {
      auto get_pos_bind = [this](peter_msgs::GetPosition3DRequest &req, peter_msgs::GetPosition3DResponse &res)
      {
        return GetPos(req, res);
      };
      auto get_pos_so = create_service_options(peter_msgs::GetPosition3D, "get", get_pos_bind);
      get_position_service_ = private_ros_node_->advertiseService(get_pos_so);
    }

    ros_queue_thread_ = std::thread([this]
                                    { QueueThread(); });
    private_ros_queue_thread_ = std::thread([this]
                                            { PrivateQueueThread(); });
  }

  pos_pid_ = common::PID(kP_pos_, 0, kD_pos_, 0, 0, max_vel_, -max_vel_);
  vel_pid_ = common::PID(kP_vel_, 0, kD_vel_, 0, 0, max_force_, -max_force_);

  auto update = [this](common::UpdateInfo const &info)
  { OnUpdate(info); };
  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(update);

  target_position_ = link_->WorldPose().Pos();
}

void Position3dPlugin::OnUpdate(common::UpdateInfo const &info)
{
  (void) info;
  constexpr auto dt{0.001};

  auto const pos = link_->WorldPose().Pos();
  auto const vel_ = link_->WorldLinearVel();

  pos_error_ = pos - target_position_;
  auto const target_vel = pos_error_.Normalized() * pos_pid_.Update(pos_error_.Length(), dt);

  auto const vel_error = vel_ - target_vel;
  auto force = vel_error.Normalized() * vel_pid_.Update(vel_error.Length(), dt);

  if (gravity_compensation_)
  {
    auto const max_i = total_mass_ * model_->GetWorld()->Gravity().Length();
    auto const z_comp = kI_vel_ * z_integral_;

    // FIXME: there's a bug waiting here, one of these branches is wrong...
    if (vel_error.Z() < 0 and z_comp < max_i)
    {
      z_integral_ += -vel_error.Z();
    } else if (vel_error.Z() > 0 and z_comp > 0)
    {
      z_integral_ += -vel_error.Z();
    }
    force.Z(force.Z() + z_comp);
  }

  if (enabled_)
  {
    link_->AddForce(force);
  }
}

bool Position3dPlugin::OnStop(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res)
{
  (void) req;
  (void) res;

  enabled_ = true;
  target_position_ = link_->WorldPose().Pos();
  return true;
}

bool Position3dPlugin::OnEnable(peter_msgs::Position3DEnableRequest &req, peter_msgs::Position3DEnableResponse &res)
{
  (void) res;
  enabled_ = req.enable;
  if (req.enable)
  {
    target_position_ = link_->WorldPose().Pos();
  }
  return true;
}

bool Position3dPlugin::OnSet(peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res)
{
  (void) res;
  // Only set the position, don't step physics
  enabled_ = true;
  target_position_.X(req.position.x);
  target_position_.Y(req.position.y);
  target_position_.Z(req.position.z);
  return true;
}

bool Position3dPlugin::OnMove(peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res)
{
  (void) res;
  enabled_ = true;
  target_position_.X(req.position.x);
  target_position_.Y(req.position.y);
  target_position_.Z(req.position.z);

  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const steps = static_cast<unsigned int>(req.timeout / seconds_per_step);

  // Wait until the setpoint or timeout is reached
  for (auto i{0ul}; i < steps; ++i)
  {
    model_->GetWorld()->Step(1);
    auto const reached = pos_error_.Length() < 0.001;
    if (reached)
    {
      break;
    }
  }
  return true;
}

bool Position3dPlugin::GetPos(peter_msgs::GetPosition3DRequest &req, peter_msgs::GetPosition3DResponse &res)
{
  (void) req;
  auto const pos = link_->WorldPose().Pos();
  res.pos.x = pos.X();
  res.pos.y = pos.Y();
  res.pos.z = pos.Z();
  return true;
}

void Position3dPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void Position3dPlugin::PrivateQueueThread()
{
  double constexpr timeout = 0.01;
  while (private_ros_node_->ok())
  {
    private_queue_.callAvailable(ros::WallDuration(timeout));
  }
}

}  // namespace gazebo
