#include <link_bot_gazebo/link_position_3d_pid_controller.h>
#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include <ros/console.h>

namespace gazebo
{

LinkPosition3dPIDController::LinkPosition3dPIDController(char const *plugin_name,
                                                         physics::LinkPtr link,
                                                         double const kp_pos,
                                                         double const kp_vel,
                                                         double const max_force,
                                                         double const max_vel,
                                                         bool const grav_comp) :
    BaseLinkPositionController(plugin_name, link),
    kP_pos_(kp_pos),
    kP_vel_(kp_vel),
    max_force_(max_force),
    max_vel_(max_vel),
    gravity_compensation_(grav_comp)
{
  if (!link_)
  {
    ROS_ERROR_STREAM_NAMED(plugin_name_, "pointer to the link " << scoped_link_name_ << " is null");
    return;
  }

  // compute total mass
  auto const model = link_->GetModel();
  const std::vector<physics::LinkPtr> &links = model->GetLinks();
  for (const auto &link : links)
  {
    total_mass_ += link->GetInertial()->Mass();
  }

  pos_pid_ = common::PID(kP_pos_, 0, kD_pos_, 0, 0, max_vel_, -max_vel_);
  vel_pid_ = common::PID(kP_vel_, 0, kD_vel_, 0, 0, max_force_, -max_force_);
}

void LinkPosition3dPIDController::Update(ignition::math::Vector3d const &setpoint)
{
  if (!link_)
  {
    ROS_ERROR_STREAM_NAMED(plugin_name_, "pointer to the link " << scoped_link_name_ << " is null");
    return;
  }

  auto const dt = link_->GetWorld()->Physics()->GetMaxStepSize();

  auto const pos = link_->WorldPose().Pos();
  auto const vel_ = link_->WorldLinearVel();

  pos_error_ = pos - setpoint;
  auto const target_vel = pos_error_.Normalized() * pos_pid_.Update(pos_error_.Length(), dt);

  auto const vel_error = vel_ - target_vel;
  auto force = vel_error.Normalized() * vel_pid_.Update(vel_error.Length(), dt);

  if (gravity_compensation_)
  {
    auto const max_i = total_mass_ * link_->GetWorld()->Gravity().Length();
    auto const z_comp = kI_vel_ * z_integral_;

    // FIXME: there's a bug waiting here, one of these branches is wrong but I don't know which one...
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

}  // namespace gazebo
