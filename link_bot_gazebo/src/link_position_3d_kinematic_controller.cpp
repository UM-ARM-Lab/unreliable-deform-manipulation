#include <link_bot_gazebo/link_position_3d_kinematic_controller.h>
#include <ros/console.h>
#include <gazebo/physics/physics.hh>

namespace gazebo
{

LinkPosition3dKinematicController::LinkPosition3dKinematicController(char const *plugin_name, physics::LinkPtr link) :
    BaseLinkPositionController(plugin_name, link)
{}

void LinkPosition3dKinematicController::Update(ignition::math::Vector3d const &setpoint)
{
  if (!link_)
  {
    ROS_ERROR_STREAM_NAMED(plugin_name_, "pointer to the link " << scoped_link_name_ << " is null");
    return;
  }

  // compute the current output position
  auto const current_position = Get();
  if (!current_position)
  {
    ROS_ERROR_STREAM_NAMED(plugin_name_, "failed to get link " << scoped_link_name_ << " state");
    return;
  }

  // update the output position, this is what makes "speed" mean something
  auto const dt = link_->GetWorld()->Physics()->GetMaxStepSize();
  // step_size is a decimal, from 0 to 1. take a step from current to setpoint
  auto const delta_distance = speed_mps_ * dt;
  auto const distance = current_position->Distance(setpoint);
  auto const step_size = std::fmin(delta_distance / distance, 1);
  auto const direction = (setpoint - *current_position);
  auto const output_position = [&]()
  {
    if (distance > 1e-4)
    {
      return *current_position + direction * step_size;
    } else
    {
      return *current_position;
    }
  }();

  if (!output_position.IsFinite())
  {
    ROS_ERROR_STREAM_NAMED(plugin_name_, ""
        << " current " << current_position->X() << " " << current_position->Y() << " " << current_position->Z()
        << " setpoint " << setpoint_.X() << " " << setpoint_.Y() << " " << setpoint_.Z()
        << " distance " << distance
        << " dt " << dt
        << " step " << step_size
        << " output " << output_position.X() << " " << output_position.Y() << " " << output_position.Z()
    );
    return;
  }

  // actually move the link
  link_->SetAngularVel(ignition::math::Vector3d::Zero);
  link_->SetLinearVel(ignition::math::Vector3d::Zero);
  auto const current_pose = link_->WorldPose();
  ignition::math::Pose3d pose{output_position, current_pose.Rot()};
  ROS_DEBUG_STREAM_THROTTLE_NAMED(1.0, plugin_name_, "Setting pose " << pose);
  link_->SetWorldPose(pose);
}

void LinkPosition3dKinematicController::OnEnable(bool enable)
{
  BaseLinkPositionController::OnEnable(enable);
  while (true)
  {
    // I'm seeing that calling "SetKinematic" actually just toggles kinematic,
    // so this ensure it gets set to the right value
    link_->SetKinematic(enable);
    auto const is_k = link_->GetKinematic();
    ROS_DEBUG_STREAM_NAMED(plugin_name_, "enable =" << enable << ". is kinematic = " << link_->GetKinematic());
    if (is_k == enable)
    {
      break;
    }
  }
}

} // namespace gazebo