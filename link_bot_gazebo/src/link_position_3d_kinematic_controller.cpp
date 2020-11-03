#include <link_bot_gazebo/link_position_3d_kinematic_controller.h>
#include <ros/console.h>
#include <link_bot_gazebo/gazebo_plugin_utils.h>

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
  link_->SetAngularVel(ignition::math::Vector3d::Zero);
  link_->SetLinearVel(ignition::math::Vector3d::Zero);
  auto const current_pose = link_->WorldPose();
  ignition::math::Pose3d pose{setpoint, current_pose.Rot()};
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