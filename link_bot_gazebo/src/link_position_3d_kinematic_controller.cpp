#include <link_bot_gazebo/link_position_3d_kinematic_controller.h>

namespace gazebo
{

LinkPosition3dKinematicController::LinkPosition3dKinematicController(char const *plugin_name,
                                                                     std::string scoped_link_name,
                                                                     physics::WorldPtr world) :
    BaseLinkPositionController(plugin_name, scoped_link_name, world)
{}

void LinkPosition3dKinematicController::Update()
{
  link_->SetAngularVel(ignition::math::Vector3d::Zero);
  link_->SetLinearVel(ignition::math::Vector3d::Zero);
  ignition::math::Pose3d pose;
  pose.Pos().X(setpoint_.x);
  pose.Pos().Y(setpoint_.y);
  pose.Pos().Z(setpoint_.z);
  link_->SetWorldPose(pose);
}

void LinkPosition3dKinematicController::Stop()
{
  setpoint_ = Get();
}

void LinkPosition3dKinematicController::Enable(bool enable)
{
  link_->SetKinematic(enable);
}

void LinkPosition3dKinematicController::Set(geometry_msgs::Point position)
{
  setpoint_ = position;
}


} // namespace gazebo