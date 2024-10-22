#pragma once

#include <geometry_msgs/Point.h>
#include <gazebo/physics/World.hh>
#include <string>
#include <link_bot_gazebo/base_link_position_controller.h>

namespace gazebo
{
class LinkPosition3dKinematicController : public BaseLinkPositionController
{
 public:
  LinkPosition3dKinematicController(char const *plugin_name, physics::LinkPtr link);

  void Update(ignition::math::Vector3d const &setpoint) override;


  void OnEnable(bool enable) override;
};

}  // namespace gazebo