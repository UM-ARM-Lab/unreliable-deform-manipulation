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
  LinkPosition3dKinematicController(char const *plugin_name,
                                    std::string scoped_link_name,
                                    physics::WorldPtr world);

  void Stop() override;

  void Enable(bool enable) override;

  void Set(geometry_msgs::Point position) override;

  [[nodiscard]] geometry_msgs::Point Get() const override;


  physics::WorldPtr world_;
  physics::ModelPtr model_;
  char const *plugin_name_;
};

}  // namespace gazebo