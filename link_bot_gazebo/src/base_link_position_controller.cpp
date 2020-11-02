#include <link_bot_gazebo/base_link_position_controller.h>
#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include <ros/console.h>

namespace gazebo
{

BaseLinkPositionController::BaseLinkPositionController(char const *plugin_name,
                                                       std::string scoped_link_name,
                                                       physics::WorldPtr world) :
    world_(world),
    plugin_name_(plugin_name)
{
  link_ = GetLink(plugin_name_, world, scoped_link_name);
}

geometry_msgs::Point BaseLinkPositionController::Get() const
{
  auto const pos = link_->WorldPose().Pos();
  geometry_msgs::Point out_pos;
  out_pos.x = pos.X();
  out_pos.y = pos.Y();
  out_pos.z = pos.Z();

  return out_pos;
}

}  // namespace gazebo
