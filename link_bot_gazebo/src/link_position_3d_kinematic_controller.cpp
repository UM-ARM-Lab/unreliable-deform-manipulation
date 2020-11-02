#include <link_bot_gazebo/link_position_3d_kinematic_controller.h>

namespace gazebo
{

LinkPosition3dKinematicController::LinkPosition3dKinematicController(char const *plugin_name,
                                                                     std::string scoped_link_name,
                                                                     physics::WorldPtr world) :
    BaseLinkPositionController(plugin_name, scoped_link_name, world)
{}

void LinkPosition3dKinematicController::Stop()
{}

void LinkPosition3dKinematicController::Enable(bool enable)
{}

void LinkPosition3dKinematicController::Set(geometry_msgs::Point position)
{}


} // namespace gazebo