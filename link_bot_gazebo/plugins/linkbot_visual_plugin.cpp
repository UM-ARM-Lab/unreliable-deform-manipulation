#include "linkbot_visual_plugin.h"

namespace gazebo {

LinkBotVisualPlugin::LinkBotVisualPlugin() {}

void LinkBotVisualPlugin::Load(rendering::VisualPtr _visual, sdf::ElementPtr _sdf)
{
  this->updateConnection = event::Events::ConnectPreRender(std::bind(&LinkBotVisualPlugin::Update, this));
}
void LinkBotVisualPlugin::Update()
{
//  visual->CreateDynamicLine()
}

GZ_REGISTER_VISUAL_PLUGIN(LinkBotVisualPlugin)

};  // namespace gazebo
