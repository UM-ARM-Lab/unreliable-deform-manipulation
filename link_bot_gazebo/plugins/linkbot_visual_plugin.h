#pragma once

#include <gazebo/common/Plugin.hh>
#include <gazebo/rendering/rendering.hh>

namespace gazebo {

class LinkBotVisualPlugin : public VisualPlugin {
 public:
  LinkBotVisualPlugin();

  virtual void Load(rendering::VisualPtr _visual, sdf::ElementPtr _sdf);

  void Update();

  rendering::VisualPtr visual;

  event::ConnectionPtr updateConnection;
};

};  // namespace gazebo
