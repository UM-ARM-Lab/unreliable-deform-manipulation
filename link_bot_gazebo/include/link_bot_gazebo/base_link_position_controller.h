#pragma once

#include <geometry_msgs/Point.h>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/Link.hh>

namespace gazebo
{

class BaseLinkPositionController
{
 public:
  BaseLinkPositionController(char const *plugin_name,
                             std::string scoped_link_name,
                             physics::WorldPtr world);

  virtual void Update()
  {};

  virtual void Stop() = 0;

  virtual void Enable(bool enable) = 0;

  virtual void Set(geometry_msgs::Point position) = 0;

  [[nodiscard]] virtual geometry_msgs::Point Get() const;

  physics::WorldPtr world_;
  char const *plugin_name_;
  physics::LinkPtr link_;
};

}
