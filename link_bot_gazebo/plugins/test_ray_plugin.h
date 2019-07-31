#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo {

class TestRayPlugin : public WorldPlugin {
  physics::WorldPtr world_;
  gazebo::physics::RayShapePtr ray;
  event::ConnectionPtr update_connection_;

 public:
  void Load(physics::WorldPtr world, sdf::ElementPtr _sdf) override;

  void OnUpdate();
};

}  // namespace gazebo
