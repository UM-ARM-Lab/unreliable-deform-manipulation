#include <geometry_msgs/Point32.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>

#include <ignition/math.hh>

#include "gazebo/gazebo.hh"
#include "gazebo/plugins/CameraPlugin.hh"

namespace gazebo {
class MyCameraPlugin : public CameraPlugin {
 private:
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;

 public:
  void Load(sensors::SensorPtr parent, sdf::ElementPtr sdf) override
  {
    CameraPlugin::Load(parent, sdf);

    if (!ros::isInitialized()) {
      auto argc = 0;
      char **argv = nullptr;
      ros::init(argc, argv, "my_camera", ros::init_options::NoSigintHandler);
    }

    ros_node_ = std::make_unique<ros::NodeHandle>("my_camera");

    ros_queue_thread_ = std::thread(std::bind(&MyCameraPlugin::QueueThread, this));
  }

  void QueueThread()
  {
    double constexpr timeout = 0.01;
    while (ros_node_->ok()) {
      queue_.callAvailable(ros::WallDuration(timeout));
    }
  }

  ~MyCameraPlugin() override
  {
    queue_.clear();
    queue_.disable();
    ros_node_->shutdown();
    ros_queue_thread_.join();
  }
};

// Register this plugin with the simulator
GZ_REGISTER_SENSOR_PLUGIN(MyCameraPlugin)
}  // namespace gazebo
