#include <geometry_msgs/Point32.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ignition/math.hh>

#include "gazebo/gazebo.hh"
#include "gazebo/plugins/CameraPlugin.hh"
#include "link_bot_gazebo/CameraProjection.h"

namespace gazebo {
class MyCameraPlugin : public CameraPlugin {
 private:
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::ServiceServer projection_service_;
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

    auto xy_to_rowcol = [&](link_bot_gazebo::CameraProjectionRequest &req,
                            link_bot_gazebo::CameraProjectionResponse &res) {
      ignition::math::Vector3d vec{req.xyz.x, req.xyz.y, req.xyz.z};
      auto const point = camera->Project(vec);
      res.rowcol.x = point.X();
      res.rowcol.y = point.Y();
      return true;
    };

    ros_node_ = std::make_unique<ros::NodeHandle>("my_camera");

    {
      auto so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::CameraProjection>("xy_to_rowcol", xy_to_rowcol,
                                                                                        ros::VoidConstPtr(), &queue_);
      projection_service_ = ros_node_->advertiseService(so);
    }

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
}