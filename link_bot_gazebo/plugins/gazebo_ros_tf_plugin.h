#ifndef GAZEBO_ROS_TF_PLUGIN_H
#define GAZEBO_ROS_TF_PLUGIN_H

#include <thread>

#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo
{
class GazeboRosTfPlugin : public WorldPlugin
{
public:
  ~GazeboRosTfPlugin() override;

  void Load(physics::WorldPtr world, sdf::ElementPtr sdf) override;

private:
  std::unique_ptr<ros::NodeHandle> ph_;
  ros::CallbackQueue queue_;
  std::thread callback_queue_thread_;
  std::thread periodic_event_thread_;

  tf2_ros::TransformBroadcaster tb_;
  tf2_ros::StaticTransformBroadcaster stb_;

  event::ConnectionPtr update_connection_;
  physics::WorldPtr world_;
  physics::Model_V models_;
  std::string frame_id_;

  void PrivateQueueThread();

  void PeriodicUpdate();
};
}  // namespace gazebo

#endif
