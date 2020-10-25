#include <boost/algorithm/string.hpp>

#include "gazebo_ros_tf_plugin.h"

#define create_service_options(type, name, bind)                                                                       \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

#define create_service_options_private(type, name, bind)                                                               \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &private_queue_)

namespace gazebo
{
GazeboRosTfPlugin::~GazeboRosTfPlugin()
{
  queue_.clear();
  queue_.disable();
  ph_->shutdown();
  callback_queue_thread_.join();
}

void GazeboRosTfPlugin::Load(physics::WorldPtr world, sdf::ElementPtr sdf)
{
  world_ = world;

  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                         << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
    return;
  }

  {
    if (!sdf->HasElement("frame"))
    {
      frame_id_ = "robot_root";
      ROS_INFO_STREAM("using default frame " << frame_id_);
    }
    else
    {
      frame_id_ = sdf->GetElement("frame")->Get<std::string>();
      ROS_INFO_STREAM("using non-standard frame " << frame_id_);
    }

    // Find the links we're supposed to publish TF for
    if (!sdf->HasElement("model_names"))
    {
      ROS_WARN("No element model_names, this plugin will do nothing.");
    }
    else
    {
      auto const model_names_str = sdf->GetElement("model_names")->Get<std::string>();
      boost::split(model_names_, model_names_str, boost::is_any_of(" "));
    }
  }

  ph_ = std::make_unique<ros::NodeHandle>("ros_tf_plugin");
  callback_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

  ROS_INFO("Finished loading ROS TF plugin!\n");

  periodic_event_thread_ = std::thread([this] {
    while (true)
    {
      PeriodicUpdate();
      usleep(100000);
    }
  });
}

void GazeboRosTfPlugin::PeriodicUpdate()
{
  for (auto const model_name : model_names_)
  {
    auto const model = world_->ModelByName(model_name);
    if (!model)
    {
      ROS_WARN_STREAM_THROTTLE(1, "could not find model with name " << model_name);
      ROS_WARN_STREAM_THROTTLE(1, "possible model names are:");
      for (auto const available_model : world_->Models())
      {
        ROS_WARN_STREAM_THROTTLE(1, available_model->GetScopedName() << "[ scoped: " << available_model->GetScopedName() << " ]");
      }
      continue;
    }

    auto const pose = model->WorldPose();
    geometry_msgs::TransformStamped transform_msg;
    transform_msg.header.frame_id = frame_id_;
    transform_msg.header.stamp = ros::Time::now();
    transform_msg.child_frame_id = model->GetName() + "_gazebo";
    transform_msg.transform.translation.x = pose.Pos().X();
    transform_msg.transform.translation.y = pose.Pos().Y();
    transform_msg.transform.translation.z = pose.Pos().Z();
    transform_msg.transform.rotation.w = pose.Rot().W();
    transform_msg.transform.rotation.x = pose.Rot().X();
    transform_msg.transform.rotation.y = pose.Rot().Y();
    transform_msg.transform.rotation.z = pose.Rot().Z();
    tb_.sendTransform(transform_msg);
  }
}

void GazeboRosTfPlugin::PrivateQueueThread()
{
  double constexpr timeout = 0.01;
  while (ph_->ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

GZ_REGISTER_WORLD_PLUGIN(GazeboRosTfPlugin)

}  // namespace gazebo