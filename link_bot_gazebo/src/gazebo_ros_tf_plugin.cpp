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
    } else
    {
      frame_id_ = sdf->GetElement("frame")->Get<std::string>();
      ROS_INFO_STREAM("using non-standard frame " << frame_id_);
    }
  }

  ph_ = std::make_unique<ros::NodeHandle>("gazebo_ros_tf_plugin");
  callback_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

  ROS_INFO("Finished loading ROS TF plugin!\n");

  auto periodic_update_func = [this]
  {
    while (true)
    {
      PeriodicUpdate();
      ros::Duration(0.001).sleep();
    }
  };
  periodic_event_thread_ = std::thread(periodic_update_func);
}

void GazeboRosTfPlugin::PeriodicUpdate()
{
  auto const now = ros::Time::now();
  auto const dt = now - last_tf_update_;
  if (dt == ros::Duration(0))
  {
    return;
  }

  for (auto const &model: world_->Models())
  {
    for (auto const &link: model->GetLinks())
    {
      auto const model_name = model->GetName();
      auto const link_name = link->GetName();
      auto const frame_id = get_frame_id(model_name, link_name);

      auto const pose = link->WorldPose();
      geometry_msgs::TransformStamped transform_msg;
      transform_msg.header.frame_id = frame_id_;
      transform_msg.header.stamp = now;
      transform_msg.child_frame_id = frame_id;
      transform_msg.transform.translation.x = pose.Pos().X();
      transform_msg.transform.translation.y = pose.Pos().Y();
      transform_msg.transform.translation.z = pose.Pos().Z();
      transform_msg.transform.rotation.w = pose.Rot().W();
      transform_msg.transform.rotation.x = pose.Rot().X();
      transform_msg.transform.rotation.y = pose.Rot().Y();
      transform_msg.transform.rotation.z = pose.Rot().Z();
      tb_.sendTransform(transform_msg);
    }

    last_tf_update_ = now;
  }
}

std::string GazeboRosTfPlugin::get_frame_id(const std::string &model_name, const std::string &link_name) const
{
  auto frame_id = ros::names::append(model_name, link_name);
  frame_id = ros::names::append("gazebo", frame_id);
  return frame_id;
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