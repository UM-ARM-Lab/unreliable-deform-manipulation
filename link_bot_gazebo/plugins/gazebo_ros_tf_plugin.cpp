#include "gazebo_ros_tf_plugin.h"

#define create_service_options(type, name, bind)                                                                       \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

#define create_service_options_private(type, name, bind)                                                               \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &private_queue_)

namespace gazebo
{
GZ_REGISTER_WORLD_PLUGIN(GazeboRosTfPlugin)

GazeboRosTfPlugin::~GazeboRosTfPlugin()
{
  queue_.clear();
  queue_.disable();
  ph_->shutdown();
  callback_queue_thread_.join();
}

void GazeboRosTfPlugin::Load(physics::WorldPtr world, sdf::ElementPtr /*sdf*/)
{
  world_ = world;
  victor_ = world_->ModelByName("victor_and_rope");
  table_ = world_->ModelByName("table");

  if (!victor_)
  {
    ROS_ERROR("could not find model by name victor_and_rope");
  }
  if (!table_)
  {
    ROS_ERROR("could not find model by name table");
  }

  // setup ROS stuff
  if (!ros::isInitialized())
  {
    int argc = 0;
    ros::init(argc, nullptr, "ros_tf_plugin", ros::init_options::NoSigintHandler);
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
  if (!victor_)
  {
    return;
  }

  std::string const link_name = "victor_and_rope::victor::victor_root";
  auto const victor_root_link = victor_->GetLink(link_name);
  if (!victor_root_link)
  {
    ROS_ERROR_STREAM("there is no link " << link_name);
    ROS_ERROR_STREAM("possible link names are:");
    for (auto const l : victor_->GetLinks())
    {
      ROS_ERROR_STREAM(l->GetScopedName());
    }
    return;
  }

  auto const victor_root_pose = victor_root_link->WorldPose();

  geometry_msgs::TransformStamped victor_root_tf;
  victor_root_tf.header.frame_id = "world";
  victor_root_tf.header.stamp = ros::Time::now();
  victor_root_tf.child_frame_id = "victor_root";
  victor_root_tf.transform.translation.x = victor_root_pose.Pos().X();
  victor_root_tf.transform.translation.y = victor_root_pose.Pos().Y();
  victor_root_tf.transform.translation.z = victor_root_pose.Pos().Z();
  victor_root_tf.transform.rotation.w = victor_root_pose.Rot().W();
  victor_root_tf.transform.rotation.x = victor_root_pose.Rot().X();
  victor_root_tf.transform.rotation.y = victor_root_pose.Rot().Y();
  victor_root_tf.transform.rotation.z = victor_root_pose.Rot().Z();
  tb_.sendTransform(victor_root_tf);

  // FIXME: This assumes that the entire table body goes "up" from the base
  if (table_)
  {
    std::string const table_link_name = "table::surface";
    auto const table_link = table_->GetLink(table_link_name);
    if (!table_link)
    {
      ROS_ERROR_STREAM("no link " << table_link_name);
      ROS_ERROR_STREAM("possible link names are:");
      for (auto const l : table_->GetLinks())
      {
        ROS_ERROR_STREAM(l->GetScopedName());
      }
      return;
    }

    auto const table_surface_pose = table_link->WorldPose();
    geometry_msgs::TransformStamped table_surface_tf;
    table_surface_tf.header.frame_id = "world";
    table_surface_tf.header.stamp = victor_root_tf.header.stamp;
    table_surface_tf.child_frame_id = "table_surface";
    table_surface_tf.transform.translation.x = table_surface_pose.Pos().X();
    table_surface_tf.transform.translation.y = table_surface_pose.Pos().Y();
    table_surface_tf.transform.translation.z = table_surface_pose.Pos().Z();
    table_surface_tf.transform.rotation.w = table_surface_pose.Rot().W();
    table_surface_tf.transform.rotation.x = table_surface_pose.Rot().X();
    table_surface_tf.transform.rotation.y = table_surface_pose.Rot().Y();
    table_surface_tf.transform.rotation.z = table_surface_pose.Rot().Z();
    tb_.sendTransform(table_surface_tf);
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

}  // namespace gazebo