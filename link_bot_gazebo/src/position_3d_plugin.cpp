#include "position_3d_plugin.h"

#include <link_bot_gazebo/mymath.hpp>

#include <link_bot_gazebo/link_position_3d_pid_controller.h>
#include <link_bot_gazebo/link_position_3d_kinematic_controller.h>
#include <ros/console.h>
#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include <link_bot_gazebo/kinematic_robotiq_3f_gripper_plugin.h>


#define create_service_options(type, name, bind)                                                                       \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

constexpr static auto const PLUGIN_NAME = "position_3d_plugin";

namespace gazebo
{
GZ_REGISTER_WORLD_PLUGIN(Position3dPlugin)

Position3dPlugin::~Position3dPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  private_ros_node_->shutdown();
  ros_queue_thread_.join();
  private_ros_queue_thread_.join();
}

void Position3dPlugin::Load(physics::WorldPtr world, sdf::ElementPtr sdf)
{
  world_ = world;

  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. ");
    return;
  }

  CreateServices();

  // FIXME: none of this code is threadsafe
}

void Position3dPlugin::CreateServices()
{
  private_ros_node_ = std::make_unique<ros::NodeHandle>("position_3d_plugin");

  {
    auto stop_bind = [this](peter_msgs::Position3DStopRequest &req,
                            peter_msgs::Position3DStopResponse &res) { return OnStop(req, res); };
    auto stop_so = create_service_options(peter_msgs::Position3DStop, "stop", stop_bind);
    stop_service_ = private_ros_node_->advertiseService(stop_so);
  }

  {
    auto register_bind = [this](peter_msgs::RegisterPosition3DControllerRequest &req,
                                peter_msgs::RegisterPosition3DControllerResponse &res)
    {
      return OnRegister(req, res);
    };
    auto register_so = create_service_options(peter_msgs::RegisterPosition3DController, "register", register_bind);
    register_service_ = private_ros_node_->advertiseService(register_so);
  }

  {
    auto unregister_bind = [this](peter_msgs::UnregisterPosition3DControllerRequest &req,
                                  peter_msgs::UnregisterPosition3DControllerResponse &res)
    {
      return OnUnregister(req, res);
    };
    auto unregister_so = create_service_options(peter_msgs::UnregisterPosition3DController, "unregister",
                                                unregister_bind);
    unregister_service_ = private_ros_node_->advertiseService(unregister_so);
  }

  {
    auto enable_bind = [this](peter_msgs::Position3DEnableRequest &req, peter_msgs::Position3DEnableResponse &res)
    {
      return OnEnable(req, res);
    };
    auto enable_so = create_service_options(peter_msgs::Position3DEnable, "enable", enable_bind);
    enable_service_ = private_ros_node_->advertiseService(enable_so);
  }

  {
    auto follow_bind = [this](peter_msgs::Position3DFollowRequest &req, peter_msgs::Position3DFollowResponse &res)
    {
      return OnFollow(req, res);
    };
    auto follow_so = create_service_options(peter_msgs::Position3DFollow, "follow", follow_bind);
    follow_service_ = private_ros_node_->advertiseService(follow_so);
  }

  {
    auto pos_set_bind = [this](peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res)
    {
      return OnSet(req, res);
    };
    auto pos_set_so = create_service_options(peter_msgs::Position3DAction, "set", pos_set_bind);
    set_service_ = private_ros_node_->advertiseService(pos_set_so);
  }

  {
    auto get_pos_bind = [this](peter_msgs::GetPosition3DRequest &req, peter_msgs::GetPosition3DResponse &res)
    {
      return GetPos(req, res);
    };
    auto get_pos_so = create_service_options(peter_msgs::GetPosition3D, "get", get_pos_bind);
    get_position_service_ = private_ros_node_->advertiseService(get_pos_so);
  }

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
  private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

  auto update = [this](common::UpdateInfo const & /*info*/) { OnUpdate(); };
  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(update);
}


bool Position3dPlugin::OnRegister(peter_msgs::RegisterPosition3DControllerRequest &req,
                                  peter_msgs::RegisterPosition3DControllerResponse &res)
{
  auto const link = GetLink(PLUGIN_NAME, world_, req.scoped_link_name);
  if (!link)
  {
    return true;
  }

  if (req.controller_type == "pid")
  {
    // TODO: where do PID params come from?
    controllers_map_.emplace(req.scoped_link_name,
                             std::make_unique<LinkPosition3dPIDController>(PLUGIN_NAME,
                                                                           link,
                                                                           1.0,
                                                                           1.0,
                                                                           1.0,
                                                                           1.0,
                                                                           true));
    ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "registered PID controller for link " << req.scoped_link_name);
  } else if (req.controller_type == "kinematic")
  {
    controllers_map_.emplace(req.scoped_link_name,
                             std::make_unique<LinkPosition3dKinematicController>(PLUGIN_NAME, link));
    ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "registered kinematic controller for link " << req.scoped_link_name);
  } else
  {
    ROS_ERROR_STREAM_NAMED(PLUGIN_NAME, "unimplemented controller type " << req.controller_type);
  }

  return true;
}

bool Position3dPlugin::OnUnregister(peter_msgs::UnregisterPosition3DControllerRequest &req,
                                    peter_msgs::UnregisterPosition3DControllerResponse &res)
{
  auto const num_removed = controllers_map_.erase(req.scoped_link_name);
  res.success = num_removed > 0;
  return true;
}

bool Position3dPlugin::OnStop(peter_msgs::Position3DStopRequest &req, peter_msgs::Position3DStopResponse &res)
{
  (void) res; // unused

  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend())
  {
    it->second->OnStop();
  } else
  {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }
  return true;
}

bool Position3dPlugin::OnEnable(peter_msgs::Position3DEnableRequest &req, peter_msgs::Position3DEnableResponse &res)
{
  (void) res;

  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend())
  {
    it->second->OnEnable(req.enable);
  } else
  {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }
  return true;
}

bool Position3dPlugin::OnFollow(peter_msgs::Position3DFollowRequest &req, peter_msgs::Position3DFollowResponse &res)
{
  (void) res;
  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend())
  {
    it->second->OnFollow(req.frame_id);
  } else
  {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }
  return true;
}

bool Position3dPlugin::OnSet(peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res)
{
  (void) res;
  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend())
  {
    it->second->OnEnable(true);
    it->second->Set(req.position);
  } else
  {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }
  return true;
}

bool Position3dPlugin::GetPos(peter_msgs::GetPosition3DRequest &req, peter_msgs::GetPosition3DResponse &res)
{
  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend())
  {
    auto const pos = it->second->Get();
    if (pos)
    {
      res.pos = ign_vector_3d_to_point(*pos);
    }
  } else
  {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }
  return true;
}

void Position3dPlugin::OnUpdate()
{
  for (auto &[k, v] : controllers_map_)
  {
    v->OnUpdate();
  }
}

void Position3dPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void Position3dPlugin::PrivateQueueThread()
{
  double constexpr timeout = 0.01;
  while (private_ros_node_->ok())
  {
    private_queue_.callAvailable(ros::WallDuration(timeout));
  }
}

}  // namespace gazebo