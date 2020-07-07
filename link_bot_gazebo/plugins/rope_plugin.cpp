#include "rope_plugin.h"

#include <std_srvs/EmptyRequest.h>

#include <boost/regex.hpp>

#include <cstdio>
#include <gazebo/common/Time.hh>
#include <gazebo/common/Timer.hh>
#include <memory>
#include <sstream>

#include "enumerate.h"

namespace gazebo
{
void RopePlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf)
{
  model_ = parent;

  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized())
  {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "rope_plugin", ros::init_options::NoSigintHandler);
  }

  auto set_state_bind = [this](auto &&req, auto &&res) { return SetRopeState(req, res); };
  auto set_state_so = ros::AdvertiseServiceOptions::create<peter_msgs::SetRopeState>("set_rope_state", set_state_bind,
                                                                                     ros::VoidPtr(), &queue_);

  auto get_state_bind = [this](auto &&req, auto &&res) { return GetRopeState(req, res); };
  auto get_state_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetRopeState>("get_rope_state", get_state_bind,
                                                                                     ros::VoidPtr(), &queue_);

  set_state_service_ = ros_node_.advertiseService(set_state_so);
  get_state_service_ = ros_node_.advertiseService(get_state_so);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });

  {
    if (!sdf->HasElement("rope_length"))
    {
      printf("using default rope length=%f\n", length_);
    }
    else
    {
      length_ = sdf->GetElement("rope_length")->Get<double>();
    }

    if (!sdf->HasElement("num_links"))
    {
      printf("using default num_links=%u\n", num_links_);
    }
    else
    {
      num_links_ = sdf->GetElement("num_links")->Get<unsigned int>();
    }
  }
  ROS_INFO("Rope Plugin finished initializing!");
}

bool RopePlugin::SetRopeState(peter_msgs::SetRopeStateRequest &req, peter_msgs::SetRopeStateResponse &)
{
  for (auto pair : enumerate(model_->GetJoints()))
  {
    auto const &[i, joint] = pair;
    if (i < req.joint_angles_axis1.size())
    {
      joint->SetPosition(0, req.joint_angles_axis1[i]);
      joint->SetPosition(1, req.joint_angles_axis2[i]);
    }
  }
  ignition::math::Pose3d pose{ req.model_pose.position.x,    req.model_pose.position.y,    req.model_pose.position.z,
                               req.model_pose.orientation.w, req.model_pose.orientation.x, req.model_pose.orientation.y,
                               req.model_pose.orientation.z };
  model_->SetWorldPose(pose);
  return true;
}

bool RopePlugin::GetRopeState(peter_msgs::GetRopeStateRequest &, peter_msgs::GetRopeStateResponse &res)
{
  static peter_msgs::GetRopeStateResponse previous_res;
  static auto initialized = false;

  for (auto const &joint : model_->GetJoints())
  {
    res.joint_angles_axis1.push_back(joint->Position(0));
    res.joint_angles_axis2.push_back(joint->Position(1));
  }
  for (auto const &pair : enumerate(model_->GetLinks()))
  {
    auto const &[i, link] = pair;
    auto const name = link->GetName();
    boost::regex e(".*rope_link_\\d+");
    if (boost::regex_match(name, e))
    {
      // ROS_INFO_STREAM("using link with name " << name);
      geometry_msgs::Point pt;
      pt.x = link->WorldPose().Pos().X();
      pt.y = link->WorldPose().Pos().Y();
      pt.z = link->WorldPose().Pos().Z();
      res.positions.emplace_back(pt);

      geometry_msgs::Point velocity;
      if (initialized)
      {
        velocity.x = pt.x - previous_res.positions[i].x;
        velocity.y = pt.y - previous_res.positions[i].y;
        velocity.z = pt.z - previous_res.positions[i].z;
      }
      else
      {
        velocity.x = 0;
        velocity.y = 0;
        velocity.z = 0;
      }
      res.velocities.emplace_back(velocity);
    }
    else
    {
      // ROS_INFO_STREAM("skipping link with name " << name);
    }
  }
  res.model_pose.position.x = model_->WorldPose().Pos().X();
  res.model_pose.position.y = model_->WorldPose().Pos().Y();
  res.model_pose.position.z = model_->WorldPose().Pos().Z();
  res.model_pose.orientation.x = model_->WorldPose().Rot().X();
  res.model_pose.orientation.y = model_->WorldPose().Rot().Y();
  res.model_pose.orientation.z = model_->WorldPose().Rot().Z();
  res.model_pose.orientation.w = model_->WorldPose().Rot().W();

  previous_res = res;
  initialized = true;

  return true;
}

void RopePlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

RopePlugin::~RopePlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  ros_queue_thread_.join();
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(RopePlugin)
}  // namespace gazebo
