#include "rope_plugin.h"

#include <std_srvs/EmptyRequest.h>

#include <cstdio>
#include <gazebo/common/Time.hh>
#include <gazebo/common/Timer.hh>
#include <memory>
#include <sstream>

#include "enumerate.h"

namespace gazebo {

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-vararg"
void RopePlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf)
{
  model_ = parent;

  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized()) {
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

  auto state_bind = [this](auto &&req, auto &&res) { return StateService(req, res); };
  auto service_so = ros::AdvertiseServiceOptions::create<peter_msgs::LinkBotState>("link_bot_state", state_bind,
                                                                                   ros::VoidPtr(), &queue_);
  constexpr auto link_bot_service_name{"link_bot"};
  auto get_object_link_bot_bind = [this](auto &&req, auto &&res) { return GetObjectRope(req, res); };
  auto get_object_link_bot_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetObject>(
      link_bot_service_name, get_object_link_bot_bind, ros::VoidPtr(), &queue_);

  set_state_service_ = ros_node_.advertiseService(set_state_so);
  get_state_service_ = ros_node_.advertiseService(get_state_so);
  register_object_pub_ = ros_node_.advertise<std_msgs::String>("register_object", 10, true);
  state_service_ = ros_node_.advertiseService(service_so);
  get_object_link_bot_service_ = ros_node_.advertiseService(get_object_link_bot_so);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });

  gzwarn << "[" << model_->GetScopedName() << "] Waiting for object server\n";
  while (register_object_pub_.getNumSubscribers() < 1) {
  }

  {
    std_msgs::String register_object;
    register_object.data = link_bot_service_name;
    register_object_pub_.publish(register_object);
  }

  {
    if (!sdf->HasElement("rope_length")) {
      printf("using default rope length=%f\n", length_);
    }
    else {
      length_ = sdf->GetElement("rope_length")->Get<double>();
    }

    if (!sdf->HasElement("num_links")) {
      printf("using default num_links=%u\n", num_links_);
    }
    else {
      num_links_ = sdf->GetElement("num_links")->Get<unsigned int>();
    }
  }
  gzlog << "Rope Plugin finished initializing!\n";
}
#pragma clang diagnostic pop

bool RopePlugin::StateService(peter_msgs::LinkBotStateRequest &req, peter_msgs::LinkBotStateResponse &res)
{
  // get all links named "link_%d" where d is in [1, num_links)
  for (auto const &link : model_->GetLinks()) {
    auto const name = link->GetName();
    int link_idx;
    auto const n_matches = sscanf(name.c_str(), "link_%d", &link_idx);
    if (n_matches == 1 and link_idx >= 1 and link_idx <= num_links_) {
      geometry_msgs::Point pt;
      pt.x = link->WorldPose().Pos().X();
      pt.y = link->WorldPose().Pos().Y();
      pt.z = link->WorldPose().Pos().Z();
      res.points.emplace_back(pt);
      res.link_names.emplace_back(name);
    }
  }

  res.header.stamp = ros::Time::now();

  return true;
}

bool RopePlugin::GetObjectRope(peter_msgs::GetObjectRequest &req, peter_msgs::GetObjectResponse &res)
{
  res.object.name = "link_bot";
  for (auto link_idx{1U}; link_idx <= num_links_; ++link_idx) {
    std::stringstream ss;
    ss << "link_" << link_idx;
    auto link_name = ss.str();
    auto const link = model_->GetLink(link_name);
    peter_msgs::NamedPoint named_point;
    float const x = link->WorldPose().Pos().X();
    float const y = link->WorldPose().Pos().Y();
    float const z = link->WorldPose().Pos().Z();
    res.object.state_vector.push_back(x);
    res.object.state_vector.push_back(y);
    res.object.state_vector.push_back(z);
    named_point.point.x = x;
    named_point.point.y = y;
    named_point.point.z = z;
    named_point.name = link_name;
    res.object.points.emplace_back(named_point);
  }

  return true;
}

bool RopePlugin::SetRopeState(peter_msgs::SetRopeStateRequest &req, peter_msgs::SetRopeStateResponse &res)
{
  for (auto pair : enumerate(model_->GetJoints())) {
    auto const &[i, joint] = pair;
    if (i < req.joint_angles_axis1.size()) {
      joint->SetPosition(0, req.joint_angles_axis1[i]);
      joint->SetPosition(1, req.joint_angles_axis2[i]);
    }
  }
  ignition::math::Pose3d pose{req.model_pose.position.x,    req.model_pose.position.y,    req.model_pose.position.z,
                              req.model_pose.orientation.w, req.model_pose.orientation.x, req.model_pose.orientation.y,
                              req.model_pose.orientation.z};
  model_->SetWorldPose(pose);
  return true;
}

bool RopePlugin::GetRopeState(peter_msgs::GetRopeStateRequest &req, peter_msgs::GetRopeStateResponse &res)
{
  for (auto const &joint : model_->GetJoints()) {
    res.joint_angles_axis1.push_back(joint->Position(0));
    res.joint_angles_axis2.push_back(joint->Position(1));
  }
  for (auto const &link : model_->GetLinks()) {
    auto const name = link->GetName();
    int link_idx;
    auto const n_matches = sscanf(name.c_str(), "link_%d", &link_idx);
    // TODO: is sccanf really the most modern way to do this? use boost regex?
    if (n_matches == 1 and link_idx >= 1 and link_idx <= num_links_) {
      geometry_msgs::Point pt;
      pt.x = link->WorldPose().Pos().X();
      pt.y = link->WorldPose().Pos().Y();
      pt.z = link->WorldPose().Pos().Z();
      res.points.emplace_back(pt);
    }
  }
  res.model_pose.position.x = model_->WorldPose().Pos().X();
  res.model_pose.position.y = model_->WorldPose().Pos().Y();
  res.model_pose.position.z = model_->WorldPose().Pos().Z();
  res.model_pose.orientation.x = model_->WorldPose().Rot().X();
  res.model_pose.orientation.y = model_->WorldPose().Rot().Y();
  res.model_pose.orientation.z = model_->WorldPose().Rot().Z();
  res.model_pose.orientation.w = model_->WorldPose().Rot().W();
  return true;
}

void RopePlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok()) {
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