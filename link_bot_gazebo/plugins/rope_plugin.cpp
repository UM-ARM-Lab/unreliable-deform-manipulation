#include "rope_plugin.h"

#include <std_srvs/EmptyRequest.h>

#include <cstdio>
#include <gazebo/common/Time.hh>
#include <gazebo/common/Timer.hh>
#include <ignition/math/Vector3.hh>
#include <memory>
#include <sstream>

#include "enumerate.h"

namespace gazebo {

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-vararg"
void MultiLinkBotModelPlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf)
{
  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "rope_plugin", ros::init_options::NoSigintHandler);
  }

  auto set_config_bind = [this](auto &&req, auto &&res) { return SetRopeConfigCallback(req, res); };
  auto config_so = ros::AdvertiseServiceOptions::create<peter_msgs::SetRopeConfiguration>(
      "set_rope_config", set_config_bind, ros::VoidPtr(), &queue_);

  auto state_bind = [this](auto &&req, auto &&res) { return StateServiceCallback(req, res); };
  auto service_so = ros::AdvertiseServiceOptions::create<peter_msgs::LinkBotState>("link_bot_state", state_bind,
                                                                                   ros::VoidPtr(), &queue_);
  constexpr auto link_bot_service_name{"link_bot"};
  auto get_object_link_bot_bind = [this](auto &&req, auto &&res) { return GetObjectLinkBotCallback(req, res); };
  auto get_object_link_bot_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetObject>(
      link_bot_service_name, get_object_link_bot_bind, ros::VoidPtr(), &queue_);

  set_configuration_service_ = ros_node_.advertiseService(config_so);
  register_object_pub_ = ros_node_.advertise<std_msgs::String>("register_object", 10, true);
  state_service_ = ros_node_.advertiseService(service_so);
  get_object_link_bot_service_ = ros_node_.advertiseService(get_object_link_bot_so);
  objects_service_ = ros_node_.serviceClient<peter_msgs::GetObjects>("objects");

  ros_queue_thread_ = std::thread([this] { QueueThread(); });

  while (register_object_pub_.getNumSubscribers() < 1) {
  }

  {
    std_msgs::String register_object;
    register_object.data = link_bot_service_name;
    register_object_pub_.publish(register_object);
  }

  model_ = parent;

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

bool MultiLinkBotModelPlugin::StateServiceCallback(peter_msgs::LinkBotStateRequest &req,
                                                   peter_msgs::LinkBotStateResponse &res)
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
      res.points.emplace_back(pt);
      res.link_names.emplace_back(name);
    }
  }

  auto const link = model_->GetLink("head");
  geometry_msgs::Point pt;
  pt.x = link->WorldPose().Pos().X();
  pt.y = link->WorldPose().Pos().Y();
  res.points.emplace_back(pt);
  res.link_names.emplace_back("head");

  res.header.stamp = ros::Time::now();

  return true;
}

bool MultiLinkBotModelPlugin::GetObjectLinkBotCallback(peter_msgs::GetObjectRequest &req,
                                                       peter_msgs::GetObjectResponse &res)
{
  res.object.name = "link_bot";
  std::vector<float> state_vector;
  for (auto link_idx{1U}; link_idx <= num_links_; ++link_idx) {
    std::stringstream ss;
    ss << "link_" << link_idx;
    auto link_name = ss.str();
    auto const link = model_->GetLink(link_name);
    peter_msgs::NamedPoint named_point;
    float const x = link->WorldPose().Pos().X();
    float const y = link->WorldPose().Pos().Y();
    state_vector.push_back(x);
    state_vector.push_back(y);
    named_point.point.x = x;
    named_point.point.y = y;
    named_point.name = link_name;
    res.object.points.emplace_back(named_point);
  }

  auto const link = model_->GetLink("head");
  peter_msgs::NamedPoint head_point;
  geometry_msgs::Point pt;
  float const x = link->WorldPose().Pos().X();
  float const y = link->WorldPose().Pos().Y();
  state_vector.push_back(x);
  state_vector.push_back(y);
  head_point.point.x = x;
  head_point.point.y = y;
  res.object.state_vector = state_vector;
  head_point.name = "head";
  res.object.points.emplace_back(head_point);

  return true;
}

bool MultiLinkBotModelPlugin::SetRopeConfigCallback(peter_msgs::SetRopeConfigurationRequest &req,
                                                    peter_msgs::SetRopeConfigurationResponse &res)
{
  for (auto pair : enumerate(model_->GetJoints())) {
    auto const [i, joint] = pair;
    if (i < req.joint_angles.size()) {
      joint->SetPosition(0, req.joint_angles[i]);
    }
  }
  return true;
}

void MultiLinkBotModelPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

MultiLinkBotModelPlugin::~MultiLinkBotModelPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  ros_queue_thread_.join();
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(MultiLinkBotModelPlugin)
}  // namespace gazebo
