#include "tether_plugin.h"

namespace gazebo {
GZ_REGISTER_MODEL_PLUGIN(TetherPlugin);

TetherPlugin::~TetherPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_->shutdown();
  ros_queue_thread_.join();
}

void TetherPlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf)
{
  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "tether_plugin", ros::init_options::NoSigintHandler);
    return;
  }

  // Get sdf parameters
  {
    if (!sdf->HasElement("num_links")) {
      printf("using default num_links=%u\n", num_links_);
    }
    else {
      num_links_ = sdf->GetElement("num_links")->Get<unsigned int>();
    }
  }

  model_ = parent;

  auto state_bind = boost::bind(&TetherPlugin::StateServiceCallback, this, _1, _2);
  auto state_service_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::GetObject>("/tether", state_bind,
                                                                                       ros::VoidPtr(), &queue_);
  ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
  state_service_ = ros_node_->advertiseService(state_service_so);
  register_tether_pub_ = ros_node_->advertise<std_msgs::String>("/register_object", 10, true);

  ros_queue_thread_ = std::thread(std::bind(&TetherPlugin::QueueThread, this));

  std_msgs::String register_object;
  register_object.data = "tether";
  register_tether_pub_.publish(register_object);

  // plus 1 because we want both end points inclusive
  ros_node_->setParam("/tether/n_state", static_cast<int>((num_links_ + 1) * 2));
}

bool TetherPlugin::StateServiceCallback(link_bot_gazebo::GetObjectRequest &req,
                                        link_bot_gazebo::GetObjectResponse &res)
{
  res.object.name = "tether";
  for (auto link_idx{1U}; link_idx <= num_links_; ++link_idx) {
    std::stringstream ss;
    ss << "link_" << link_idx;
    auto link_name = ss.str();
    auto const link = model_->GetLink(link_name);
    link_bot_gazebo::NamedPoint named_point;
    named_point.point.x = link->WorldPose().Pos().X();
    named_point.point.y = link->WorldPose().Pos().Y();
    named_point.name = link_name;
    res.object.points.emplace_back(named_point);
  }

  auto const link = model_->GetLink("head");
  link_bot_gazebo::NamedPoint head_point;
  head_point.point.x = link->WorldPose().Pos().X();
  head_point.point.y = link->WorldPose().Pos().Y();
  head_point.name = "head";
  res.object.points.emplace_back(head_point);


  return true;
}

void TetherPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_->ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}
}  // namespace gazebo
