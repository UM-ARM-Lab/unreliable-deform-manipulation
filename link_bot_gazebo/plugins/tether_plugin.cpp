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
    if (!sdf->HasElement("kP_pos")) {
      printf("using default kP_pos=%f\n", kP_pos_);
    }
    else {
      kP_pos_ = sdf->GetElement("kP_pos")->Get<double>();
    }

    if (!sdf->HasElement("kI_pos")) {
      printf("using default kI_pos=%f\n", kI_pos_);
    }
    else {
      kI_pos_ = sdf->GetElement("kI_pos")->Get<double>();
    }

    if (!sdf->HasElement("kD_pos")) {
      printf("using default kD_pos=%f\n", kD_pos_);
    }
    else {
      kD_pos_ = sdf->GetElement("kD_pos")->Get<double>();
    }

    if (!sdf->HasElement("max_vel")) {
      printf("using default max_vel=%f\n", max_vel_);
    }
    else {
      max_vel_ = sdf->GetElement("max_vel")->Get<double>();
    }

    if (!sdf->HasElement("kP_vel")) {
      printf("using default kP_vel=%f\n", kP_vel_);
    }
    else {
      kP_vel_ = sdf->GetElement("kP_vel")->Get<double>();
    }

    if (!sdf->HasElement("kI_vel")) {
      printf("using default kI_vel=%f\n", kI_vel_);
    }
    else {
      kI_vel_ = sdf->GetElement("kI_vel")->Get<double>();
    }

    if (!sdf->HasElement("kD_vel")) {
      printf("using default kD_vel=%f\n", kD_vel_);
    }
    else {
      kD_vel_ = sdf->GetElement("kD_vel")->Get<double>();
    }

    if (!sdf->HasElement("max_force")) {
      printf("using default max_force=%f\n", max_force_);
    }
    else {
      max_force_ = sdf->GetElement("max_force")->Get<double>();
    }

    if (sdf->HasElement("linkName")) {
      this->link_name_ = sdf->Get<std::string>("linkName");
    }
    else {
      ROS_FATAL_STREAM("This plugin requires a `linkName` parameter tag");
      return;
    }
  }

  model_ = parent;

  link_ = model_->GetLink(link_name_);
  if (!link_) {
    ROS_ERROR_NAMED("tether", "link not found");
    const std::vector<physics::LinkPtr> &links = model_->GetLinks();
    for (unsigned i = 0; i < links.size(); i++) {
      ROS_ERROR_STREAM_NAMED("tether", " -- Link " << i << ": " << links[i]->GetName());
    }
    return;
  }

  auto stop_bind = boost::bind(&TetherPlugin::OnStop, this, _1);
  auto stop_so =
      ros::SubscribeOptions::create<std_msgs::Empty>("/tether_stop", 1, stop_bind, ros::VoidPtr(), &queue_);
  auto enable_bind = boost::bind(&TetherPlugin::OnEnable, this, _1);
  auto enable_so = ros::SubscribeOptions::create<link_bot_gazebo::ModelsEnable>(
      "/tether_enable", 1, enable_bind, ros::VoidPtr(), &queue_);
  auto pos_action_bind = boost::bind(&TetherPlugin::OnAction, this, _1);
  auto pos_action_so = ros::SubscribeOptions::create<link_bot_gazebo::ModelsPoses>(
      "/tether_action", 1, pos_action_bind, ros::VoidPtr(), &queue_);

  ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
  enable_sub_ = ros_node_->subscribe(enable_so);
  action_sub_ = ros_node_->subscribe(pos_action_so);
  stop_sub_ = ros_node_->subscribe(stop_so);

  ros_queue_thread_ = std::thread(std::bind(&TetherPlugin::QueueThread, this));

  constexpr auto max_vel_integral{0};
  constexpr auto max_integral{0};
  pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);
  vel_pid_ = common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force_, -max_force_);

  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(boost::bind(&TetherPlugin::OnUpdate, this));

  target_pose_ = link_->WorldPose();
}

void TetherPlugin::OnUpdate()
{
  constexpr auto dt{0.001};

  auto const pos = link_->WorldPose().Pos();
  auto const pose_error =  target_pose_.Pos() - pos;

  auto const target_vel = pos_pid_.Update(-pose_error.Length(), dt);
  target_velocity_ = pose_error.Normalized() * target_vel;

  auto const vel = link_->WorldLinearVel();
  auto const vel_error = target_velocity_ - vel;
  auto const force_mag = vel_pid_.Update(-vel_error.Length(), dt);
  ignition::math::Vector3d force;
  force = vel_error.Normalized() * force_mag;

  // FIXME: this assumes the collision goemetry is a box
  auto const i{0u};
  auto const collision = link_->GetCollision(i);
  auto const box = boost::dynamic_pointer_cast<physics::BoxShape>(collision->GetShape());
  auto const z = box->Size().Z();

  if (enabled_) {
    ignition::math::Vector3d const push_position(0, 0, -z / 2.0 + 0.01);
    link_->AddForceAtRelativePosition(force, push_position);
  }
}

void TetherPlugin::OnStop(std_msgs::EmptyConstPtr const msg) { target_pose_ = link_->WorldPose(); }

void TetherPlugin::OnEnable(link_bot_gazebo::ModelsEnableConstPtr const msg)
{
  for (auto const &model_name : msg->model_names) {
    if (model_name == model_->GetScopedName()) {
      enabled_ = msg->enable;
    }
  }
}

void TetherPlugin::OnAction(link_bot_gazebo::ModelsPosesConstPtr const msg)
{
  for (auto const &action : msg->actions) {
    if (action.model_name == model_->GetScopedName()) {
      target_pose_.Pos().X(action.pose.position.x);
      target_pose_.Pos().Y(action.pose.position.y);
      target_pose_.Rot().X(action.pose.orientation.x);
      target_pose_.Rot().Y(action.pose.orientation.y);
      target_pose_.Rot().Z(action.pose.orientation.z);
      target_pose_.Rot().W(action.pose.orientation.w);
    }
  }
}

void TetherPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_->ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}
}  // namespace gazebo
