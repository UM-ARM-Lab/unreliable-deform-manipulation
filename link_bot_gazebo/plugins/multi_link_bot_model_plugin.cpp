#include "multi_link_bot_model_plugin.h"

#include <exception>
#include <ignition/math/Vector3.hh>
#include <memory>

namespace gazebo {

bool in_contact(msgs::Contacts const &contacts)
{
  for (auto i{0u}; i < contacts.contact_size(); ++i) {
    auto const &contact = contacts.contact(i);
    std::cout << contact.collision1() << " " << contact.collision2() << '\n';
    if (contact.collision1() == "link_bot::head::head_collision" and
        contact.collision2() != "ground_plane::link::collision") {
      return true;
    }
    if (contact.collision2() == "link_bot::head::head_collision" and
        contact.collision1() != "ground_plane::link::collision") {
      return true;
    }
  }
  return false;
}

void MultiLinkBotModelPlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf)
{
  // Make sure the ROS node for Gazebo has already been initalized
  // Initialize ros, if it has not already bee initialized.
  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "multi_link_bot_model_plugin", ros::init_options::NoSigintHandler);
  }

  ros_node_ = std::make_unique<ros::NodeHandle>("multi_link_bot_model_plugin");

  auto joy_bind = boost::bind(&MultiLinkBotModelPlugin::OnJoy, this, _1);
  auto joy_so = ros::SubscribeOptions::create<sensor_msgs::Joy>("/joy", 1, joy_bind, ros::VoidPtr(), &queue_);
  auto vel_action_bind = boost::bind(&MultiLinkBotModelPlugin::OnAction, this, _1);
  auto vel_action_so = ros::SubscribeOptions::create<link_bot_gazebo::MultiLinkBotPositionAction>(
      "/multi_link_bot_position_action", 1, vel_action_bind, ros::VoidPtr(), &queue_);
  auto config_bind = boost::bind(&MultiLinkBotModelPlugin::OnConfiguration, this, _1);
  auto config_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotConfiguration>(
      "/link_bot_configuration", 1, config_bind, ros::VoidPtr(), &queue_);
  auto state_bind = boost::bind(&MultiLinkBotModelPlugin::StateServiceCallback, this, _1, _2);
  auto service_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::LinkBotState>("/link_bot_state", state_bind,
                                                                                        ros::VoidPtr(), &queue_);

  joy_sub_ = ros_node_->subscribe(joy_so);
  action_sub_ = ros_node_->subscribe(vel_action_so);
  config_sub_ = ros_node_->subscribe(config_so);
  state_service_ = ros_node_->advertiseService(service_so);

  ros_queue_thread_ = std::thread(std::bind(&MultiLinkBotModelPlugin::QueueThread, this));

  model_ = parent;

  if (!sdf->HasElement("kP")) {
    printf("using default kP=%f\n", kP_);
  }
  else {
    kP_ = sdf->GetElement("kP")->Get<double>();
  }

  if (!sdf->HasElement("kI")) {
    printf("using default kI=%f\n", kI_);
  }
  else {
    kI_ = sdf->GetElement("kI")->Get<double>();
  }

  if (!sdf->HasElement("kD")) {
    printf("using default kD=%f\n", kD_);
  }
  else {
    kD_ = sdf->GetElement("kD")->Get<double>();
  }

  if (!sdf->HasElement("gripper1_link")) {
    throw std::invalid_argument("no gripper1_link tag provided");
  }

  auto const &gripper1_link_name = sdf->GetElement("gripper1_link")->Get<std::string>();
  gripper1_link_ = model_->GetLink(gripper1_link_name);


  if (sdf->HasElement("gripper2_link")) {
    auto const &gripper2_link_name = sdf->GetElement("gripper1_link")->Get<std::string>();
    gripper2_link_ = model_->GetLink(gripper2_link_name);
  }

  ROS_INFO("kP=%f, kI=%f, kD=%f", kP_, kI_, kD_);

  updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&MultiLinkBotModelPlugin::OnUpdate, this));
  constexpr auto max_force{125};
  constexpr auto max_integral{100};
  gripper1_x_pos_pid_ = common::PID(kP_, kI_, kD_, max_integral, -max_integral, max_force, -max_force);
  gripper1_y_pos_pid_ = common::PID(kP_, kI_, kD_, max_integral, -max_integral, max_force, -max_force);
  gripper2_x_pos_pid_ = common::PID(kP_, kI_, kD_, max_integral, -max_integral, max_force, -max_force);
  gripper2_y_pos_pid_ = common::PID(kP_, kI_, kD_, max_integral, -max_integral, max_force, -max_force);

  std::cout << "MultiLinkBot Model Plugin finished initializing!\n";
}

void MultiLinkBotModelPlugin::OnUpdate()
{
  constexpr auto dt{0.001};

  ignition::math::Vector3d force{};
  auto const gripper1_position = gripper1_link_->WorldPose().Pos();
  auto const gripper1_error = gripper1_position - gripper1_target_position_;
  force.X(gripper1_x_pos_pid_.Update(gripper1_error.X(), dt));
  force.Y(gripper1_y_pos_pid_.Update(gripper1_error.Y(), dt));
  gripper1_link_->AddForce(force);

  if (gripper2_link_) {
    auto const gripper2_position = gripper2_link_->WorldPose().Pos();
    auto const gripper2_error = gripper2_position - gripper2_target_position_;
    force.X(gripper2_x_pos_pid_.Update(gripper2_error.X(), dt));
    force.Y(gripper2_y_pos_pid_.Update(gripper2_error.Y(), dt));
    gripper2_link_->AddForce(force);
  }
}

void MultiLinkBotModelPlugin::OnJoy(sensor_msgs::JoyConstPtr const msg)
{
  gripper1_target_position_.X(-msg->axes[0]);
  gripper1_target_position_.Y(msg->axes[1]);
  gripper2_target_position_.X(msg->axes[3]);
  gripper2_target_position_.Y(-msg->axes[4]);
}

void MultiLinkBotModelPlugin::OnAction(link_bot_gazebo::MultiLinkBotPositionActionConstPtr const msg)
{
  gripper1_target_position_.X(msg->gripper1_pos.x);
  gripper1_target_position_.Y(msg->gripper1_pos.y);
  gripper2_target_position_.X(msg->gripper2_pos.x);
  gripper2_target_position_.Y(msg->gripper2_pos.y);
}

void MultiLinkBotModelPlugin::OnConfiguration(link_bot_gazebo::LinkBotConfigurationConstPtr msg)
{
  auto const &joints = model_->GetJoints();

  if (joints.size() != msg->joint_angles_rad.size()) {
    ROS_ERROR("Model as %lu joints config message had %lu", joints.size(), msg->joint_angles_rad.size());
    return;
  }

  ignition::math::Pose3d pose{};
  pose.Pos().X(msg->tail_pose.x);
  pose.Pos().Y(msg->tail_pose.y);
  pose.Pos().Z(0.05);
  pose.Rot() = ignition::math::Quaterniond::EulerToQuaternion(0, 0, msg->tail_pose.theta);
  model_->SetWorldPose(pose);
  model_->SetWorldTwist({0, 0, 0}, {0, 0, 0});

  for (size_t i = 0; i < joints.size(); ++i) {
    auto const &joint = joints[i];
    joint->SetPosition(0, msg->joint_angles_rad[i]);
    joint->SetVelocity(0, 0);
  }
}

bool MultiLinkBotModelPlugin::StateServiceCallback(link_bot_gazebo::LinkBotStateRequest &req,
                                                   link_bot_gazebo::LinkBotStateResponse &res)
{
  auto const &tail = model_->GetLink("link_0");
  auto const &mid = model_->GetLink("link_4");
  auto const &head = model_->GetLink("head");

  res.tail_x = tail->WorldPose().Pos().X();
  res.tail_y = tail->WorldPose().Pos().Y();
  res.mid_x = mid->WorldPose().Pos().X();
  res.mid_y = mid->WorldPose().Pos().Y();
  res.head_x = head->WorldPose().Pos().X();
  res.head_y = head->WorldPose().Pos().Y();

  res.overstretched = 0;

  res.gripper1_force.x = gripper1_x_pos_pid_.GetCmd();
  res.gripper1_force.y = gripper1_y_pos_pid_.GetCmd();
  res.gripper1_force.z = 0;

  res.gripper2_force.x = gripper2_x_pos_pid_.GetCmd();
  res.gripper2_force.y = gripper2_y_pos_pid_.GetCmd();
  res.gripper2_force.z = 0;

  auto const gripper1_velocity = gripper1_link_->WorldLinearVel();
  res.gripper1_velocity.x = gripper1_velocity.X();
  res.gripper1_velocity.y = gripper1_velocity.Y();
  res.gripper1_velocity.z = 0;

  if (gripper2_link_) {
    auto const gripper2_velocity = gripper2_link_->WorldLinearVel();
    res.gripper2_velocity.x = gripper2_velocity.X();
    res.gripper2_velocity.y = gripper2_velocity.Y();
    res.gripper2_velocity.z = 0;
  }

  return true;
}

void MultiLinkBotModelPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_->ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

MultiLinkBotModelPlugin::~MultiLinkBotModelPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_->shutdown();
  ros_queue_thread_.join();
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(MultiLinkBotModelPlugin)
}  // namespace gazebo
