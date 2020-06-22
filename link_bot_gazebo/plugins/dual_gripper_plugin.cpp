#include "dual_gripper_plugin.h"

#include <sensor_msgs/JointState.h>
#include <std_msgs/Empty.h>

#include <boost/range/combine.hpp>
#include <functional>

#include "enumerate.h"

#define create_service_options(type, name, bind) \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

#define create_service_options_private(type, name, bind) \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &private_queue_)

namespace gazebo {
GZ_REGISTER_MODEL_PLUGIN(DualGripperPlugin)

DualGripperPlugin::~DualGripperPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  private_ros_node_->shutdown();
  ros_queue_thread_.join();
  private_ros_queue_thread_.join();
}

void DualGripperPlugin::Load(physics::ModelPtr parent, sdf::ElementPtr /*sdf*/)
{
  model_ = parent;
  world_ = parent->GetWorld();

  gripper1_ = model_->GetLink("gripper1");
  gripper2_ = model_->GetLink("gripper2");
  if (!gripper1_) {
    gzerr << "No link gripper1 found\n";
    gzerr << "Links in the model:\n";
    for (const auto &l : model_->GetLinks()) {
      gzerr << l->GetName() << "\n";
    }
  }
  else if (!gripper2_) {
    gzerr << "No link gripper1 found\n";
    gzerr << "Links in the model:\n";
    for (const auto &l : model_->GetLinks()) {
      gzerr << l->GetName() << "\n";
    }
  }

  // setup ROS stuff
  if (!ros::isInitialized()) {
    int argc = 0;
    ros::init(argc, nullptr, model_->GetScopedName(), ros::init_options::NoSigintHandler);
  }

  auto pos_action_bind = [this](peter_msgs::DualGripperTrajectoryRequest &req,
                                peter_msgs::DualGripperTrajectoryResponse &res) { return OnAction(req, res); };
  auto action_so = create_service_options_private(peter_msgs::DualGripperTrajectory, "execute_dual_gripper_trajectory",
                                                  pos_action_bind);

  auto get_bind = [this](peter_msgs::GetDualGripperPointsRequest &req, peter_msgs::GetDualGripperPointsResponse &res) {
    return OnGet(req, res);
  };
  auto get_so = create_service_options_private(peter_msgs::GetDualGripperPoints, "get_dual_gripper_points", get_bind);

  auto set_bind = [this](peter_msgs::SetDualGripperPointsRequest &req, peter_msgs::SetDualGripperPointsResponse &res) {
    return OnSet(req, res);
  };
  auto set_so = create_service_options_private(peter_msgs::SetDualGripperPoints, "set_dual_gripper_points", set_bind);

  auto get_gripper1_bind = [this](auto &&req, auto &&res) { return GetGripper1Callback(req, res); };
  auto get_gripper1_so = create_service_options(peter_msgs::GetObject, "gripper1", get_gripper1_bind);

  auto get_gripper2_bind = [this](auto &&req, auto &&res) { return GetGripper2Callback(req, res); };
  auto get_gripper2_so = create_service_options(peter_msgs::GetObject, "gripper2", get_gripper2_bind);

  private_ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
  action_service_ = ros_node_.advertiseService(action_so);
  get_service_ = ros_node_.advertiseService(get_so);
  set_service_ = ros_node_.advertiseService(set_so);
  joint_states_pub_ = ros_node_.advertise<sensor_msgs::JointState>("joint_states", 10);
  auto interrupt_callback = [this](std_msgs::EmptyConstPtr const &/*msg*/) { this->interrupted_ = true; };
  interrupt_sub_ = ros_node_.subscribe<std_msgs::Empty>("interrupt_trajectory", 10, interrupt_callback);
  register_object_pub_ = ros_node_.advertise<std_msgs::String>("register_object", 10, true);
  get_gripper1_service_ = ros_node_.advertiseService(get_gripper1_so);
  get_gripper2_service_ = ros_node_.advertiseService(get_gripper2_so);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
  private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

  gzwarn << "[" << model_->GetScopedName() << "] Waiting for object server\n";
  while (register_object_pub_.getNumSubscribers() < 1) {
  }

  {
    std_msgs::String register_object;
    register_object.data = "gripper1";
    register_object_pub_.publish(register_object);
  }
  {
    std_msgs::String register_object;
    register_object.data = "gripper2";
    register_object_pub_.publish(register_object);
  }

  auto update = [this](common::UpdateInfo const &/*info*/) { OnUpdate(); };
  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(update);
}

void DualGripperPlugin::OnUpdate()
{
  sensor_msgs::JointState msg;
  for (auto const &j : model_->GetJoints()) {
    // FIXME: why is this not equal to physics::Joint::HINGE_JOINT??
    if (j->GetType() == 576) {  // revolute
      msg.name.push_back(j->GetName());
      msg.position.push_back(j->Position(0));
      msg.velocity.push_back(j->GetVelocity(0));
      msg.effort.push_back(j->GetForce(0));
    }
  }
  msg.header.stamp = ros::Time::now();
  joint_states_pub_.publish(msg);
}

bool DualGripperPlugin::OnAction(peter_msgs::DualGripperTrajectoryRequest &req,
                                 peter_msgs::DualGripperTrajectoryResponse &res)
{
  interrupted_ = false;
  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const steps = static_cast<unsigned int>(req.settling_time_seconds / seconds_per_step);

  if (gripper1_ and gripper2_) {
    for (auto point_pair : boost::combine(req.gripper1_points, req.gripper2_points)) {
      geometry_msgs::Point point1, point2;
      boost::tie(point1, point2) = point_pair;
      gripper1_->SetWorldPose({point1.x, point1.y, point1.z, 0, 0, 0});
      gripper2_->SetWorldPose({point2.x, point2.y, point2.z, 0, 0, 0});
      for (auto t{0}; t <= steps; ++t) {
        world_->Step(1);
        if (interrupted_) {
          return true;
        }
      }
    }
  }
  return true;
}

bool DualGripperPlugin::OnGet(peter_msgs::GetDualGripperPointsRequest &req,
                              peter_msgs::GetDualGripperPointsResponse &res)
{
  if (gripper1_ and gripper2_) {
    res.gripper1.x = gripper1_->WorldPose().Pos().X();
    res.gripper1.y = gripper1_->WorldPose().Pos().Y();
    res.gripper1.z = gripper1_->WorldPose().Pos().Z();
    res.gripper2.x = gripper2_->WorldPose().Pos().X();
    res.gripper2.y = gripper2_->WorldPose().Pos().Y();
    res.gripper2.z = gripper2_->WorldPose().Pos().Z();
  }
  return true;
}

bool DualGripperPlugin::OnSet(peter_msgs::SetDualGripperPointsRequest &req,
                              peter_msgs::SetDualGripperPointsResponse &res)
{
  if (gripper1_ and gripper2_) {
    ignition::math::Pose3d gripper1_pose;
    gripper1_pose.Set(req.gripper1.x, req.gripper1.y, req.gripper1.z, 0, 0, 0);
    gripper1_->SetWorldPose(gripper1_pose);

    ignition::math::Pose3d gripper2_pose;
    gripper2_pose.Set(req.gripper2.x, req.gripper2.y, req.gripper2.z, 0, 0, 0);
    gripper2_->SetWorldPose(gripper2_pose);
  }
  return true;
}

bool DualGripperPlugin::GetGripper1Callback(peter_msgs::GetObjectRequest &req, peter_msgs::GetObjectResponse &res)
{
  if (gripper1_) {
    res.object.name = "gripper1";
    res.object.state_vector.push_back(gripper1_->WorldPose().Pos().X());
    res.object.state_vector.push_back(gripper1_->WorldPose().Pos().Y());
    res.object.state_vector.push_back(gripper1_->WorldPose().Pos().Z());
    geometry_msgs::Point gripper1_point;
    gripper1_point.x = gripper1_->WorldPose().Pos().X();
    gripper1_point.y = gripper1_->WorldPose().Pos().Y();
    gripper1_point.z = gripper1_->WorldPose().Pos().Z();
    peter_msgs::NamedPoint gripper1_named_point;
    gripper1_named_point.point = gripper1_point;
    gripper1_named_point.name = "gripper1";
    res.object.points.push_back(gripper1_named_point);
  }
  return true;
}

bool DualGripperPlugin::GetGripper2Callback(peter_msgs::GetObjectRequest &req, peter_msgs::GetObjectResponse &res)
{
  if (gripper2_) {
    res.object.name = "gripper2";
    res.object.state_vector.push_back(gripper2_->WorldPose().Pos().X());
    res.object.state_vector.push_back(gripper2_->WorldPose().Pos().Y());
    res.object.state_vector.push_back(gripper2_->WorldPose().Pos().Z());
    geometry_msgs::Point gripper2_point;
    gripper2_point.x = gripper2_->WorldPose().Pos().X();
    gripper2_point.y = gripper2_->WorldPose().Pos().Y();
    gripper2_point.z = gripper2_->WorldPose().Pos().Z();
    peter_msgs::NamedPoint gripper2_named_point;
    gripper2_named_point.point = gripper2_point;
    gripper2_named_point.name = "gripper2";
    res.object.points.push_back(gripper2_named_point);
  }
  return true;
}

void DualGripperPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void DualGripperPlugin::PrivateQueueThread()
{
  double constexpr timeout = 0.01;
  while (private_ros_node_->ok()) {
    private_queue_.callAvailable(ros::WallDuration(timeout));
  }
}

}  // namespace gazebo
