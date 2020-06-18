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

void DualGripperPlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf)
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

  private_ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
  action_service_ = ros_node_.advertiseService(action_so);
  get_service_ = ros_node_.advertiseService(get_so);
  joint_states_pub_ = ros_node_.advertise<sensor_msgs::JointState>("joint_states", 10);
  auto interrupt_callback = [this](std_msgs::EmptyConstPtr const &msg) { this->interrupted_ = true; };
  interrupt_sub_ = ros_node_.subscribe<std_msgs::Empty>("interrupt_trajectory", 10, interrupt_callback);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
  private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

  auto update = [this](common::UpdateInfo const &info) { OnUpdate(); };
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
}  // namespace gazebo

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
