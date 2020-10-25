#include "dual_gripper_plugin.h"

#include <std_msgs/Empty.h>

#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include <boost/range/combine.hpp>
#include <ros/duration.h>
#include <functional>

#define create_service_options(type, name, bind)                                                                       \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

#define create_service_options_private(type, name, bind)                                                               \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &private_queue_)

namespace gazebo
{
GZ_REGISTER_MODEL_PLUGIN(DualGripperPlugin)

constexpr auto PLUGIN_NAME{"DualGripperPlugin"};

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

  left_gripper_ = GetLink(PLUGIN_NAME, model_, "left_gripper");
  right_gripper_ = GetLink(PLUGIN_NAME, model_, "right_gripper");

  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                         << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
    return;
  }

  auto enable_bind = [this](std_srvs::SetBoolRequest &req, std_srvs::SetBoolResponse &res)
  { return OnEnable(req, res); };
  auto enable_so = create_service_options_private(std_srvs::SetBool, "enable", enable_bind);

  auto pos_action_bind = [this](peter_msgs::DualGripperTrajectoryRequest &req,
                                peter_msgs::DualGripperTrajectoryResponse &res)
  { return OnAction(req, res); };
  auto action_so = create_service_options_private(peter_msgs::DualGripperTrajectory, "execute_dual_gripper_trajectory",
                                                  pos_action_bind);

  auto get_bind = [this](peter_msgs::GetDualGripperPointsRequest &req, peter_msgs::GetDualGripperPointsResponse &res)
  {
    return OnGet(req, res);
  };
  auto get_so = create_service_options_private(peter_msgs::GetDualGripperPoints, "get_dual_gripper_points", get_bind);

  auto set_bind = [this](peter_msgs::SetDualGripperPointsRequest &req, peter_msgs::SetDualGripperPointsResponse &res)
  {
    return OnSet(req, res);
  };
  auto set_so = create_service_options_private(peter_msgs::SetDualGripperPoints, "set_dual_gripper_points", set_bind);

  private_ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());

  enable_service_ = private_ros_node_->advertiseService(enable_so);
  action_service_ = private_ros_node_->advertiseService(action_so);

  get_service_ = private_ros_node_->advertiseService(get_so);
  set_service_ = private_ros_node_->advertiseService(set_so);

  ros_queue_thread_ = std::thread([this]
                                  { QueueThread(); });
  private_ros_queue_thread_ = std::thread([this]
                                          { PrivateQueueThread(); });

  auto update = [this](common::UpdateInfo const & /*info*/)
  { OnUpdate(); };
  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(update);
  ROS_INFO("Dual gripper plugin finished initializing!");
}

void DualGripperPlugin::OnUpdate()
{
}

bool DualGripperPlugin::OnAction(peter_msgs::DualGripperTrajectoryRequest &req,
                                 peter_msgs::DualGripperTrajectoryResponse &res)
{
  (void) res;
  interrupted_ = false;
  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const steps = static_cast<int>(req.settling_time_seconds / seconds_per_step);

  if (left_gripper_ and right_gripper_)
  {
    for (auto point_pair : boost::combine(req.left_gripper_points, req.right_gripper_points))
    {
      geometry_msgs::Point point1, point2;
      boost::tie(point1, point2) = point_pair;
      left_gripper_->SetWorldPose({point1.x, point1.y, point1.z, 0, 0, 0});
      right_gripper_->SetWorldPose({point2.x, point2.y, point2.z, 0, 0, 0});
      ros::Duration(req.settling_time_seconds).sleep();
//      for (auto t{0}; t <= steps; ++t)
//      {
//        world_->Step(1);
//        if (interrupted_)
//        {
//          return true;
//        }
//      }
    }
  }
  return true;
}

bool DualGripperPlugin::OnGet(peter_msgs::GetDualGripperPointsRequest &req,
                              peter_msgs::GetDualGripperPointsResponse &res)
{
  (void) req;
  if (left_gripper_ and right_gripper_)
  {
    res.left_gripper.x = left_gripper_->WorldPose().Pos().X();
    res.left_gripper.y = left_gripper_->WorldPose().Pos().Y();
    res.left_gripper.z = left_gripper_->WorldPose().Pos().Z();
    res.right_gripper.x = right_gripper_->WorldPose().Pos().X();
    res.right_gripper.y = right_gripper_->WorldPose().Pos().Y();
    res.right_gripper.z = right_gripper_->WorldPose().Pos().Z();
    return true;
  } else
  {
    res.left_gripper.x = -999;
    res.right_gripper.x = -999;
    return false;
  }
}

bool DualGripperPlugin::OnSet(peter_msgs::SetDualGripperPointsRequest &req,
                              peter_msgs::SetDualGripperPointsResponse &res)
{
  (void) res;
  if (left_gripper_ and right_gripper_)
  {
    ignition::math::Pose3d left_gripper_pose;
    left_gripper_pose.Set(req.left_gripper.x, req.left_gripper.y, req.left_gripper.z, 0, 0, 0);
    left_gripper_->SetWorldPose(left_gripper_pose);

    ignition::math::Pose3d right_gripper_pose;
    right_gripper_pose.Set(req.right_gripper.x, req.right_gripper.y, req.right_gripper.z, 0, 0, 0);
    right_gripper_->SetWorldPose(right_gripper_pose);
  }
  return true;
}

void DualGripperPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void DualGripperPlugin::PrivateQueueThread()
{
  double constexpr timeout = 0.01;
  while (private_ros_node_->ok())
  {
    private_queue_.callAvailable(ros::WallDuration(timeout));
  }
}

bool DualGripperPlugin::OnEnable(std_srvs::SetBoolRequest &req, std_srvs::SetBoolResponse &res)
{
  if (left_gripper_ and right_gripper_)
  {
    left_gripper_->SetKinematic(req.data);
    right_gripper_->SetKinematic(req.data);
    // how can we zero out the velocity of the grippers?
    left_gripper_->SetAngularVel(ignition::math::Vector3d::Zero);
    left_gripper_->SetLinearVel(ignition::math::Vector3d::Zero);
    right_gripper_->SetAngularVel(ignition::math::Vector3d::Zero);
    right_gripper_->SetLinearVel(ignition::math::Vector3d::Zero);
    res.success = true;
  }
  else
  {
    res.success = false;
    res.message = "null pointers to gripper links";
  }
  return true;
}

}  // namespace gazebo
