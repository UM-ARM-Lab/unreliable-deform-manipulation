#include "kinematic_victor_plugin.h"

#include <sensor_msgs/JointState.h>
#include <std_msgs/Empty.h>

#include <functional>

#include "enumerate.h"

#define create_service_options(type, name, bind)                                                                       \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

#define create_service_options_private(type, name, bind)                                                               \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &private_queue_)

namespace gazebo
{
GZ_REGISTER_MODEL_PLUGIN(KinematicVictorPlugin)

KinematicVictorPlugin::~KinematicVictorPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  private_ros_node_->shutdown();
  ros_queue_thread_.join();
  private_ros_queue_thread_.join();
}

void KinematicVictorPlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf)
{
  model_ = parent;
  world_ = parent->GetWorld();

  // setup ROS stuff
  if (!ros::isInitialized())
  {
    int argc = 0;
    ros::init(argc, nullptr, model_->GetScopedName(), ros::init_options::NoSigintHandler);
  }

  private_ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
  joint_states_pub_ = ros_node_.advertise<sensor_msgs::JointState>("joint_states", 10);
  auto execute = [this](const TrajServer::GoalConstPtr &goal) { this->FollowJointTrajectory(goal); };
  follow_traj_server_ = std::make_unique<TrajServer>(*private_ros_node_, "follow_joint_trajectory", execute, false);
  follow_traj_server_->start();

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
  private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

  auto update = [this](common::UpdateInfo const &info) { OnUpdate(); };
  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(update);
}

void KinematicVictorPlugin::OnUpdate()
{
  sensor_msgs::JointState msg;
  for (auto const &j : model_->GetJoints())
  {
    // FIXME: why is this not equal to physics::Joint::HINGE_JOINT??
    if (j->GetType() == 576)
    {  // revolute
      msg.name.push_back(j->GetName());
      msg.position.push_back(j->Position(0));
      msg.velocity.push_back(j->GetVelocity(0));
      msg.effort.push_back(j->GetForce(0));
    }
  }
  joint_states_pub_.publish(msg);
}

void KinematicVictorPlugin::FollowJointTrajectory(const TrajServer::GoalConstPtr &goal)
{
  auto result = control_msgs::FollowJointTrajectoryResult();
  result.error_code = control_msgs::FollowJointTrajectoryResult::SUCCESSFUL;
  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const settling_time_seconds = goal->goal_time_tolerance.toSec();
  auto const steps = static_cast<unsigned int>(settling_time_seconds / seconds_per_step);
  for (auto const &point : goal->trajectory.points)
  {
    for (auto const &pair : enumerate(goal->trajectory.joint_names))
    {
      auto const &[joint_idx, joint_name] = pair;
      // Step the world
      world_->Step(steps);
      auto joint = model_->GetJoint(joint_name);
      if (joint)
      {
        joint->SetPosition(0, point.positions[joint_idx]);
      }
      else
      {
        result.error_code = control_msgs::FollowJointTrajectoryResult::INVALID_JOINTS;
        follow_traj_server_->setSucceeded(result);
      }
    }
  }
  follow_traj_server_->setSucceeded(result);
}

void KinematicVictorPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void KinematicVictorPlugin::PrivateQueueThread()
{
  double constexpr timeout = 0.01;
  while (private_ros_node_->ok())
  {
    private_queue_.callAvailable(ros::WallDuration(timeout));
  }
}
}  // namespace gazebo
