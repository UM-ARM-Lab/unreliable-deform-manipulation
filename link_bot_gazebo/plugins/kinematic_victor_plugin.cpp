#include "kinematic_victor_plugin.h"

#include <sensor_msgs/JointState.h>
#include <std_msgs/Empty.h>

#include <algorithm>
#include <functional>

#include "enumerate.h"

#define create_service_options(type, name, bind)                                                                       \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

#define create_service_options_private(type, name, bind)                                                               \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &private_queue_)

#define CANARY do { std::cerr << "Reached " << __FILE__ << ":" << __LINE__ << std::endl; } while(false);

static std::istream& operator>>(std::istream &_in, std::vector<double>&_vec)
{
  // Remove any commas from the data
  std::string content{std::istreambuf_iterator<char>(_in), std::istreambuf_iterator<char>()};
  content.erase(std::remove(content.begin(), content.end(), ','), content.end());

  // FIXME: error handling
  // Now parse the data using a typical list of doubles approch
  std::stringstream ss(content);
  _vec.clear();
  double val;
  while (ss >> val)
  {
    _vec.push_back(val);
  }

  return _in;
}

namespace gazebo {
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
  if (!ros::isInitialized()) {
    int argc = 0;
    ros::init(argc, nullptr, model_->GetScopedName(), ros::init_options::NoSigintHandler);
  }

  model_ = parent;
  world_ = model_->GetWorld();

  // Handle initial_position tags because the SDFormat used does not at this version
  auto victor_sdf = sdf->GetParent();
  {
    auto joint = victor_sdf->GetElement("joint");
    while (joint)
    {
      if (joint->HasElement("axis"))
      {
        auto axis = joint->GetElement("axis");
        if (axis->HasElement("initial_position"))
        {
          auto initial_position = axis->GetElement("initial_position")->Get<double>();
          model_->GetJoint(joint->Get<std::string>("name"))->SetPosition(0, initial_position);
        }
      }
      joint = joint->GetNextElement("joint");
    }
  }
  // Handle any Victor specific overrides set in the plugin
  {
    auto arm = sdf->GetElement("arm");
    while (arm)
    {
      // FIXME: Have the joint name logic in the SDF rather than the CPP?
      if (arm->HasElement("initial_positions"))
      {
        auto vals = arm->GetElement("initial_positions")->Get<std::vector<double>>();
        if (vals.size() != 7)
        {
          gzwarn << "<initial_positions> is not of length 7, ignoring.\n";
        }
        else
        {
          auto const arm_prefix = "victor::victor_" + arm->Get<std::string>("name") + "_arm_joint_";
          for (auto const &[idx, val] : enumerate(vals)) {
            auto const joint_name = arm_prefix + std::to_string(idx+1);
            model_->GetJoint(joint_name)->SetPosition(0, val);
          }
        }
      }
      arm = arm->GetNextElement("arm");
    }
  }

  // Mimic fixed joints between Victor's grippers and the grippers attached to the rope (if it exists)
  {
    left_flange_ = model_->GetLink(left_flange_name_);
    right_flange_ = model_->GetLink(right_flange_name_);
    gripper1_ = model_->GetLink(gripper1_name_);
    gripper2_ = model_->GetLink(gripper2_name_);
    if (!left_flange_)
    {
      ROS_ERROR_STREAM("Invalid link name for Victor left flange: " << left_flange_name_);
    }
    if (!right_flange_)
    {
      ROS_ERROR_STREAM("Invalid link name for Victor left flange: " << right_flange_name_);
    }
    if (!gripper1_)
    {
      ROS_ERROR_STREAM("Invalid link name for rope gripper1: " << gripper1_name_);
    }
    if (!gripper2_)
    {
      ROS_ERROR_STREAM("Invalid link name for rope gripper2: " << gripper2_name_);
    }
    if (left_flange_ && gripper1_)
    {
      left_flange_to_gripper1_ = gripper1_->WorldPose() - left_flange_->WorldPose();
    }
    if (right_flange_ && gripper2_)
    {
      right_flange_to_gripper2_ = gripper2_->WorldPose() - right_flange_->WorldPose();
    }
  }

  // Setup ROS stuff
  {
    private_ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
    joint_states_pub_ = ros_node_.advertise<sensor_msgs::JointState>("joint_states", 10);
    auto execute = [this](const TrajServer::GoalConstPtr &goal) { this->FollowJointTrajectory(goal); };
    follow_traj_server_ = std::make_unique<TrajServer>(ros_node_, "follow_joint_trajectory", execute, false);
    follow_traj_server_->start();

    ros_queue_thread_ = std::thread([this] { QueueThread(); });
    private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

    auto update = [this](common::UpdateInfo const &/*info*/) { OnUpdate(); };
    update_connection_ = event::Events::ConnectWorldUpdateBegin(update);
  }
}

void KinematicVictorPlugin::OnUpdate()
{
  // Remove the leading "victor::" from the joint names
  auto const n_removed = std::strlen("victor::");
  sensor_msgs::JointState msg;
  for (auto const &j : model_->GetJoints())
  {
    // FIXME: why is this not equal to physics::Joint::HINGE_JOINT??
    if (j->GetType() == 576) {  // revolute
      msg.name.push_back(j->GetName().substr(n_removed));
      msg.position.push_back(j->Position(0));
      msg.velocity.push_back(j->GetVelocity(0));
      msg.effort.push_back(j->GetForce(0));
    }
  }
  msg.header.stamp = ros::Time::now();
  joint_states_pub_.publish(msg);
}

void KinematicVictorPlugin::FollowJointTrajectory(const TrajServer::GoalConstPtr &goal)
{
  world_->SetPaused(true);

  auto result = control_msgs::FollowJointTrajectoryResult();
  result.error_code = control_msgs::FollowJointTrajectoryResult::SUCCESSFUL;
  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const settling_time_seconds = goal->goal_time_tolerance.toSec();
  auto const steps = static_cast<unsigned int>(settling_time_seconds / seconds_per_step);
  ROS_INFO_STREAM("Received trajectory with " <<
                  "seconds_per_step: " << seconds_per_step <<
                  "  settling_time_seconds: " << settling_time_seconds <<
                  "  steps per point: " << steps <<
                  "  points: " << goal->trajectory.points.size());
  for (auto const &point : goal->trajectory.points) {
    // Move Victor to the specified joint configuration
    for (auto const &pair : enumerate(goal->trajectory.joint_names)) {
      auto const &[joint_idx, joint_name] = pair;
      auto joint = model_->GetJoint("victor::" + joint_name);
      if (joint) {
        joint->SetPosition(0, point.positions[joint_idx]);
      }
      else {
        gzerr << "Invalid joint: " << "victor::" + joint_name << std::endl;
        result.error_code = control_msgs::FollowJointTrajectoryResult::INVALID_JOINTS;
        follow_traj_server_->setSucceeded(result);
        world_->SetPaused(false);
        return;
      }
    }
    // Move the rope kinematic grippers to match
    if (left_flange_ && right_flange_ && gripper1_ && gripper2_)
    {
      // Note that in ignition math, adding on the right is equivalent to multiplying on the left in Eigen
      auto const gripper1_pose = left_flange_to_gripper1_ + left_flange_->WorldPose();
      auto const gripper2_pose = right_flange_to_gripper2_ + right_flange_->WorldPose();
      // FIXME: only change the position, not the whole pose?
      gripper1_->SetWorldPose(gripper1_pose);
      gripper2_->SetWorldPose(gripper2_pose);
    }
    // Step the world
    world_->Step(steps);
  }
  follow_traj_server_->setSucceeded(result);

  world_->SetPaused(false);
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
