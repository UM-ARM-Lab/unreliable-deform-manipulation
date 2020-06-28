#include "kinematic_victor_plugin.h"

#include <ros/subscribe_options.h>
#include <std_msgs/Empty.h>

#include <algorithm>
#include <functional>

#include "enumerate.h"

#define create_service_options(type, name, bind)                                                                       \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

#define create_service_options_private(type, name, bind)                                                               \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &private_queue_)

#define CANARY                                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    std::cerr << "Reached " << __FILE__ << ":" << __LINE__ << std::endl;                                               \
  } while (false);

static std::istream &operator>>(std::istream &_in, std::vector<double> &_vec)
{
  // Remove any commas from the data
  std::string content{ std::istreambuf_iterator<char>(_in), std::istreambuf_iterator<char>() };
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

namespace gazebo
{
GZ_REGISTER_MODEL_PLUGIN(KinematicVictorPlugin)

KinematicVictorPlugin::KinematicVictorPlugin() : tf_listener_(tf_buffer_)
{
}

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
  if (!ros::isInitialized())
  {
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
          for (auto const &[idx, val] : enumerate(vals))
          {
            auto const joint_name = arm_prefix + std::to_string(idx + 1);
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
  }

  // Setup ROS stuff
  {
    private_ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
    joint_states_pub_ = ros_node_.advertise<sensor_msgs::JointState>("joint_states", 10);
    left_arm_motion_status_pub_ =
        ros_node_.advertise<victor_hardware_interface::MotionStatus>("left_arm/motion_status", 10);
    right_arm_motion_status_pub_ =
        ros_node_.advertise<victor_hardware_interface::MotionStatus>("right_arm/motion_status", 10);
    left_gripper_status_pub_ =
        ros_node_.advertise<victor_hardware_interface::Robotiq3FingerStatus>("left_arm/gripper_status", 10);
    right_gripper_status_pub_ =
        ros_node_.advertise<victor_hardware_interface::Robotiq3FingerStatus>("right_arm/gripper_status", 10);
    auto left_arm_motion_command_sub_options = ros::SubscribeOptions::create<victor_hardware_interface::MotionCommand>(
        "left_arm/motion_command", 10, boost::bind(&KinematicVictorPlugin::OnLeftArmMotionCommand, this, _1),
        ros::VoidPtr(), &queue_);
    left_arm_motion_command_sub_ = ros_node_.subscribe(left_arm_motion_command_sub_options);
    auto right_arm_motion_command_sub_options = ros::SubscribeOptions::create<victor_hardware_interface::MotionCommand>(
        "right_arm/motion_command", 10, boost::bind(&KinematicVictorPlugin::OnRightArmMotionCommand, this, _1),
        ros::VoidPtr(), &queue_);
    right_arm_motion_command_sub_ = ros_node_.subscribe(right_arm_motion_command_sub_options);
    auto execute = [this](const TrajServer::GoalConstPtr &goal) { this->FollowJointTrajectory(goal); };
    follow_traj_server_ = std::make_unique<TrajServer>(ros_node_, "follow_joint_trajectory", execute, false);
    follow_traj_server_->start();

    ros_queue_thread_ = std::thread([this] { QueueThread(); });
    private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });
  }

  periodic_event_thread_ = std::thread([this] {
    while (true)
    {
      // Make the grippers match the initial tool positions
      usleep(50000);
      PeriodicUpdate();
      TeleportGrippers();
    }
  });
}

void KinematicVictorPlugin::PeriodicUpdate()
{
  // Remove the leading "victor::" from the joint names
  auto const n_removed = std::strlen("victor::");
  sensor_msgs::JointState msg;
  for (auto const &j : model_->GetJoints())
  {
    // FIXME: why is this not equal to physics::Joint::HINGE_JOINT??
    if (j->GetType() == 576)
    {  // revolute
      msg.name.push_back(j->GetName().substr(n_removed));
      msg.position.push_back(j->Position(0));
      msg.velocity.push_back(j->GetVelocity(0));
      msg.effort.push_back(j->GetForce(0));
    }
  }
  msg.header.stamp = ros::Time::now();
  joint_states_pub_.publish(msg);

  PublishLeftArmMotionStatus();
  PublishRightArmMotionStatus();

  PublishLeftGripperStatus();
  PublishRightGripperStatus();
}

void KinematicVictorPlugin::PublishLeftGripperStatus()
{
  victor_hardware_interface::Robotiq3FingerStatus status;
  left_gripper_status_pub_.publish(status);
}

void KinematicVictorPlugin::PublishRightGripperStatus()
{
  victor_hardware_interface::Robotiq3FingerStatus status;
  right_gripper_status_pub_.publish(status);
}

void KinematicVictorPlugin::PublishLeftArmMotionStatus()
{
  victor_hardware_interface::MotionStatus left_arm_motion_status;
  std::vector<double *> joint_angles{
    &left_arm_motion_status.measured_joint_position.joint_1, &left_arm_motion_status.measured_joint_position.joint_2,
    &left_arm_motion_status.measured_joint_position.joint_3, &left_arm_motion_status.measured_joint_position.joint_4,
    &left_arm_motion_status.measured_joint_position.joint_5, &left_arm_motion_status.measured_joint_position.joint_6,
    &left_arm_motion_status.measured_joint_position.joint_7,
  };
  left_arm_motion_status_pub_.publish(left_arm_motion_status);
  for (auto joint_idx{ 1u }; joint_idx <= 7u; ++joint_idx)
  {
    std::stringstream joint_name_ss;
    joint_name_ss << "victor::victor_left_arm_joint_" << joint_idx;
    auto joint = model_->GetJoint(joint_name_ss.str());
    if (joint)
    {
      *joint_angles[joint_idx - 1] = joint->Position(0);
    }
    else
    {
      ROS_ERROR_STREAM("Failed to get joint with name " << joint_name_ss.str());
      return;
    }
  }
  left_arm_motion_status.header.stamp = ros::Time::now();
  left_arm_motion_status_pub_.publish(left_arm_motion_status);
}

void KinematicVictorPlugin::PublishRightArmMotionStatus()
{
  victor_hardware_interface::MotionStatus right_arm_motion_status;
  std::vector<double *> joint_angles{
    &right_arm_motion_status.measured_joint_position.joint_1, &right_arm_motion_status.measured_joint_position.joint_2,
    &right_arm_motion_status.measured_joint_position.joint_3, &right_arm_motion_status.measured_joint_position.joint_4,
    &right_arm_motion_status.measured_joint_position.joint_5, &right_arm_motion_status.measured_joint_position.joint_6,
    &right_arm_motion_status.measured_joint_position.joint_7,
  };
  for (auto joint_idx{ 1u }; joint_idx <= 7u; ++joint_idx)
  {
    std::stringstream joint_name_ss;
    joint_name_ss << "victor::victor_right_arm_joint_" << joint_idx;
    auto joint = model_->GetJoint(joint_name_ss.str());
    if (joint)
    {
      *joint_angles[joint_idx - 1] = joint->Position(0);
    }
    else
    {
      ROS_ERROR_STREAM("Failed to get joint with name " << joint_name_ss.str());
      return;
    }
  }
  right_arm_motion_status.header.stamp = ros::Time::now();
  right_arm_motion_status_pub_.publish(right_arm_motion_status);
}

void KinematicVictorPlugin::OnLeftArmMotionCommand(const victor_hardware_interface::MotionCommandConstPtr &msg)
{
  std::vector<double> joint_angles{
    msg->joint_position.joint_1, msg->joint_position.joint_2, msg->joint_position.joint_3, msg->joint_position.joint_4,
    msg->joint_position.joint_5, msg->joint_position.joint_6, msg->joint_position.joint_7,
  };
  for (auto joint_idx{ 1u }; joint_idx <= 7u; ++joint_idx)
  {
    std::stringstream joint_name_ss;
    joint_name_ss << "victor::victor_left_arm_joint_" << joint_idx;
    auto joint = model_->GetJoint(joint_name_ss.str());
    if (joint)
    {
      joint->SetPosition(0, joint_angles[joint_idx - 1]);
    }
    else
    {
      ROS_ERROR_STREAM("Failed to get joint with name " << joint_name_ss.str());
      return;
    }
  }
}

void KinematicVictorPlugin::OnRightArmMotionCommand(const victor_hardware_interface::MotionCommandConstPtr &msg)
{
  std::vector<double> joint_angles{
    msg->joint_position.joint_1, msg->joint_position.joint_2, msg->joint_position.joint_3, msg->joint_position.joint_4,
    msg->joint_position.joint_5, msg->joint_position.joint_6, msg->joint_position.joint_7,
  };
  for (auto joint_idx{ 1u }; joint_idx <= 7u; ++joint_idx)
  {
    std::stringstream joint_name_ss;
    joint_name_ss << "victor::victor_right_arm_joint_" << joint_idx;
    auto joint = model_->GetJoint(joint_name_ss.str());
    if (joint)
    {
      joint->SetPosition(0, joint_angles[joint_idx - 1]);
    }
    else
    {
      ROS_ERROR_STREAM("Failed to get joint with name " << joint_name_ss.str());
      return;
    }
  }
}

void KinematicVictorPlugin::FollowJointTrajectory(const TrajServer::GoalConstPtr &goal)
{
  world_->SetPaused(true);

  auto result = control_msgs::FollowJointTrajectoryResult();
  result.error_code = control_msgs::FollowJointTrajectoryResult::SUCCESSFUL;
  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const settling_time_seconds = goal->goal_time_tolerance.toSec();
  auto const steps = static_cast<unsigned int>(settling_time_seconds / seconds_per_step);
  ROS_INFO_STREAM("Received trajectory with "
                  << "seconds_per_step: " << seconds_per_step << "  settling_time_seconds: " << settling_time_seconds
                  << "  steps per point: " << steps << "  points: " << goal->trajectory.points.size());
  for (auto const &point : goal->trajectory.points)
  {
    // Move Victor to the specified joint configuration
    for (auto const &pair : enumerate(goal->trajectory.joint_names))
    {
      auto const &[joint_idx, joint_name] = pair;
      auto joint = model_->GetJoint("victor::" + joint_name);
      if (joint)
      {
        joint->SetPosition(0, point.positions[joint_idx]);
      }
      else
      {
        gzerr << "Invalid joint: "
              << "victor::" + joint_name << std::endl;
        result.error_code = control_msgs::FollowJointTrajectoryResult::INVALID_JOINTS;
        follow_traj_server_->setSucceeded(result);
        world_->SetPaused(false);
        return;
      }

      // Make the grippers match the tool positions
      TeleportGrippers();
    }

    // Step the world
    world_->Step(steps);
  }

  follow_traj_server_->setSucceeded(result);

  // TODO: this should possibly be removed
  world_->SetPaused(false);
}

void KinematicVictorPlugin::TeleportGrippers()
{
  if (left_flange_ && right_flange_ && gripper1_ && gripper2_)
  {
    // Gripper 1, left tool
    {
      auto gripper1_pose = gripper1_->WorldPose();

      geometry_msgs::TransformStamped left_tool_transform;
      try
      {
        left_tool_transform = tf_buffer_.lookupTransform("world", "victor_left_tool", ros::Time(0));
        gripper1_pose.Pos().X(left_tool_transform.transform.translation.x);
        gripper1_pose.Pos().Y(left_tool_transform.transform.translation.y);
        gripper1_pose.Pos().Z(left_tool_transform.transform.translation.z);
        gripper1_->SetWorldPose(gripper1_pose);
      }
      catch (tf2::TransformException &ex)
      {
        ROS_WARN("failed to lookup transform to victor_left_tool: %s", ex.what());
      }
    }

    // Gripper 2, right tool
    {
      auto gripper2_pose = gripper2_->WorldPose();

      geometry_msgs::TransformStamped right_tool_transform;
      try
      {
        right_tool_transform = tf_buffer_.lookupTransform("world", "victor_right_tool", ros::Time(0));
        gripper2_pose.Pos().X(right_tool_transform.transform.translation.x);
        gripper2_pose.Pos().Y(right_tool_transform.transform.translation.y);
        gripper2_pose.Pos().Z(right_tool_transform.transform.translation.z);
        gripper2_->SetWorldPose(gripper2_pose);
      }
      catch (tf2::TransformException &ex)
      {
        ROS_WARN("failed to lookup transform to victor_right_tool: %s", ex.what());
      }
    }
  }
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
