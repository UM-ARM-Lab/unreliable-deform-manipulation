#include "kinematic_victor_plugin.h"

#include <peter_msgs/GetJointState.h>
#include <peter_msgs/SetBool.h>
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

static ignition::math::Pose3d ToIgnition(geometry_msgs::Transform const &transform)
{
  return ignition::math::Pose3d(transform.translation.x, transform.translation.y, transform.translation.z,
                                transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z);
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
  model_ = parent;
  world_ = model_->GetWorld();

  if (!ros::isInitialized())
  {
    int argc = 0;
    ros::init(argc, nullptr, model_->GetScopedName(), ros::init_options::NoSigintHandler);
  }

  // Handle initial_position tags because the SDFormat used does not at this version
  {
    auto victor_and_rope_sdf = sdf->GetParent();
    auto joint = victor_and_rope_sdf->GetElement("joint");
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
      if (joint->HasElement("axis2"))
      {
        auto axis = joint->GetElement("axis2");
        if (axis->HasElement("initial_position"))
        {
          auto initial_position = axis->GetElement("initial_position")->Get<double>();
          model_->GetJoint(joint->Get<std::string>("name"))->SetPosition(1, initial_position);
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

    // Extract rope links connected to the floating grippers for use in FollowJointTrajectory
    if (gripper1_)
    {
      auto const parent_links = gripper1_->GetParentJointsLinks();
      ROS_WARN_COND(parent_links.size() != 1, "Victor plus rope kinematic structure is different than expected."
                                              " Check this code for correctness");
      gripper1_rope_link_ = parent_links[0];
    }
    if (gripper2_)
    {
      auto const parent_links = gripper2_->GetParentJointsLinks();
      ROS_WARN_COND(parent_links.size() != 1, "Victor plus rope kinematic structure is different than expected."
                                              " Check this code for correctness");
      gripper2_rope_link_ = parent_links[0];
    }
    if (gripper1_rope_link_ && gripper2_rope_link_)
    {
      auto const dist_factor = sdf->Get<double>("max_dist_between_gripper_and_link_scale_factor", 1.1);
      ROS_WARN_COND(!dist_factor.second, "max_dist_between_gripper_and_link_scale_factor not set in victor.sdf, "
                                         "defaulting to 1.1");
      double const gripper1_dist = (gripper1_->WorldPose().Pos() - gripper1_rope_link_->WorldPose().Pos()).Length();
      double const gripper2_dist = (gripper2_->WorldPose().Pos() - gripper2_rope_link_->WorldPose().Pos()).Length();
      max_dist_between_gripper_and_link_ = dist_factor.first * std::max(gripper1_dist, gripper2_dist);
      ROS_WARN_STREAM_COND(max_dist_between_gripper_and_link_ < 1e-3, "max_dist_between_gripper_and_link_ is set to "
                                                                          << max_dist_between_gripper_and_link_
                                                                          << ". This appears abnormally low.");
    }
    else
    {
      max_dist_between_gripper_and_link_ = std::numeric_limits<double>::max();
      ROS_WARN_STREAM("Errors getting correct links for overstretching detection. Setting max dist to "
                      << max_dist_between_gripper_and_link_);
    }

    while (!tf_buffer_.canTransform(left_flange_tf_name_, gripper1_tf_name_, ros::Time(0)))
    {
      ROS_INFO_STREAM_THROTTLE(1.0, "Waiting for transform between " << left_flange_tf_name_ << " and "
                                                                     << gripper1_tf_name_);
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(0.01s);
    }
    while (!tf_buffer_.canTransform(right_flange_tf_name_, gripper2_tf_name_, ros::Time(0)))
    {
      ROS_INFO_STREAM_THROTTLE(1.0, "Waiting for transform between " << right_flange_tf_name_ << " and "
                                                                     << gripper2_tf_name_);
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(0.01s);
    }

    TeleportGrippers();
  }

  // Setup ROS publishers, subscribers, and services, action servers
  {
    private_ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());

    auto grasping_rope_bind = [this](peter_msgs::SetBoolRequest &req, peter_msgs::SetBoolResponse &res) {
      (void)res;
      grasping_rope_ = req.data;
      if (grasping_rope_)
      {
        TeleportGrippers();
      }
      return true;
    };
    auto grasping_rope_so =
        create_service_options_private(peter_msgs::SetBool, "set_grasping_rope", grasping_rope_bind);
    grasping_rope_server_ = ros_node_.advertiseService(grasping_rope_so);

    auto joint_state_bind = [this](peter_msgs::GetJointStateRequest &req, peter_msgs::GetJointStateResponse &res) {
      (void)req;
      res.joint_state = GetJointStates();
      return true;
    };
    auto joint_state_so = create_service_options_private(peter_msgs::GetJointState, "joint_states", joint_state_bind);
    joint_state_server_ = ros_node_.advertiseService(joint_state_so);
    joint_states_pub_ = ros_node_.advertise<sensor_msgs::JointState>("joint_states", 1);
    left_arm_motion_status_pub_ =
        ros_node_.advertise<victor_hardware_interface::MotionStatus>("left_arm/motion_status", 1);
    right_arm_motion_status_pub_ =
        ros_node_.advertise<victor_hardware_interface::MotionStatus>("right_arm/motion_status", 1);
    left_gripper_status_pub_ =
        ros_node_.advertise<victor_hardware_interface::Robotiq3FingerStatus>("left_arm/gripper_status", 1);
    right_gripper_status_pub_ =
        ros_node_.advertise<victor_hardware_interface::Robotiq3FingerStatus>("right_arm/gripper_status", 1);
    auto left_arm_motion_command_sub_options = ros::SubscribeOptions::create<victor_hardware_interface::MotionCommand>(
        "left_arm/motion_command", 1, boost::bind(&KinematicVictorPlugin::OnLeftArmMotionCommand, this, _1),
        ros::VoidPtr(), &queue_);
    left_arm_motion_command_sub_ = ros_node_.subscribe(left_arm_motion_command_sub_options);
    auto right_arm_motion_command_sub_options = ros::SubscribeOptions::create<victor_hardware_interface::MotionCommand>(
        "right_arm/motion_command", 1, boost::bind(&KinematicVictorPlugin::OnRightArmMotionCommand, this, _1),
        ros::VoidPtr(), &queue_);
    right_arm_motion_command_sub_ = ros_node_.subscribe(right_arm_motion_command_sub_options);
    auto execute = [this](const TrajServer::GoalConstPtr &goal) { this->FollowJointTrajectory(goal); };
    follow_traj_server_ = std::make_unique<TrajServer>(ros_node_, "follow_joint_trajectory", execute, false);
    follow_traj_server_->start();

    ros_queue_thread_ = std::thread([this] { QueueThread(); });
    private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });
  }

  // Publish the robot state at roughly 100 Hz
  periodic_event_thread_ = std::thread([this] {
    while (ros::ok())
    {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(0.01s);
      std::lock_guard lock(ros_mutex_);
      PeriodicUpdate();
    }
  });

  ROS_INFO("kinematic_victor_plugin loaded.");
}

void KinematicVictorPlugin::PeriodicUpdate()
{
  PublishJointStates();

  PublishLeftArmMotionStatus();
  PublishRightArmMotionStatus();

  PublishLeftGripperStatus();
  PublishRightGripperStatus();
}

sensor_msgs::JointState KinematicVictorPlugin::GetJointStates()
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
  return msg;
}

void KinematicVictorPlugin::PublishJointStates()
{
  auto const msg = GetJointStates();
  joint_states_pub_.publish(msg);
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
  std::lock_guard lock(ros_mutex_);

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
  std::lock_guard lock(ros_mutex_);

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
  auto result = control_msgs::FollowJointTrajectoryResult();
  result.error_code = control_msgs::FollowJointTrajectoryResult::SUCCESSFUL;
  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const settling_time_seconds = goal->goal_time_tolerance.toSec();
  auto const steps = static_cast<unsigned int>(settling_time_seconds / seconds_per_step);
  ROS_INFO_STREAM("Received trajectory with "
                  << "seconds_per_step: " << seconds_per_step << "  settling_time_seconds: " << settling_time_seconds
                  << "  steps per point: " << steps << "  points: " << goal->trajectory.points.size());

  for (auto const &[point_idx, point] : enumerate(goal->trajectory.points))
  {
    // Set the kinematic position of Victor plus the rope grippers, then simulate
    {
      std::lock_guard lock(ros_mutex_);
      // Move Victor to the specified joint configuration
      for (auto const &[joint_idx, joint_name] : enumerate(goal->trajectory.joint_names))
      {
        auto joint = model_->GetJoint("victor::" + joint_name);
        if (joint)
        {
          joint->SetPosition(0, point.positions[joint_idx]);
        }
        else
        {
          ROS_ERROR_STREAM("Invalid joint: "
                           << "victor::" + joint_name);
          result.error_code = control_msgs::FollowJointTrajectoryResult::INVALID_JOINTS;
          follow_traj_server_->setAborted(result);
          return;
        }
      }

      // Make the grippers match the tool positions, then step the world to allow the rope to "catch up"
      TeleportGrippers();
      PublishJointStates();
      world_->Step(steps);
    }

    // Check if the rope has become overstretched
    auto const rewind_needed = [this] {
      auto const gripper1_dist = (gripper1_->WorldPose().Pos() - gripper1_rope_link_->WorldPose().Pos()).Length();
      auto const gripper2_dist = (gripper2_->WorldPose().Pos() - gripper2_rope_link_->WorldPose().Pos()).Length();
      return (gripper1_dist > max_dist_between_gripper_and_link_) ||
             (gripper2_dist > max_dist_between_gripper_and_link_);
    };

    if (rewind_needed())
    {
      ROS_WARN_STREAM("Requested action overstretched the rope at point_idx " << point_idx << ", rewinding.");

      for (auto rewind_idx = point_idx; rewind_idx > 0; --rewind_idx)
      {
        auto const &rewind_point = goal->trajectory.points[rewind_idx - 1];
        std::lock_guard lock(ros_mutex_);
        // Move Victor to the specified joint configuration
        for (auto const &[joint_idx, joint_name] : enumerate(goal->trajectory.joint_names))
        {
          auto joint = model_->GetJoint("victor::" + joint_name);
          joint->SetPosition(0, rewind_point.positions[joint_idx]);
        }

        // Make the grippers match the tool positions, then step the world to allow the rope to "catch up"
        TeleportGrippers();
        world_->Step(steps);

        if (!rewind_needed())
        {
          ROS_WARN_STREAM("Rewind stopped at rewind_idx " << rewind_idx - 1);
          break;
        }
      }

      ROS_ERROR_COND(rewind_needed(), "Rewind unable to find unstretched state. Rewound to start of trajectory.");
      break;
    }
  }

  follow_traj_server_->setSucceeded(result);
}

void KinematicVictorPlugin::TeleportGrippers()
{
  if (not grasping_rope_)
  {
    return;
  }
  if (left_flange_ && right_flange_ && gripper1_ && gripper2_)
  {
    // Gripper 1, left tool
    {
      try
      {
        auto gripper1_rot = gripper1_->WorldPose().Rot();
        auto const left_tool_offset =
            ToIgnition(tf_buffer_.lookupTransform(left_flange_tf_name_, gripper1_tf_name_, ros::Time(0)).transform);
        auto gripper1_pose = left_tool_offset + left_flange_->WorldPose();
        gripper1_pose.Rot() = gripper1_rot;
        gripper1_->SetWorldPose(gripper1_pose);
      }
      catch (tf2::TransformException &ex)
      {
        ROS_WARN_STREAM("Failed to lookup transform between " << left_flange_tf_name_ << " and " << gripper1_tf_name_
                                                              << ex.what());
      }
    }

    // Gripper 2, right tool
    {
      try
      {
        auto gripper2_rot = gripper2_->WorldPose().Rot();
        auto const right_tool_offset =
            ToIgnition(tf_buffer_.lookupTransform(right_flange_tf_name_, gripper2_tf_name_, ros::Time(0)).transform);
        auto gripper2_pose = right_tool_offset + right_flange_->WorldPose();
        gripper2_pose.Rot() = gripper2_rot;
        gripper2_->SetWorldPose(gripper2_pose);
      }
      catch (tf2::TransformException &ex)
      {
        ROS_WARN_STREAM("Failed to lookup transform between " << left_flange_tf_name_ << " and " << gripper1_tf_name_
                                                              << ex.what());
      }
    }
  }
  else
  {
    ROS_ERROR_THROTTLE(1.0, "Attempting to teleport the grippers, but some pointers are null");
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
