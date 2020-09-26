#include "kinematic_victor_plugin.h"

#include <peter_msgs/GetBool.h>
#include <peter_msgs/GetJointState.h>
#include <peter_msgs/SetDualGripperPoints.h>
#include <ros/subscribe_options.h>
#include <std_msgs/Empty.h>
#include <std_srvs/SetBool.h>
#include <boost/algorithm/string.hpp>

#include <algorithm>
#include <functional>

#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include "enumerate.h"

#define create_service_options(type, name, bind)                                                                       \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

#define create_service_options_private(type, name, bind)                                                               \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &private_queue_)

static ignition::math::Pose3d ToIgnition(geometry_msgs::Transform const &transform)
{
  return ignition::math::Pose3d(transform.translation.x, transform.translation.y, transform.translation.z,
                                transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z);
}

namespace gazebo
{
GZ_REGISTER_MODEL_PLUGIN(KinematicVictorPlugin)

auto const PLUGIN_NAME{ "KinematicVictorPlugin" };

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
    auto joint_element = sdf->GetParent()->GetElement("joint");
    while (joint_element)
    {
      ROS_INFO_STREAM("Setting initial position for joint " << joint_element->GetAttribute("name")->GetAsString());
      if (joint_element->HasElement("axis"))
      {
        auto axis = joint_element->GetElement("axis");
        if (axis->HasElement("initial_position"))
        {
          auto initial_position = axis->GetElement("initial_position")->Get<double>();
          auto joint = GetJoint(PLUGIN_NAME, model_, joint_element->Get<std::string>("name"));
          joint->SetPosition(0, initial_position);
        }
      }
      if (joint_element->HasElement("axis2"))
      {
        auto axis = joint_element->GetElement("axis2");
        if (axis->HasElement("initial_position"))
        {
          auto initial_position = axis->GetElement("initial_position")->Get<double>();
          auto joint = GetJoint(PLUGIN_NAME, model_, joint_element->Get<std::string>("name"));
          joint->SetPosition(1, initial_position);
        }
      }
      joint_element = joint_element->GetNextElement("joint");
    }
  }
  // Handle any Victor specific overrides set in the plugin
  {
    if (sdf->HasElement("joint_names"))
    {
      auto joint_names_str = sdf->GetElement("joint_names")->Get<std::string>();
      boost::split(joint_names_, joint_names_str, boost::is_any_of(" ,\n"));
    }
    else
    {
      ROS_ERROR_STREAM_NAMED(PLUGIN_NAME, "Victor plugin SDF missing joint names");
    }
  }

  // Mimic fixed joints between Victor's grippers and the grippers attached to the rope (if it exists)
  {
    left_flange_ = model_->GetLink(left_flange_name_);
    right_flange_ = model_->GetLink(right_flange_name_);
    if (!left_flange_)
    {
      ROS_ERROR_STREAM("Invalid link name for Victor left flange: " << left_flange_name_);
    }
    if (!right_flange_)
    {
      ROS_ERROR_STREAM("Invalid link name for Victor left flange: " << right_flange_name_);
    }
  }

  // Setup ROS publishers, subscribers, and services, action servers
  {
    private_ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());

    auto grasping_rope_bind = [this](std_srvs::SetBoolRequest &req, std_srvs::SetBoolResponse &res) {
      (void)res;
      grasping_rope_ = req.data;
      ROS_INFO_STREAM("grasping state set to " << static_cast<bool>(req.data));
      if (grasping_rope_)
      {
        TeleportGrippers();
        ignore_overstretching_ = false;
      }
      else
      {
        ignore_overstretching_ = true;
      }
      return true;
    };
    auto grasping_rope_so = create_service_options_private(std_srvs::SetBool, "set_grasping_rope", grasping_rope_bind);
    grasping_rope_server_ = ros_node_.advertiseService(grasping_rope_so);

    auto ignore_overstretching_bind = [this](std_srvs::SetBoolRequest &req, std_srvs::SetBoolResponse &res) {
      (void)res;
      ignore_overstretching_ = req.data;
      return true;
    };
    auto ignore_overstretching_so =
        create_service_options_private(std_srvs::SetBool, "set_ignore_overstretching", ignore_overstretching_bind);
    ignore_overstretching_server_ = ros_node_.advertiseService(ignore_overstretching_so);

    auto joint_state_bind = [this](peter_msgs::GetJointStateRequest &req, peter_msgs::GetJointStateResponse &res) {
      (void)req;
      res.joint_state = GetJointStates();
      return true;
    };
    auto joint_state_so = create_service_options_private(peter_msgs::GetJointState, "joint_states", joint_state_bind);
    joint_state_server_ = ros_node_.advertiseService(joint_state_so);
    joint_states_pub_ = ros_node_.advertise<sensor_msgs::JointState>("joint_states", 1);

    rope_overstretched_srv_ = ros_node_.serviceClient<peter_msgs::GetBool>("rope_overstretched");
    set_dual_gripper_points_srv_ = ros_node_.serviceClient<peter_msgs::SetDualGripperPoints>("set_dual_gripper_points");

    left_arm_motion_status_pub_ =
        ros_node_.advertise<victor_hardware_interface_msgs::MotionStatus>("left_arm/motion_status", 1);
    right_arm_motion_status_pub_ =
        ros_node_.advertise<victor_hardware_interface_msgs::MotionStatus>("right_arm/motion_status", 1);
    left_gripper_status_pub_ =
        ros_node_.advertise<victor_hardware_interface_msgs::Robotiq3FingerStatus>("left_arm/gripper_status", 1);
    right_gripper_status_pub_ =
        ros_node_.advertise<victor_hardware_interface_msgs::Robotiq3FingerStatus>("right_arm/gripper_status", 1);
    auto left_arm_motion_command_sub_options = ros::SubscribeOptions::create<victor_hardware_interface_msgs::MotionCommand>(
        "left_arm/motion_command", 1, boost::bind(&KinematicVictorPlugin::OnLeftArmMotionCommand, this, _1),
        ros::VoidPtr(), &queue_);
    left_arm_motion_command_sub_ = ros_node_.subscribe(left_arm_motion_command_sub_options);
    auto right_arm_motion_command_sub_options = ros::SubscribeOptions::create<victor_hardware_interface_msgs::MotionCommand>(
        "right_arm/motion_command", 1, boost::bind(&KinematicVictorPlugin::OnRightArmMotionCommand, this, _1),
        ros::VoidPtr(), &queue_);
    right_arm_motion_command_sub_ = ros_node_.subscribe(right_arm_motion_command_sub_options);
    auto execute = [this](const TrajServer::GoalConstPtr &goal) { this->FollowJointTrajectory(goal); };
    follow_traj_server_ =
        std::make_unique<TrajServer>(ros_node_, "both_arms_controller/follow_joint_trajectory", execute, false);
    follow_traj_server_->start();

    ros_queue_thread_ = std::thread([this] { QueueThread(); });
    private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

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
  sensor_msgs::JointState msg;
  for (auto const joint_name : joint_names_)
  {
    auto j = GetJoint(PLUGIN_NAME, model_, joint_name);
    msg.name.push_back(joint_name);
    msg.position.push_back(j->Position(0));
    msg.velocity.push_back(j->GetVelocity(0));
    msg.effort.push_back(j->GetForce(0));
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
  victor_hardware_interface_msgs::Robotiq3FingerStatus status;
  left_gripper_status_pub_.publish(status);
}

void KinematicVictorPlugin::PublishRightGripperStatus()
{
  victor_hardware_interface_msgs::Robotiq3FingerStatus status;
  right_gripper_status_pub_.publish(status);
}

void KinematicVictorPlugin::PublishLeftArmMotionStatus()
{
  victor_hardware_interface_msgs::MotionStatus left_arm_motion_status;
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
    joint_name_ss << "victor_left_arm_joint_" << joint_idx;
    auto joint = GetJoint(PLUGIN_NAME, model_, joint_name_ss.str());
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
  victor_hardware_interface_msgs::MotionStatus right_arm_motion_status;
  std::vector<double *> joint_angles{
    &right_arm_motion_status.measured_joint_position.joint_1, &right_arm_motion_status.measured_joint_position.joint_2,
    &right_arm_motion_status.measured_joint_position.joint_3, &right_arm_motion_status.measured_joint_position.joint_4,
    &right_arm_motion_status.measured_joint_position.joint_5, &right_arm_motion_status.measured_joint_position.joint_6,
    &right_arm_motion_status.measured_joint_position.joint_7,
  };
  for (auto joint_idx{ 1u }; joint_idx <= 7u; ++joint_idx)
  {
    std::stringstream joint_name_ss;
    joint_name_ss << "victor_right_arm_joint_" << joint_idx;
    auto joint = GetJoint(PLUGIN_NAME, model_, joint_name_ss.str());
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

void KinematicVictorPlugin::OnLeftArmMotionCommand(const victor_hardware_interface_msgs::MotionCommandConstPtr &msg)
{
  std::lock_guard lock(ros_mutex_);

  std::vector<double> joint_angles{
    msg->joint_position.joint_1, msg->joint_position.joint_2, msg->joint_position.joint_3, msg->joint_position.joint_4,
    msg->joint_position.joint_5, msg->joint_position.joint_6, msg->joint_position.joint_7,
  };
  for (auto joint_idx{ 1u }; joint_idx <= 7u; ++joint_idx)
  {
    std::stringstream joint_name_ss;
    joint_name_ss << "victor_left_arm_joint_" << joint_idx;
    auto joint = GetJoint(PLUGIN_NAME, model_, joint_name_ss.str());
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

void KinematicVictorPlugin::OnRightArmMotionCommand(const victor_hardware_interface_msgs::MotionCommandConstPtr &msg)
{
  std::lock_guard lock(ros_mutex_);

  std::vector<double> joint_angles{
    msg->joint_position.joint_1, msg->joint_position.joint_2, msg->joint_position.joint_3, msg->joint_position.joint_4,
    msg->joint_position.joint_5, msg->joint_position.joint_6, msg->joint_position.joint_7,
  };
  for (auto joint_idx{ 1u }; joint_idx <= 7u; ++joint_idx)
  {
    std::stringstream joint_name_ss;
    joint_name_ss << "victor_right_arm_joint_" << joint_idx;
    auto joint = GetJoint(PLUGIN_NAME, model_, joint_name_ss.str());
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
  auto const requested_settling_time = goal->goal_time_tolerance.toSec();
  auto const default_settling_time = 0.5;
  auto const settling_time_seconds = requested_settling_time == 0 ? default_settling_time : requested_settling_time;
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
        auto joint = GetJoint(PLUGIN_NAME, model_, joint_name);
        if (joint)
        {
          joint->SetPosition(0, point.positions[joint_idx]);
        }
      }

      // Make the grippers match the tool positions, then step the world to allow the rope to "catch up"
      TeleportGrippers();
      PublishJointStates();
      world_->Step(steps);
    }

    // Check if the rope has become overstretched
    peter_msgs::GetBool rope_overstretched;
    rope_overstretched_srv_.call(rope_overstretched);

    if (not ignore_overstretching_ and rope_overstretched.response.data)
    {
      ROS_WARN_STREAM("Requested action overstretched the rope at point_idx " << point_idx << ", rewinding.");

      for (auto rewind_idx = point_idx; rewind_idx > 0; --rewind_idx)
      {
        auto const &rewind_point = goal->trajectory.points[rewind_idx - 1];
        std::lock_guard lock(ros_mutex_);
        // Move Victor to the specified joint configuration
        for (auto const &[joint_idx, joint_name] : enumerate(goal->trajectory.joint_names))
        {
          auto joint = GetJoint(PLUGIN_NAME, model_, joint_name);
          joint->SetPosition(0, rewind_point.positions[joint_idx]);
        }

        // Make the grippers match the tool positions, then step the world to allow the rope to "catch up"
        TeleportGrippers();
        world_->Step(steps);

        peter_msgs::GetBool rope_still_overstretched;
        rope_overstretched_srv_.call(rope_still_overstretched);

        if (not ignore_overstretching_ and rope_still_overstretched.response.data)
        {
          ROS_WARN_STREAM("Rewind stopped at rewind_idx " << rewind_idx - 1);
          break;
        }
      }

      peter_msgs::GetBool rope_final_still_overstretched;
      rope_overstretched_srv_.call(rope_final_still_overstretched);
      ROS_ERROR_COND(not ignore_overstretching_ and rope_final_still_overstretched.response.data, "Rewind unable to "
                                                                                                  "find unstretched "
                                                                                                  "state. Rewound "
                                                                                                  "to start of "
                                                                                                  "trajectory.");
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
  if (left_flange_ && right_flange_)
  {
    try
    {
      auto const left_tool_offset =
          ToIgnition(tf_buffer_.lookupTransform(left_flange_tf_name_, gripper1_tf_name_, ros::Time(0)).transform);
      auto gripper1_pose = left_tool_offset + left_flange_->WorldPose();

      auto const right_tool_offset =
          ToIgnition(tf_buffer_.lookupTransform(right_flange_tf_name_, gripper2_tf_name_, ros::Time(0)).transform);
      auto gripper2_pose = right_tool_offset + right_flange_->WorldPose();

      // call service to teleport grippers
      peter_msgs::SetDualGripperPoints set_dual_gripper_points;
      set_dual_gripper_points.request.gripper1.x = gripper1_pose.Pos().X();
      set_dual_gripper_points.request.gripper1.y = gripper1_pose.Pos().Y();
      set_dual_gripper_points.request.gripper1.z = gripper1_pose.Pos().Z();
      set_dual_gripper_points.request.gripper2.x = gripper2_pose.Pos().X();
      set_dual_gripper_points.request.gripper2.y = gripper2_pose.Pos().Y();
      set_dual_gripper_points.request.gripper2.z = gripper2_pose.Pos().Z();
      set_dual_gripper_points_srv_.call(set_dual_gripper_points);
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN_STREAM("Failed to lookup transform when teleporting grippers" << ex.what());
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
