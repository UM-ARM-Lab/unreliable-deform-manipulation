#include "kinematic_val_plugin.h"

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

static ignition::math::Pose3d ToIgnition(geometry_msgs::Transform const &transform)
{
  return ignition::math::Pose3d(transform.translation.x, transform.translation.y, transform.translation.z,
                                transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z);
}

namespace gazebo
{
GZ_REGISTER_MODEL_PLUGIN(KinematicValPlugin)

KinematicValPlugin::KinematicValPlugin() : tf_listener_(tf_buffer_)
{
}

KinematicValPlugin::~KinematicValPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  private_ros_node_->shutdown();
  ros_queue_thread_.join();
  private_ros_queue_thread_.join();
}

void KinematicValPlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf)
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
    auto val_and_rope_sdf = sdf->GetParent();
    auto joint = val_and_rope_sdf->GetElement("joint");
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
  // Handle any Val specific overrides set in the plugin
  {
    if (sdf->HasElement("initial_positions"))
    {
      auto joint_element = sdf->GetElement("initial_positions")->GetFirstElement();
      while (joint_element)
      {
        auto const joint_name = "hdt_michigan::" + joint_element->GetName();
        auto joint_position = joint_element->Get<double>();
        auto const joint = model_->GetJoint(joint_name);
        if (!joint)
        {
          ROS_ERROR_STREAM("Failed to set initial joint position for joint " << joint_name);
          ROS_ERROR_STREAM("Possible joint names are:");
          for (auto const available_joint : model_->GetJoints())
          {
            ROS_ERROR_STREAM(available_joint->GetName());
          }
        }
        else
        {
          ROS_INFO_STREAM("Setting joint " << joint_name << " to position " << joint_position);
          joint->SetPosition(0, joint_position);
        }
        joint_element = joint_element->GetNextElement();
      }
    }
  }

  // Mimic fixed joints between Val's grippers and the grippers attached to the rope (if it exists)
  {
    left_flange_ = model_->GetLink(left_flange_name_);
    right_flange_ = model_->GetLink(right_flange_name_);
    gripper1_ = model_->GetLink(gripper1_name_);
    gripper2_ = model_->GetLink(gripper2_name_);
    if (!left_flange_)
    {
      ROS_ERROR_STREAM("Invalid link name for Val left flange: " << left_flange_name_);
    }
    if (!right_flange_)
    {
      ROS_ERROR_STREAM("Invalid link name for Val left flange: " << right_flange_name_);
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
      ROS_WARN_COND(parent_links.size() != 1, "Val plus rope kinematic structure is different than expected."
                                              " Check this code for correctness");
      gripper1_rope_link_ = parent_links[0];
    }
    if (gripper2_)
    {
      auto const parent_links = gripper2_->GetParentJointsLinks();
      ROS_WARN_COND(parent_links.size() != 1, "Val plus rope kinematic structure is different than expected."
                                              " Check this code for correctness");
      gripper2_rope_link_ = parent_links[0];
    }
    if (gripper1_rope_link_ && gripper2_rope_link_)
    {
      auto const dist_factor = sdf->Get<double>("max_dist_between_gripper_and_link_scale_factor", 1.1);
      ROS_WARN_COND(!dist_factor.second, "max_dist_between_gripper_and_link_scale_factor not set in val.sdf, "
                                         "defaulting to 1.1");
      double const gripper1_dist = (gripper1_->WorldPose().Pos() - gripper1_rope_link_->WorldPose().Pos()).Length();
      double const gripper2_dist = (gripper2_->WorldPose().Pos() - gripper2_rope_link_->WorldPose().Pos()).Length();
      ROS_WARN_STREAM(gripper1_dist << " " << gripper2_dist << " " << dist_factor.first);
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
      ROS_INFO_STREAM("grasping state set to " << static_cast<bool>(req.data));
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
    auto execute = [this](const TrajServer::GoalConstPtr &goal) { this->FollowJointTrajectory(goal); };
    follow_traj_server_ =
        std::make_unique<TrajServer>(ros_node_, "both_arms_controller/follow_joint_trajectory", execute, false);
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

  ROS_INFO("kinematic_val_plugin loaded.");
}

void KinematicValPlugin::PeriodicUpdate()
{
  PublishJointStates();
}

sensor_msgs::JointState KinematicValPlugin::GetJointStates()
{
  // FIXME: this is horribly hacky. this can be addressed if we just un-nest the models, and make the overstretching
  // stuff it's own plugin. not sure if that's possible. Another solution would be just make victor dynamic and use
  // gazebo_ros_control.
  auto const n_removed = std::strlen("hdt_michigan::");
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

void KinematicValPlugin::PublishJointStates()
{
  auto const msg = GetJointStates();
  joint_states_pub_.publish(msg);
}

void KinematicValPlugin::FollowJointTrajectory(const TrajServer::GoalConstPtr &goal)
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
    // Set the kinematic position of Val plus the rope grippers, then simulate
    {
      std::lock_guard lock(ros_mutex_);
      // Move Val to the specified joint configuration
      for (auto const &[joint_idx, joint_name] : enumerate(goal->trajectory.joint_names))
      {
        auto joint = model_->GetJoint("hdt_michigan::" + joint_name);
        if (joint)
        {
          joint->SetPosition(0, point.positions[joint_idx]);
        }
        else
        {
          ROS_ERROR_STREAM("Invalid joint: "
                           << "hdt_michigan::" + joint_name);
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
      ROS_WARN_STREAM(gripper1_->WorldPose().Pos().Z() << " "
      << gripper1_rope_link_->WorldPose().Pos().Z() << " "
      << gripper1_dist << " ");
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
        // Move Val to the specified joint configuration
        for (auto const &[joint_idx, joint_name] : enumerate(goal->trajectory.joint_names))
        {
          auto joint = model_->GetJoint("hdt_michigan::" + joint_name);
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

void KinematicValPlugin::TeleportGrippers()
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

void KinematicValPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void KinematicValPlugin::PrivateQueueThread()
{
  double constexpr timeout = 0.01;
  while (private_ros_node_->ok())
  {
    private_queue_.callAvailable(ros::WallDuration(timeout));
  }
}
}  // namespace gazebo
