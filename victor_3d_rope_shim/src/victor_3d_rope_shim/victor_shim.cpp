#include "victor_3d_rope_shim/assert.h"
#include "victor_3d_rope_shim/eigen_ros_conversions.hpp"
#include "victor_3d_rope_shim/eigen_transforms.hpp"
#include "victor_3d_rope_shim/moveit_print_state.h"
#include "victor_3d_rope_shim/victor_shim.hpp"

#include <std_msgs/String.h>
#include <std_srvs/Empty.h>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/path_utils.hpp>
#include <arc_utilities/pretty_print.hpp>
#include <arc_utilities/ros_helpers.hpp>
#include <memory>
#include <algorithm>

#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/GetPlanningScene.h>
#include <pluginlib/class_loader.h>
#include <boost/scoped_ptr.hpp>

namespace pm = peter_msgs;
namespace geomsg = geometry_msgs;
std::vector<std::string> const VICTOR_TORSO_BODIES = {
  "victor_pedestal",
  "victor_base_plate",
  "victor_left_arm_mount",
  "victor_right_arm_mount",
};
using ColorBuilder = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>;
auto constexpr TO_RADIANS = M_PI / 180.0;
auto constexpr TO_DEGREES = 180.0 / M_PI;
auto constexpr ALLOWED_PLANNING_TIME = 10.0;
auto constexpr TRANSLATION_STEP_SIZE = 0.001;

////////////////////////////////////////////////////////////////////////////////

Eigen::MatrixXd cleanZeros(Eigen::MatrixXd const& input, double const threshold = 1e-10)
{
  return (input.cwiseAbs().array() < threshold).select(0, input);
}

// std::pair<Eigen::Translation3d, Eigen::Translation3d> toGripperPositions(pm::Action const& action)
// {
//   MPS_ASSERT(action.action.size() == 6);
//   return { Eigen::Translation3d(action.action[0], action.action[1], action.action[2]),
//            Eigen::Translation3d(action.action[3], action.action[4], action.action[5]) };
// }

std::pair<Eigen::Translation3d, Eigen::Translation3d> toGripperPositions(geometry_msgs::Point const& g1,
                                                                         geometry_msgs::Point const& g2)
{
  return { Eigen::Translation3d(g1.x, g1.y, g1.z),
           Eigen::Translation3d(g2.x, g2.y, g2.z) };
}

////////////////////////////////////////////////////////////////////////////////

Pose lookupTransform(tf2_ros::Buffer const& buffer,
                     std::string const& parent_frame,
                     std::string const& child_frame,
                     ros::Time const& target_time = ros::Time(0),
                     ros::Duration const& timeout = ros::Duration(0))
{
  // Wait for up to timeout amount of time, then try to lookup the transform,
  // letting TF2's exception handling throw if needed
  if (!buffer.canTransform(parent_frame, child_frame, target_time, timeout))
  {
    ROS_WARN_STREAM("Unable to lookup transform between " << parent_frame << " and " << child_frame
                                                          << ". Defaulting to Identity.");
    return Pose::Identity();
  }
  auto const tform = buffer.lookupTransform(parent_frame, child_frame, target_time);
  return ConvertTo<Pose>(tform.transform);
}

template <typename Srv>
bool noOpServiceCallback(typename Srv::Request& req, typename Srv::Response& res)
{
  (void)req;
  (void)res;
  return true;
}

trajectory_msgs::JointTrajectory MergeTrajectories(trajectory_msgs::JointTrajectory const& traj_a,
                                                   trajectory_msgs::JointTrajectory const& traj_b)
{
  // Verify that the trajectories are mergable via simple concatenation of joint lists and points
  // to a "first level" of inspection
  {
    // Ignore the header for now, is this even used anywhere in particular?
    // MPS_ASSERT(traj_a.header == traj_b.header);
    auto const traj_a_names = std::set<std::string>(traj_a.joint_names.begin(), traj_a.joint_names.end());
    auto const traj_b_names = std::set<std::string>(traj_b.joint_names.begin(), traj_b.joint_names.end());
    std::vector<std::string> intersection(traj_a_names.size() + traj_b_names.size(), "");
    auto const end = std::set_intersection(traj_a_names.begin(), traj_a_names.end(),
                                          traj_b_names.begin(), traj_b_names.end(),
                                          intersection.begin());
    MPS_ASSERT(intersection.begin() == end && "Trajectories must be for different joints");
  }

  auto const mergable_size = std::min(traj_a.points.size(), traj_b.points.size());
  // Debugging
  {
    ROS_INFO_STREAM("Merging trajectories of starting sizes " << traj_a.points.size() << " and " << traj_b.points.size());
    if (traj_a.points.size() != mergable_size)
    {
      ROS_WARN_STREAM("Merging trajectories of uneven length, discarding " << traj_a.points.size() - mergable_size << " points from traj_a");
    }
    if (traj_b.points.size() != mergable_size)
    {
      ROS_WARN_STREAM("Merging trajectories of uneven length, discarding " << traj_b.points.size() - mergable_size << " points from traj_b");
    }
  }

  // Merge the trajectories, checking each individual point for compatability as we go
  trajectory_msgs::JointTrajectory merged = traj_a;
  merged.points.resize(mergable_size);
  merged.joint_names.insert(merged.joint_names.begin(), traj_b.joint_names.begin(), traj_b.joint_names.end());
  for (size_t idx = 0; idx < merged.points.size(); ++idx)
  {
    trajectory_msgs::JointTrajectoryPoint& point_a = merged.points.at(idx);
    trajectory_msgs::JointTrajectoryPoint const& point_b = traj_b.points.at(idx);

    MPS_ASSERT(point_a.positions.size()     == point_b.positions.size());
    MPS_ASSERT(point_a.velocities.size()    == point_b.velocities.size());
    MPS_ASSERT(point_a.accelerations.size() == point_b.accelerations.size());
    MPS_ASSERT(point_a.effort.size()        == point_b.effort.size());
    MPS_ASSERT(point_a.time_from_start      == point_b.time_from_start);

    point_a.positions.insert(    point_a.positions.begin(),     point_b.positions.begin(),     point_b.positions.end());
    point_a.velocities.insert(   point_a.velocities.begin(),    point_b.velocities.begin(),    point_b.velocities.end());
    point_a.accelerations.insert(point_a.accelerations.begin(), point_b.accelerations.begin(), point_b.accelerations.end());
    point_a.effort.insert(       point_a.effort.begin(),        point_b.effort.begin(),        point_b.effort.end());
  }

  return merged;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

VictorInterface::VictorInterface(ros::NodeHandle nh,
                                 ros::NodeHandle ph,
                                 std::shared_ptr<tf2_ros::Buffer> tf_buffer)
  // TOOD: ROS Param lookups
  : nh_(nh)
  , ph_(ph)
  , tf_buffer_(tf_buffer)
  , world_frame_("world_origin")
  , robot_frame_("victor_root")
  , table_frame_("table_surface")
  , left_tool_frame_("victor_left_tool")
  , right_tool_frame_("victor_right_tool")
  , worldTrobot(lookupTransform(*tf_buffer_, world_frame_, robot_frame_, ros::Time(0), ros::Duration(5)))
  , robotTworld(worldTrobot.inverse(Eigen::Isometry))
  , worldTtable(lookupTransform(*tf_buffer_, world_frame_, table_frame_, ros::Time(0), ros::Duration(5)))
  , tableTworld(worldTtable.inverse(Eigen::Isometry))
  , robotTtable(robotTworld * worldTtable)
  , tableTrobot(robotTtable.inverse(Eigen::Isometry))
  , model_loader_(std::make_unique<robot_model_loader::RobotModelLoader>())
  , robot_model_(model_loader_->getModel())
  , talker_(nh_.advertise<std_msgs::String>("polly", 10, false))
  , home_state_(robot_model_)
  , traj_goal_time_tolerance_(ROSHelpers::GetParam(ph_, "traj_goal_time_tolerance", 0.1))
{
  MPS_ASSERT(!robot_model_->getJointModelGroupNames().empty());
  vis_pub_ = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 10, true);

  scene_ = std::make_shared<Scene>();
  scene_->loadManipulators(robot_model_);
  left_arm_ = std::dynamic_pointer_cast<VictorManipulator>(scene_->manipulators.at(0));
  right_arm_ = std::dynamic_pointer_cast<VictorManipulator>(scene_->manipulators.at(2));
  MPS_ASSERT(left_arm_->arm->getName().find("left") != std::string::npos);
  MPS_ASSERT(right_arm_->arm->getName().find("right") != std::string::npos);

  auto const left_flange_name = left_arm_->arm->getLinkModels().back()->getName();
  left_tool_offset_ = lookupTransform(*tf_buffer_, left_flange_name, left_tool_frame_, ros::Time(0), ros::Duration(5));
  auto const right_flange_name = right_arm_->arm->getLinkModels().back()->getName();
  right_tool_offset_ = lookupTransform(*tf_buffer_, right_flange_name, right_tool_frame_, ros::Time(0), ros::Duration(5));

  // Retrieve the planning scene obstacles if possible, otherwise default to a saved set
  {
    auto const topic = ROSHelpers::GetParam<std::string>(ph_, "get_planning_scene_topic", "get_planning_scene");
    ros::ServiceClient client = nh_.serviceClient<moveit_msgs::GetPlanningScene>(topic);
    if (!client.waitForExistence(ros::Duration(3)))
    {
      ROS_WARN_STREAM("Service [" << nh_.getNamespace() << topic << " was not available. Defaulting to a saved set");

      // NB: PlannningScene assumes everything is defined relative to the
      //     robot base frame, so we have to deal with that here
      auto constexpr wall_width = 0.1;
      // Table
      {
        // TODO: confirm measurements/make ROS params
        // auto const table = std::make_shared<shapes::Box>(30 * 0.0254, 42 * 0.0254, 0.1);
        auto const table = std::make_shared<shapes::Box>(40 * 0.0254, 44 * 0.0254, worldTtable.translation().z());
        Pose const pose(Eigen::Translation3d(0.0, 0.0, -table->size[2] / 2.0));
        scene_->staticObstacles.push_back({ table, robotTworld * worldTtable * pose });
      }
      // Protect Andrew's monitors
      {
        auto const wall = std::make_shared<shapes::Box>(5, wall_width, 3);
        Pose const pose(Eigen::Translation3d(2.8 + wall_width / 2, 1.1, 1.5));
        scene_->staticObstacles.push_back({ wall, robotTworld * pose });
      }
      // Protect Dale's monitors
      {
        auto const wall = std::make_shared<shapes::Box>(5, wall_width, 3);
        Pose const pose(Eigen::Translation3d(2.8 + wall_width / 2, -1.1, 1.5));
        scene_->staticObstacles.push_back({ wall, robotTworld * pose });
      }
      // Protect stuff behind Victor
      {
        auto const wall = std::make_shared<shapes::Box>(0.1, 2.2 + wall_width, 3);
        Pose const pose(Eigen::Translation3d(0.3, 0.0, 1.5));
        scene_->staticObstacles.push_back({ wall, robotTworld * pose });
      }

      // Create obstacles on the table
      if (false)
      {
        // Table wall near Victor (negative x)
        {
          auto const wall = std::make_shared<shapes::Box>(7 * 0.0254, 44 * 0.0254, 4 * 0.0254);
          Pose const pose(Eigen::Translation3d(-16.5 * 0.0254, 0.0, wall->size[2] / 2.0));
          scene_->staticObstacles.push_back({ wall, robotTworld * worldTtable * pose });
        }
        // Table wall away from Victor (positive x)
        {
          auto const wall = std::make_shared<shapes::Box>(7 * 0.0254, 44 * 0.0254, 4 * 0.0254);
          Pose const pose(Eigen::Translation3d(16.5 * 0.0254, 0.0, wall->size[2] / 2.0));
          scene_->staticObstacles.push_back({ wall, robotTworld * worldTtable * pose });
        }
        // Table wall to Victor's right (negative y)
        {
          auto const wall = std::make_shared<shapes::Box>(26 * 0.0254, 3 * 0.0254, 4 * 0.0254);
          Pose const pose(Eigen::Translation3d(0.0, -20.5 * 0.0254, wall->size[2] / 2.0));
          scene_->staticObstacles.push_back({ wall, robotTworld * worldTtable * pose });
        }
        // Table wall to Victor's right (positive y)
        {
          auto const wall = std::make_shared<shapes::Box>(26 * 0.0254, 3 * 0.0254, 4 * 0.0254);
          Pose const pose(Eigen::Translation3d(0.0, 20.5 * 0.0254, wall->size[2] / 2.0));
          scene_->staticObstacles.push_back({ wall, robotTworld * worldTtable * pose });
        }
      }

      planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_model_, scene_->computeCollisionWorld());
    }
    else
    {
      planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_model_, scene_->computeCollisionWorld());

      // TODO: Is this request really what we want for a more generic task?
      moveit_msgs::GetPlanningSceneRequest req;
      req.components.components =
        moveit_msgs::PlanningSceneComponents::WORLD_OBJECT_NAMES |
        moveit_msgs::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY |
        moveit_msgs::PlanningSceneComponents::OCTOMAP |
        moveit_msgs::PlanningSceneComponents::TRANSFORMS |
        moveit_msgs::PlanningSceneComponents::OBJECT_COLORS;

      moveit_msgs::GetPlanningSceneResponse resp;
      client.call(req, resp);
      planning_scene_->processPlanningSceneWorldMsg(resp.scene.world);
    }
  }

  planning_scene_publisher_ = nh.advertise<moveit_msgs::PlanningScene>("planning_scene", 1, true);

  // Disable collisions between the static obstacles and Victor's non-moving parts
  planning_scene_->getAllowedCollisionMatrixNonConst().setEntry(Scene::OBSTACLES_NAME, false);
  for (auto const& name : VICTOR_TORSO_BODIES)
  {
    planning_scene_->getAllowedCollisionMatrixNonConst().setEntry(name, Scene::OBSTACLES_NAME, true);
  }

  // Set the home state to a pose near the table for each arm
  {
    home_state_.setToDefaultValues();
    home_state_.setJointGroupPositions(left_arm_->arm, left_arm_->qHome);
    home_state_.setJointGroupPositions(right_arm_->arm, right_arm_->qHome);
    home_state_.update();
    home_state_tool_poses_world_frame_ = getToolTransforms(home_state_);
    home_state_tool_poses_table_frame_ = Transform(tableTworld, home_state_tool_poses_world_frame_);
  }
  planning_scene_->setCurrentState(home_state_);

  joint_states_listener_ = std::make_shared<Listener<sensor_msgs::JointState>>(nh, "joint_states", true);

  // Attach the cat toy/wand to Victor - actually attaches to the state in the planning scene (I think)
  // TODO: attach the wand to states directly where needed
  // NB: This code is old and should not be trusted, it was not in use when copied from old codebase
  if (false)
  {
    moveit_msgs::AttachedCollisionObject attached_object;
    // Name of the link that this object will be rigidly attached to
    attached_object.link_name = left_flange_name;
    // Frame to interpret data in, and the name of the object
    attached_object.object.header.frame_id = "victor_left_gripper_palm_surface";
    attached_object.object.id = "left_cat_toy_wand";
    attached_object.object.operation = attached_object.object.ADD;
    // Define a cylinder to be attached
    {
      shape_msgs::SolidPrimitive cylinder;
      cylinder.type = cylinder.CYLINDER;
      cylinder.dimensions.resize(2);
      cylinder.dimensions[cylinder.CYLINDER_HEIGHT] = 13.5 * 0.0254 + 1;
      cylinder.dimensions[cylinder.CYLINDER_RADIUS] = 0.005;
      // Define the pose of the cylinder
      geomsg::Pose pose;
      pose.orientation.w = 1.0;
      pose.position.z = cylinder.dimensions[cylinder.CYLINDER_HEIGHT] / 2;

      attached_object.object.primitives.push_back(cylinder);
      attached_object.object.primitive_poses.push_back(pose);
    }
    // Define the links for which collision is ignored
    attached_object.touch_links = left_arm_->gripper->getLinkModelNames();

    // Actually attach the object
    planning_scene_->processAttachedCollisionObjectMsg(attached_object);
  }
  // planning_scene_->printKnownObjects(std::cerr);

  // std::cerr << "ACM:\n";
  // planning_scene_->getAllowedCollisionMatrix().print(std::cerr);

  trajectory_client_ = std::make_unique<TrajectoryClient>("follow_joint_trajectory", true);
}

void VictorInterface::test()
{
  // Test Jacobian math
  if (false)
  {
    Eigen::Vector3d const point(0, 0, 0.10);
    Pose const servo_frame = home_state_.getGlobalLinkTransform(left_arm_->arm->getLinkModels().back()) *
                             Eigen::Translation3d(point) * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX());

    auto const arm_base_transform =
        home_state_.getGlobalLinkTransform(left_arm_->arm->getJointModels().front()->getParentLinkModel());
    Pose const moveit_transform =
        Eigen::Translation3d(servo_frame.translation()) * Eigen::Quaterniond(arm_base_transform.rotation());

    Eigen::MatrixXd moveit_jacobian;
    home_state_.getJacobian(left_arm_->arm, left_arm_->arm->getLinkModels().back(), point, moveit_jacobian, false);

    auto const moveit_frame_jacobian =
        left_arm_->getJacobianServoFrame(home_state_, left_arm_->arm->getLinkModels().back(), moveit_transform);

    auto const servo_frame_jacobian =
        left_arm_->getJacobianServoFrame(home_state_, left_arm_->arm->getLinkModels().back(), servo_frame);

    auto const robot_frame_jacobian =
        left_arm_->getJacobianRobotFrame(home_state_, left_arm_->arm->getLinkModels().back(), servo_frame);

    // Testing some EigenHelpers functions
    auto const adjoint = EigenHelpers::AdjointFromTransform(servo_frame);
    Eigen::MatrixXd manual_adjoint_applied = adjoint * servo_frame_jacobian;
    MPS_ASSERT(cleanZeros(robot_frame_jacobian).isApprox(cleanZeros(manual_adjoint_applied)));
    for (int i = 0; i < robot_frame_jacobian.cols(); ++i)
    {
      Eigen::Matrix<double, 6, 1> const twist = servo_frame_jacobian.col(i);
      auto const transformed_twist = EigenHelpers::TransformTwist(servo_frame, twist);
      MPS_ASSERT(cleanZeros(transformed_twist).isApprox(cleanZeros(robot_frame_jacobian.col(i))) && "Inconsistency in "
                                                                                                    "EigenHelpers "
                                                                                                    "functions");
    }

    if (false)
    {
      std::cerr << "moveit_jacobian:\n" << cleanZeros(moveit_jacobian) << std::endl;
      std::cerr << "moveit_frame_jacobian:\n" << cleanZeros(moveit_frame_jacobian) << std::endl;
      std::cerr << "servo_frame_jacobian:\n" << cleanZeros(servo_frame_jacobian) << std::endl;
      std::cerr << "robot_frame_jacobian:\n" << cleanZeros(robot_frame_jacobian) << std::endl;
    }

    MPS_ASSERT(cleanZeros(moveit_jacobian).isApprox(cleanZeros(moveit_frame_jacobian)) &&
               "MoveIt and 'manual' Jacobian code should produce the same result");
  }

  // Left arm, palm forward, where the rope starts in Gazebo (gripper1)
  if (false)
  {
    Eigen::Quaterniond const root_to_tool_rot =
        Eigen::AngleAxisd( 90.0 * TO_RADIANS, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(180.0 * TO_RADIANS, Eigen::Vector3d::UnitZ());
    Eigen::Translation3d const root_to_tool(1.0, 0.3, 0.95);
    Pose const world_target_pose = worldTrobot * root_to_tool * root_to_tool_rot;
    auto const ik_solutions = left_arm_->IK(world_target_pose, robotTworld, home_state_, planning_scene_);
    std::cerr << "IK Solutions at target: " << ik_solutions.size() << std::endl;

    {
      geomsg::TransformStamped transform;
      transform.header.frame_id = world_frame_;
      transform.header.stamp = ros::Time::now();
      transform.child_frame_id = "world_target_pose";
      transform.transform = ConvertTo<geomsg::Transform>(world_target_pose);
      tf_broadcaster_.sendTransform(transform);
    }

    for (auto idx = 0ul; idx < ik_solutions.size(); ++idx)
    // for (auto idx = 0ul; idx < 1; ++idx)
    {
      trajectory_msgs::JointTrajectory traj;
      traj.joint_names = left_arm_->arm->getActiveJointModelNames();
      traj.points.resize(2);
      getCurrentRobotState().copyJointGroupPositions(left_arm_->arm, traj.points[0].positions);
      traj.points[0].time_from_start = ros::Duration(0);
      traj.points[1].positions = ik_solutions.at(idx);
      traj.points[1].time_from_start = ros::Duration(1);
      followTrajectory(traj);

      // robot_state::RobotState state(home_state_);
      // state.setJointGroupPositions(left_arm_->arm, ik_solutions[idx]);
      // state.update();
      // auto const jacobian = state.getJacobian(left_arm_->arm);

      std::cerr << "config = [" << PrettyPrint::PrettyPrint(ik_solutions[idx], false, ", ") << "]';\n";
      // std::cerr << "jacobian = [\n" << jacobian << "];\n";
      std::string asdf;
      std::cin >> asdf;
    }
  }

  // Right arm, palm forward, where the rope starts in Gazebo (gripper2)
  if (false)
  {
    Eigen::Quaterniond const root_to_tool_rot =
        Eigen::AngleAxisd( 90.0 * TO_RADIANS, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(180.0 * TO_RADIANS, Eigen::Vector3d::UnitZ());
    Eigen::Translation3d const root_to_tool(1.0, -0.3, 0.95);
    Pose const world_target_pose = worldTrobot * root_to_tool * root_to_tool_rot;
    auto const ik_solutions = right_arm_->IK(world_target_pose, robotTworld, home_state_, planning_scene_);
    std::cerr << "IK Solutions at target: " << ik_solutions.size() << std::endl;

    {
      geomsg::TransformStamped transform;
      transform.header.frame_id = world_frame_;
      transform.header.stamp = ros::Time::now();
      transform.child_frame_id = "world_target_pose";
      transform.transform = ConvertTo<geomsg::Transform>(world_target_pose);
      tf_broadcaster_.sendTransform(transform);
    }

    for (auto idx = 0ul; idx < ik_solutions.size(); ++idx)
    // for (auto idx = 0ul; idx < 1; ++idx)
    {
      trajectory_msgs::JointTrajectory traj;
      traj.joint_names = right_arm_->arm->getActiveJointModelNames();
      traj.points.resize(2);
      getCurrentRobotState().copyJointGroupPositions(right_arm_->arm, traj.points[0].positions);
      traj.points[0].time_from_start = ros::Duration(0);
      traj.points[1].positions = ik_solutions.at(idx);
      traj.points[1].time_from_start = ros::Duration(1);
      followTrajectory(traj);

      // robot_state::RobotState state(home_state_);
      // state.setJointGroupPositions(left_arm_->arm, ik_solutions[idx]);
      // state.update();
      // auto const jacobian = state.getJacobian(left_arm_->arm);

      std::cerr << "config = [" << PrettyPrint::PrettyPrint(ik_solutions[idx], false, ", ") << "]';\n";
      // std::cerr << "jacobian = [\n" << jacobian << "];\n";
      std::string asdf;
      std::cin >> asdf;
    }
  }
}

robot_state::RobotState VictorInterface::getCurrentRobotState() const
{
  robot_state::RobotState state(home_state_);
  state.setToDefaultValues();
  auto const joints = joint_states_listener_->waitForNew(1000.0);
  if (joints)
  {
    moveit::core::jointStateToRobotState(*joints, state);
  }
  else
  {
    ROS_WARN("getCurrentRobotState() has no data, returning default values. This ought to be impossible.");
  }
  state.update();
  return state;
}

std::pair<Pose, Pose> VictorInterface::getToolTransforms() const
{
  return getToolTransforms(getCurrentRobotState());
}

std::pair<Pose, Pose> VictorInterface::getToolTransforms(robot_state::RobotState const& state) const
{
  const Pose left_flange_pose = state.getGlobalLinkTransform(left_arm_->arm->getLinkModels().back());
  const Pose right_flange_pose = state.getGlobalLinkTransform(right_arm_->arm->getLinkModels().back());
  return { worldTrobot * left_flange_pose * left_tool_offset_, worldTrobot * right_flange_pose * left_tool_offset_ };
}

// http://docs.ros.org/melodic/api/moveit_tutorials/html/doc/motion_planning_api/motion_planning_api_tutorial.html
trajectory_msgs::JointTrajectory VictorInterface::plan(robot_state::RobotState const& start_state,
                                                       robot_state::RobotState const& goal_state)
{
  ///////////// Start ////////////////////////////////////////////////////////

  boost::scoped_ptr<pluginlib::ClassLoader<planning_interface::PlannerManager>> planner_plugin_loader;
  planning_interface::PlannerManagerPtr planner_instance;

  std::string planner_plugin_name = "ompl_interface/OMPLPlanner";
  if (!nh_.getParam("planning_plugin", planner_plugin_name))
  {
    ROS_WARN_STREAM("Could not find planner plugin name; defaulting to " << planner_plugin_name);
  }
  try
  {
    planner_plugin_loader.reset(new pluginlib::ClassLoader<planning_interface::PlannerManager>(
      "moveit_core", "planning_interface::PlannerManager"));
  }
  catch (pluginlib::PluginlibException& ex)
  {
    ROS_FATAL_STREAM("Exception while creating planning plugin loader " << ex.what());
  }
  try
  {
    planner_instance.reset(planner_plugin_loader->createUnmanagedInstance(planner_plugin_name));
    if (!planner_instance->initialize(robot_model_, nh_.getNamespace()))
    {
      ROS_FATAL_STREAM("Could not initialize planner instance");
    }
    ROS_INFO_STREAM("Using planning interface '" << planner_instance->getDescription() << "'");
  }
  catch (pluginlib::PluginlibException& ex)
  {
    const std::vector<std::string>& classes = planner_plugin_loader->getDeclaredClasses();
    std::stringstream ss;
    for (std::size_t i = 0; i < classes.size(); ++i)
    {
      ss << classes[i] << " ";
    }
    ROS_ERROR_STREAM("Exception while loading planner '" << planner_plugin_name << "': " << ex.what() << std::endl
                                                         << "Available plugins: " << ss.str());
  }

  ///////////// Joint Space Goals ////////////////////////////////////////////

  planning_interface::MotionPlanRequest req;
  planning_interface::MotionPlanResponse res;
  req.group_name = "both_arms";

  req.goal_constraints.clear();
  req.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(
    goal_state, robot_model_->getJointModelGroup(req.group_name)));
  req.allowed_planning_time = ALLOWED_PLANNING_TIME;

  /* Re-construct the planning context */
  planning_scene_->setCurrentState(start_state);
  ROS_WARN("The following line of code will likely give a 'Found empty JointState message' error,"
           " but can probably be ignored: https://github.com/ros-planning/moveit/issues/659");
  auto context = planner_instance->getPlanningContext(planning_scene_, req, res.error_code_);
  /* Call the Planner */
  context->solve(res);
  /* Check that the planning was successful */
  if (res.error_code_.val != res.error_code_.SUCCESS)
  {
    // planning_scene_->getAllowedCollisionMatrix().print(std::cerr);
    // Error check the input start and goal states
    {
      std::cerr << "Joint limits for start_state?\n";

      PRINT_STATE_POSITIONS_WITH_JOINT_LIMITS(start_state, left_arm_->arm, std::cerr);
      PRINT_STATE_POSITIONS_WITH_JOINT_LIMITS(start_state, right_arm_->arm, std::cerr);
      collision_detection::CollisionRequest collision_request;
      collision_detection::CollisionResult collision_result;
      planning_scene_->checkCollision(collision_request, collision_result, start_state);
      std::cerr << "Collision at start_state? " << collision_result.collision << std::endl;
    }
    {
      std::cerr << "Joint limits for goal_state?\n";
      PRINT_STATE_POSITIONS_WITH_JOINT_LIMITS(goal_state, left_arm_->arm, std::cerr);
      PRINT_STATE_POSITIONS_WITH_JOINT_LIMITS(goal_state, right_arm_->arm, std::cerr);
      collision_detection::CollisionRequest collision_request;
      collision_detection::CollisionResult collision_result;
      planning_scene_->checkCollision(collision_request, collision_result, goal_state);
      std::cerr << "Collision at goal_state? " << collision_result.collision << std::endl;
    }
    ROS_ERROR("Could not compute plan successfully");
    throw_arc_exception(std::runtime_error, "Planning failed");
  }

  moveit_msgs::MotionPlanResponse msg;
  res.getMessage(msg);

  // Debugging
  if (false)
  {
    ros::Publisher display_publisher =
      nh_.advertise<moveit_msgs::DisplayTrajectory>("move_group/display_planned_path", 1, true);
    moveit_msgs::DisplayTrajectory display_trajectory;
    display_trajectory.trajectory_start = msg.trajectory_start;
    display_trajectory.trajectory.push_back(msg.trajectory);
    display_publisher.publish(display_trajectory);
  }

  return msg.trajectory.joint_trajectory;
}

void VictorInterface::followTrajectory(trajectory_msgs::JointTrajectory const& traj)
{
  std_msgs::String executing_action_str;
  executing_action_str.data = "Moving";
  talker_.publish(executing_action_str);
  if (traj.points.size() == 0)
  {
    ROS_WARN("Asked to follow trajectory of length 0; ignoring.");
    return;
  }
  if (!trajectory_client_->waitForServer(ros::Duration(3.0)))
  {
    ROS_WARN("Trajectory server not connected.");
  }

  control_msgs::FollowJointTrajectoryGoal goal;
  goal.trajectory = traj;
  for (const auto& name : goal.trajectory.joint_names)
  {
    // TODO: set thresholds for this task
    // NB: Dale: I think these are ignored by downstream code
    control_msgs::JointTolerance tol;
    tol.name = name;
    tol.position = 0.05;
    tol.velocity = 0.5;
    tol.acceleration = 1.0;
    goal.goal_tolerance.push_back(tol);
  }
  goal.goal_time_tolerance = traj_goal_time_tolerance_;
  ROS_INFO("Sending goal ...");
  trajectory_client_->sendGoalAndWait(goal);

  ROS_INFO("Goal finished");
}

void VictorInterface::gotoHome()
{
  ROS_INFO("Going home");
  joint_states_listener_->waitForNew(1000.0);
  auto const start_state = getCurrentRobotState();
  auto const goal_state = home_state_;
  ROS_INFO("Planning to home");
  auto const traj = plan(start_state, goal_state);
  followTrajectory(traj);
  ROS_INFO("At home");
  arc_helpers::Sleep(0.5);
}

void VictorInterface::moveInRobotFrame(std::pair<Eigen::Translation3d, Eigen::Translation3d> const& gripper_positions)
{
  std::pair<Eigen::Translation3d, Eigen::Translation3d> table_frame{
    (tableTrobot * gripper_positions.first).translation(),
    (tableTrobot * gripper_positions.second).translation()
  };
  moveInTableFrame(table_frame);
}

void VictorInterface::moveInTableFrame(std::pair<Eigen::Translation3d, Eigen::Translation3d> const& gripper_positions)
{
  joint_states_listener_->waitForNew(1000.0);
  moveInTableFrameJacobianIk(gripper_positions);
}

void VictorInterface::moveInTableFrameJacobianIk(
    std::pair<Eigen::Translation3d, Eigen::Translation3d> const& gripper_positions)
{
  auto const current_state = getCurrentRobotState();
  auto const current_tool_poses = getToolTransforms(current_state);

  auto target_poses = Transform(worldTtable, gripper_positions);
  target_poses.first.linear() = home_state_tool_poses_world_frame_.first.linear();
  target_poses.second.linear() = home_state_tool_poses_world_frame_.second.linear();

  // Debugging
  if (true)
  {
    auto const current_tool_poses_table_frame = Transform(tableTworld, current_tool_poses);
    // Current in table frame
    if (false)
    {
      {
        geomsg::TransformStamped transform;
        transform.header.frame_id = world_frame_;
        transform.header.stamp = ros::Time::now();
        transform.child_frame_id = "current_tool_pose_left";
        transform.transform = ConvertTo<geomsg::Transform>(current_tool_poses_table_frame.first);
        tf_broadcaster_.sendTransform(transform);
      }
      {
        geomsg::TransformStamped transform;
        transform.header.frame_id = world_frame_;
        transform.header.stamp = ros::Time::now();
        transform.child_frame_id = "current_tool_pose_right";
        transform.transform = ConvertTo<geomsg::Transform>(current_tool_poses_table_frame.second);
        tf_broadcaster_.sendTransform(transform);
      }
    }
    // Current in world frame
    if (false)
    {
      {
        geomsg::TransformStamped transform;
        transform.header.frame_id = world_frame_;
        transform.header.stamp = ros::Time::now();
        transform.child_frame_id = "current_tool_pose_left";
        transform.transform = ConvertTo<geomsg::Transform>(current_tool_poses.first);
        tf_broadcaster_.sendTransform(transform);
      }
      {
      geomsg::TransformStamped transform;
      transform.header.frame_id = world_frame_;
      transform.header.stamp = ros::Time::now();
      transform.child_frame_id = "current_tool_pose_right";
      transform.transform = ConvertTo<geomsg::Transform>(current_tool_poses.second);
      tf_broadcaster_.sendTransform(transform);
      }
    }
    // Desired in table frame
    if (false)
    {
      {
        geomsg::TransformStamped transform;
        transform.header.frame_id = table_frame_;
        transform.header.stamp = ros::Time::now();
        transform.child_frame_id = "desired_tool_pose_table_frame_left";
        transform.transform = ConvertTo<geomsg::Transform>(gripper_positions.first);
        tf_broadcaster_.sendTransform(transform);
      }
      {
        geomsg::TransformStamped transform;
        transform.header.frame_id = table_frame_;
        transform.header.stamp = ros::Time::now();
        transform.child_frame_id = "desired_tool_pose_table_frame_right";
        transform.transform = ConvertTo<geomsg::Transform>(gripper_positions.second);
        tf_broadcaster_.sendTransform(transform);
      }
    }
    // Desired in world frame
    if (true)
    {
      {
        geomsg::TransformStamped transform;
        transform.header.frame_id = world_frame_;
        transform.header.stamp = ros::Time::now();
        transform.child_frame_id = "desired_tool_pose_world_frame_left";
        transform.transform = ConvertTo<geomsg::Transform>(target_poses.first);
        tf_broadcaster_.sendTransform(transform);
      }
      {
        geomsg::TransformStamped transform;
        transform.header.frame_id = world_frame_;
        transform.header.stamp = ros::Time::now();
        transform.child_frame_id = "desired_tool_pose_world_frame_right";
        transform.transform = ConvertTo<geomsg::Transform>(target_poses.second);
        tf_broadcaster_.sendTransform(transform);
      }
    }
  }

  // Create paths for each tool with an equal number of waypoints
  Eigen::Vector3d const left_delta = target_poses.first.translation() - current_tool_poses.first.translation();
  Eigen::Vector3d const right_delta = target_poses.first.translation() - current_tool_poses.first.translation();
  auto const max_dist = std::max(left_delta.norm(), right_delta.norm());
  auto const steps = static_cast<int>(std::ceil(max_dist / TRANSLATION_STEP_SIZE));
  auto const left_path = [&]
  {
    EigenHelpers::VectorVector3d path;
    MPS_ASSERT(left_arm_->interpolate(
      current_tool_poses.first.translation(),
      target_poses.first.translation(),
      path,
      steps));
    return path;
  }();
  auto const right_path = [&]
  {
    EigenHelpers::VectorVector3d path;
    MPS_ASSERT(left_arm_->interpolate(
      current_tool_poses.second.translation(),
      target_poses.second.translation(),
      path,
      steps));
    return path;
  }();
  MPS_ASSERT(left_path.size() == right_path.size() && "Later code assumes these are of equal length for syncronization");

  // Debugging - visualize interpolated path in world frame
  if (true)
  {
    visualization_msgs::MarkerArray msg;
    msg.markers.resize(2);
    // Left
    {
      auto& m = msg.markers[0];
      m.ns = "left_interpolation_path";
      m.header.frame_id = world_frame_;
      m.header.stamp = ros::Time::now();
      m.action = m.ADD;
      m.type = m.POINTS;
      m.points.resize(left_path.size());
      m.scale.x = 0.01;
      m.scale.y = 0.01;

      m.colors.resize(left_path.size());
      auto const start_color = ColorBuilder::MakeFromFloatColors(0, 1, 0, 1);
      auto const end_color = ColorBuilder::MakeFromFloatColors(1, 1, 0, 1);

      for (size_t idx = 0; idx < left_path.size(); ++idx)
      {
        m.points[idx].x = left_path[idx].x();
        m.points[idx].y = left_path[idx].y();
        m.points[idx].z = left_path[idx].z();

        auto const ratio = static_cast<float>(idx) / static_cast<float>(std::max((left_path.size() - 1), 1ul));
        m.colors[idx] = arc_helpers::InterpolateColor(start_color, end_color, ratio);
      }
    }
    // Right
    {
      auto& m = msg.markers[1];
      m.ns = "right_interpolation_path";
      m.header.frame_id = world_frame_;
      m.header.stamp = ros::Time::now();
      m.action = m.ADD;
      m.type = m.POINTS;
      m.points.resize(right_path.size());
      m.scale.x = 0.01;
      m.scale.y = 0.01;

      m.colors.resize(right_path.size());
      auto const start_color = ColorBuilder::MakeFromFloatColors(0, 1, 0, 1);
      auto const end_color = ColorBuilder::MakeFromFloatColors(1, 1, 0, 1);

      for (size_t idx = 0; idx < right_path.size(); ++idx)
      {
        m.points[idx].x = right_path[idx].x();
        m.points[idx].y = right_path[idx].y();
        m.points[idx].z = right_path[idx].z();

        auto const ratio = static_cast<float>(idx) / static_cast<float>(std::max((right_path.size() - 1), 1ul));
        m.colors[idx] = arc_helpers::InterpolateColor(start_color, end_color, ratio);
      }
    }

    vis_pub_.publish(msg);
  }

  auto const left_cmd = [&]
  {
    auto const flange_home_frame = home_state_.getGlobalLinkTransform(left_arm_->arm->getLinkModels().back());
    trajectory_msgs::JointTrajectory cmd;
    MPS_ASSERT(left_arm_->jacobianPath3D(
      left_path,
      flange_home_frame.rotation(),
      robotTworld,
      left_tool_offset_,
      current_state,
      planning_scene_,
      cmd));
    return cmd;
  }();
  auto const right_cmd = [&]
  {
    auto const flange_home_frame = home_state_.getGlobalLinkTransform(right_arm_->arm->getLinkModels().back());
    trajectory_msgs::JointTrajectory cmd;
    MPS_ASSERT(right_arm_->jacobianPath3D(
      right_path,
      flange_home_frame.rotation(),
      robotTworld,
      right_tool_offset_,
      current_state,
      planning_scene_,
      cmd));
    return cmd;
  }();

  // Debugging - visualize JacobianIK result tip in table frame
  if (true)
  {
    visualization_msgs::MarkerArray msg;
    msg.markers.resize(2);
    // Left
    {
      auto& m = msg.markers[0];
      m.ns = "left_ik_result";
      m.header.frame_id = world_frame_;
      m.header.stamp = ros::Time::now();
      m.action = m.ADD;
      m.type = m.POINTS;
      m.points.resize(left_cmd.points.size());
      m.scale.x = 0.01;
      m.scale.y = 0.01;

      m.colors.resize(left_cmd.points.size());
      auto const start_color = ColorBuilder::MakeFromFloatColors(0, 0, 1, 1);
      auto const end_color = ColorBuilder::MakeFromFloatColors(1, 0, 1, 1);

      auto state = home_state_;
      state.setToDefaultValues();
      for (size_t idx = 0; idx < left_cmd.points.size(); ++idx)
      {
        state.setJointGroupPositions(left_arm_->arm, left_cmd.points[idx].positions);
        state.updateLinkTransforms();
        Pose const tool_pose = getToolTransforms(state).first;

        m.points[idx].x = tool_pose.translation().x();
        m.points[idx].y = tool_pose.translation().y();
        m.points[idx].z = tool_pose.translation().z();
        auto const ratio = static_cast<float>(idx) / static_cast<float>(std::max((left_cmd.points.size() - 1), 1ul));
        m.colors[idx] = arc_helpers::InterpolateColor(start_color, end_color, ratio);
      }
    }
    // Right
    {
      auto& m = msg.markers[1];
      m.ns = "right_ik_result";
      m.header.frame_id = world_frame_;
      m.header.stamp = ros::Time::now();
      m.action = m.ADD;
      m.type = m.POINTS;
      m.points.resize(right_cmd.points.size());
      m.scale.x = 0.01;
      m.scale.y = 0.01;

      m.colors.resize(right_cmd.points.size());
      auto const start_color = ColorBuilder::MakeFromFloatColors(0, 0, 1, 1);
      auto const end_color = ColorBuilder::MakeFromFloatColors(1, 0, 1, 1);

      auto state = home_state_;
      state.setToDefaultValues();
      for (size_t idx = 0; idx < right_cmd.points.size(); ++idx)
      {
        state.setJointGroupPositions(right_arm_->arm, right_cmd.points[idx].positions);
        state.updateLinkTransforms();
        Pose const tool_pose = getToolTransforms(state).second;

        m.points[idx].x = tool_pose.translation().x();
        m.points[idx].y = tool_pose.translation().y();
        m.points[idx].z = tool_pose.translation().z();
        auto const ratio = static_cast<float>(idx) / static_cast<float>(std::max((right_cmd.points.size() - 1), 1ul));
        m.colors[idx] = arc_helpers::InterpolateColor(start_color, end_color, ratio);
      }

      vis_pub_.publish(msg);
    }
  }

  // Merge the trajectories and then verify that the result is still collision free
  auto merged_cmd = MergeTrajectories(left_cmd, right_cmd);
  collision_detection::CollisionRequest collision_req;
	collision_detection::CollisionResult collision_res;
  robot_state::RobotState state = current_state;
  for (size_t idx = 0; idx < merged_cmd.points.size(); ++idx)
  {
    state.setJointGroupPositions(robot_model_->getJointModelGroup("both_arms"), merged_cmd.points[idx].positions);
    state.update();
    planning_scene_->checkCollision(collision_req, collision_res, state);
    if (collision_res.collision)
    {
      ROS_WARN_STREAM("Collision at idx " << idx << " in merged arm trajectories. Returning valid portion only");
      merged_cmd.points.resize(idx);
      break;
    }
    collision_res.clear();
  }

  followTrajectory(merged_cmd);
}

void VictorInterface::visualizePlanningScene()
{
  moveit_msgs::PlanningScene scene_msg;
  planning_scene_->setCurrentState(getCurrentRobotState());
  planning_scene_->getPlanningSceneMsg(scene_msg);
  scene_msg.is_diff = false;
  planning_scene_publisher_.publish(scene_msg);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

VictorShim::VictorShim(ros::NodeHandle nh, ros::NodeHandle ph)
{
  // Connect to external sensors and the robot
  {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>();
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    // ROS_INFO("Waiting 5 seconds for TF to collect data");
    // arc_helpers::Sleep(5);
    victor_ = std::make_shared<VictorInterface>(nh, ph, tf_buffer_);
  }

  // DualGripper control/exection
  {
    execute_traj_srv_ = nh.advertiseService("execute_dual_gripper_trajectory", &VictorShim::executeTrajectory, this);
  }
}

////////////////////////////////////////////////////////////////////////////////

bool VictorShim::executeTrajectory(pm::DualGripperTrajectory::Request& req,
                                   pm::DualGripperTrajectory::Response& res)
{
  (void)res;

  if (req.gripper1_points.size() != req.gripper2_points.size())
  {
    ROS_WARN("Mismatched gripper trajectory sizes, doing nothing.");
    return false;
  }

  // NB: positions are assumed to be in `victor_root` frame.
  ROS_INFO_STREAM("Executing dual gripper trajectory of length " << req.gripper1_points.size());
  for (size_t idx = 0; idx < req.gripper1_points.size(); ++idx)
  {
    victor_->moveInRobotFrame(toGripperPositions(req.gripper1_points[idx], req.gripper2_points[idx]));
  }

  ROS_INFO("Done trajectory");
  return true;
}
