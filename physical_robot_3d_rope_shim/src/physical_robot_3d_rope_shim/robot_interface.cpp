#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <std_msgs/String.h>
#include <arc_utilities/arc_exceptions.hpp>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/ros_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>
#include <pluginlib/class_loader.hpp>

#include "physical_robot_3d_rope_shim/robot_interface.hpp"

#include "assert.hpp"
#include "eigen_ros_conversions.hpp"
#include "eigen_transforms.hpp"
#include "ostream_operators.hpp"

auto constexpr ALLOWED_PLANNING_TIME = 10.0;
namespace pi = planning_interface;
namespace ps = planning_scene;
namespace eh = EigenHelpers;
using ColorBuilder = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Pose lookupTransform(tf2_ros::Buffer const& buffer, std::string const& parent_frame, std::string const& child_frame,
                     ros::Time const& target_time = ros::Time(0), ros::Duration const& timeout = ros::Duration(0))
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

PointSequence interpolate(const Eigen::Vector3d& from, const Eigen::Vector3d& to, const int steps)
{
  PointSequence sequence;
  MPS_ASSERT(steps > 1);
  sequence.resize(steps);
  for (int i = 0; i < steps; ++i)
  {
    const double t = i / static_cast<double>(steps - 1);
    sequence[i] = ((1.0 - t) * from) + (t * to);
  }
  return sequence;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RobotInterface::RobotInterface(ros::NodeHandle nh, ros::NodeHandle ph, std::shared_ptr<tf2_ros::Buffer> tf_buffer,
                               std::string const& group)
  : nh_(nh)
  , ph_(ph)
  , model_loader_(std::make_unique<robot_model_loader::RobotModelLoader>())
  , model_(model_loader_->getModel())
  , planning_group_(group)
  , jmg_(model_->getJointModelGroup(planning_group_))
  , num_ees_(jmg_->getAttachedEndEffectorNames().size())

  , tf_broadcaster_()
  , vis_pub_(nh_.advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 10, true))

  , tf_buffer_(tf_buffer)
  , world_frame_("world_origin")
  , robot_frame_(model_->getRootLinkName())
  , worldTrobot(lookupTransform(*tf_buffer_, world_frame_, robot_frame_, ros::Time(0), ros::Duration(1)))
  , robotTworld(worldTrobot.inverse(Eigen::Isometry))

  , home_state_(model_)

  , translation_step_size_(ROSHelpers::GetParamRequired<double>(ph_, "translation_step_size", __func__))
{
  auto const& ees = jmg_->getAttachedEndEffectorNames();

  tool_names_.resize(num_ees_);
  tool_offsets_.resize(num_ees_);
  for (auto idx = 0ul; idx < num_ees_; ++idx)
  {
    auto const ee_baselink = model_->getEndEffector(ees[idx])->getLinkModelNames().front();
    tool_names_[idx] = ees[idx] + "_tool";
    tool_offsets_[idx] = lookupTransform(*tf_buffer, ee_baselink, tool_names_[idx], ros::Time(0), ros::Duration(1));
  }
}

void RobotInterface::configureHomeState()
{
  q_home_ = lookupQHome();
  home_state_.setToDefaultValues();
  home_state_.setJointGroupPositions(jmg_, q_home_);
  home_state_.update();
  home_state_tool_poses_ = getToolTransforms(home_state_);
}

PoseSequence RobotInterface::getToolTransforms(robot_state::RobotState const& state) const
{
  auto const& ees = jmg_->getAttachedEndEffectorNames();
  PoseSequence poses(num_ees_);
  for (auto idx = 0ul; idx < num_ees_; ++idx)
  {
    auto const ee_baselink = model_->getEndEffector(ees[idx])->getLinkModelNames().front();
    poses[idx] = worldTrobot * state.getGlobalLinkTransform(ee_baselink) * tool_offsets_[idx];
  }
  return poses;
}

trajectory_msgs::JointTrajectory RobotInterface::plan(ps::PlanningScenePtr planning_scene,
                                                      robot_state::RobotState const& goal_state)
{
  ///////////// Start ////////////////////////////////////////////////////////

  std::shared_ptr<pluginlib::ClassLoader<pi::PlannerManager>> planner_plugin_loader;
  pi::PlannerManagerPtr planner_instance;

  std::string planner_plugin_name = "ompl_interface/OMPLPlanner";
  if (!nh_.getParam("planning_plugin", planner_plugin_name))
  {
    ROS_INFO("Could not find planner plugin name; defaulting to %s", planner_plugin_name);
  }
  try
  {
    planner_plugin_loader = std::make_shared<pluginlib::ClassLoader<pi::PlannerManager>>("moveit_core", "planning_"
                                                                                                        "interface::"
                                                                                                        "PlannerManage"
                                                                                                        "r");
  }
  catch (pluginlib::PluginlibException& ex)
  {
    ROS_FATAL("Exception while creating planning plugin loader %s", ex.what());
  }
  try
  {
    planner_instance.reset(planner_plugin_loader->createUnmanagedInstance(planner_plugin_name));
    if (!planner_instance->initialize(model_, nh_.getNamespace()))
    {
      ROS_FATAL("Could not initialize planner instance");
    }
    ROS_INFO("Using planning interface '%s'", planner_instance->getDescription());
  }
  catch (pluginlib::PluginlibException& ex)
  {
    const std::vector<std::string>& classes = planner_plugin_loader->getDeclaredClasses();
    std::stringstream ss;
    for (std::size_t i = 0; i < classes.size(); ++i)
    {
      ss << classes[i] << " ";
    }
    ROS_ERROR("Exception while loading planner '%s': %s\nAvailable plugins: %s", planner_plugin_name, ex.what(),
              ss.str());
  }

  ///////////// Joint Space Goals ////////////////////////////////////////////

  pi::MotionPlanRequest req;
  pi::MotionPlanResponse res;
  req.group_name = planning_group_;

  auto const& start_state = planning_scene->getCurrentState();

  req.goal_constraints.clear();
  req.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(goal_state, jmg_));
  req.allowed_planning_time = ALLOWED_PLANNING_TIME;

  /* Re-construct the planning context */
  ROS_WARN("The following line of code will likely give a 'Found empty JointState message' error,"
           " but can probably be ignored: https://github.com/ros-planning/moveit/issues/659");
  auto context = planner_instance->getPlanningContext(planning_scene, req, res.error_code_);
  /* Call the Planner */
  context->solve(res);
  /* Check that the planning was successful */
  if (res.error_code_.val != res.error_code_.SUCCESS)
  {
    // Error check the input start and goal states
    {
      std::cerr << "Joint limits for start_state?\n";
      start_state.printStatePositionsWithJointLimits(jmg_, std::cerr);
      collision_detection::CollisionRequest collision_request;
      collision_detection::CollisionResult collision_result;
      planning_scene->checkCollision(collision_request, collision_result, start_state);
      std::cerr << "Collision at start_state? " << collision_result.collision << std::endl;
    }
    {
      std::cerr << "Joint limits for goal_state?\n";
      goal_state.printStatePositionsWithJointLimits(jmg_, std::cerr);
      collision_detection::CollisionRequest collision_request;
      collision_detection::CollisionResult collision_result;
      planning_scene->checkCollision(collision_request, collision_result, goal_state);
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

trajectory_msgs::JointTrajectory RobotInterface::moveInRobotFrame(ps::PlanningScenePtr planning_scene,
                                                                  PointSequence const& target_tool_positions)
{
  return moveInWorldFrame(planning_scene, Transform(robotTworld, target_tool_positions));
}

trajectory_msgs::JointTrajectory RobotInterface::moveInWorldFrame(ps::PlanningScenePtr planning_scene,
                                                                  PointSequence const& target_tool_positions)
{
  auto const& start_state = planning_scene->getCurrentState();
  auto const start_tool_transforms = getToolTransforms(start_state);

  // Verify that the start state is collision free
  {
    collision_detection::CollisionRequest collision_request;
    collision_detection::CollisionResult collision_result;
    planning_scene->checkCollision(collision_request, collision_result, start_state);
    if (collision_result.collision)
    {
      std::cerr << "Collision at start_state:\n" << collision_result << std::endl;
      std::cerr << "Joint limits at start_state\n";
      start_state.printStatePositionsWithJointLimits(jmg_, std::cerr);

      std::string asdf = "";
      while (asdf != "c")
      {
        std::cerr << "Waiting for input/debugger attaching " << std::flush;
        std::cin >> asdf;
      }
    }
  }

  // Create paths for each tool with an equal number of waypoints
  double max_dist = 0;
  for (auto idx = 0ul; idx < num_ees_; ++idx)
  {
    auto const dist = (target_tool_positions[idx] - start_tool_transforms[idx].translation()).norm();
    max_dist = std::max(max_dist, dist);
  }
  if (max_dist < translation_step_size_)
  {
    ROS_INFO("Motion of distance %f requested. Ignoring", max_dist);
    return {};
  }
  auto const steps = static_cast<int>(std::ceil(max_dist / translation_step_size_)) + 1;
  std::vector<eh::VectorVector3d> tool_paths(num_ees_);
  for (auto idx = 0ul; idx < num_ees_; ++idx)
  {
    tool_paths[idx] = interpolate(start_tool_transforms[idx].translation(), target_tool_positions[idx], steps);
  }

  // Debugging - visualize interpolated path
  {
    visualization_msgs::MarkerArray msg;
    auto const stamp = ros::Time::now();
    for (auto tool_idx = 0ul; tool_idx < num_ees_; ++tool_idx)
    {
      auto& path = tool_paths[tool_idx];
      auto& m = msg.markers[tool_idx];
      m.ns = tool_names_[tool_idx] + "_interpolation_path";
      m.header.frame_id = world_frame_;
      m.header.stamp = stamp;
      m.action = m.ADD;
      m.type = m.POINTS;
      m.points.resize(steps);
      m.scale.x = 0.01;
      m.scale.y = 0.01;

      m.colors.resize(steps);
      auto const start_color = ColorBuilder::MakeFromFloatColors(0, 1, 0, 1);
      auto const end_color = ColorBuilder::MakeFromFloatColors(1, 1, 0, 1);

      for (auto step_idx = 0; step_idx < steps; ++step_idx)
      {
        m.points[step_idx] = ConvertTo<geometry_msgs::Point>(path[step_idx]);
        auto const ratio = static_cast<float>(step_idx) / static_cast<float>(std::max(steps - 1, 1));
        m.colors[step_idx] = arc_helpers::InterpolateColor(start_color, end_color, ratio);
      }
    }

    vis_pub_.publish(msg);
  }

  arc_helpers::Sleep(1.0);

  return {};
}
