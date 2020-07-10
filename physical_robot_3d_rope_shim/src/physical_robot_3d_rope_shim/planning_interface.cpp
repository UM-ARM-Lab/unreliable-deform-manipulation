#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <std_msgs/String.h>
#include <arc_utilities/arc_exceptions.hpp>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/ros_helpers.hpp>
#include <pluginlib/class_loader.hpp>

#include "physical_robot_3d_rope_shim/planning_interface.hpp"

#include "assert.hpp"
#include "eigen_ros_conversions.hpp"
#include "eigen_transforms.hpp"
#include "ostream_operators.hpp"

auto constexpr ALLOWED_PLANNING_TIME = 10.0;
namespace pi = planning_interface;
namespace ps = planning_scene;
namespace eh = EigenHelpers;
namespace gm = geometry_msgs;
using ColorBuilder = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>;
using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>;
using VecArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

static Pose lookupTransform(tf2_ros::Buffer const& buffer, std::string const& parent_frame,
                            std::string const& child_frame, ros::Time const& target_time = ros::Time(0),
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static PointSequence interpolate(Eigen::Vector3d const& from, Eigen::Vector3d const& to, int const steps)
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

static PoseSequence calcPoseError(PoseSequence const& curr, PoseSequence const& goal)
{
  assert(curr.size() == goal.size());
  auto const n = curr.size();
  PoseSequence err(n);
  for (auto idx = 0ul; idx < n; ++idx)
  {
    err[idx] = curr[idx].inverse(Eigen::Isometry) * goal[idx];
  }
  return err;
}

static std::pair<Eigen::VectorXd, Eigen::VectorXd> calcVecError(PoseSequence const& err)
{
  Eigen::VectorXd posVec(err.size() * 3);
  Eigen::VectorXd rotVec(err.size() * 3);
  for (auto idx = 0ul; idx < err.size(); ++idx)
  {
    auto const twist = EigenHelpers::TwistUnhat(err[idx].matrix().log());
    posVec.segment<3>(3 * idx) = err[idx].translation();
    rotVec.segment<3>(3 * idx) = twist.tail<3>();
  }
  return { posVec, rotVec };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PlanningInterace::PlanningInterace(ros::NodeHandle nh, ros::NodeHandle ph, std::shared_ptr<tf2_ros::Buffer> tf_buffer,
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

void PlanningInterace::configureHomeState()
{
  q_home_ = lookupQHome();
  home_state_.setToDefaultValues();
  home_state_.setJointGroupPositions(jmg_, q_home_);
  home_state_.update();
  home_state_tool_poses_ = getToolTransforms(home_state_);
  robot_nominal_tool_orientations_.resize(num_ees_);
  for (auto idx = 0ul; idx < num_ees_; ++idx)
  {
    robot_nominal_tool_orientations_[idx] = (robotTworld * home_state_tool_poses_[idx]).rotation();
  }
}

PoseSequence PlanningInterace::getToolTransforms(robot_state::RobotState const& state) const
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

trajectory_msgs::JointTrajectory PlanningInterace::plan(ps::PlanningScenePtr planning_scene,
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
      collision_detection::CollisionRequest request;
      request.contacts = true;
      request.verbose = true;
      collision_detection::CollisionResult result;
      planning_scene->checkCollision(request, result, start_state);
      std::cerr << "Collision at start_state? " << result.collision << std::endl;
    }
    {
      std::cerr << "Joint limits for goal_state?\n";
      goal_state.printStatePositionsWithJointLimits(jmg_, std::cerr);
      collision_detection::CollisionRequest request;
      request.contacts = true;
      request.verbose = true;
      collision_detection::CollisionResult result;
      planning_scene->checkCollision(request, result, goal_state);
      std::cerr << "Collision at goal_state? " << result.collision << std::endl;
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

trajectory_msgs::JointTrajectory PlanningInterace::moveInRobotFrame(ps::PlanningScenePtr planning_scene,
                                                                  PointSequence const& target_tool_positions)
{
  return moveInWorldFrame(planning_scene, Transform(robotTworld, target_tool_positions));
}

trajectory_msgs::JointTrajectory PlanningInterace::moveInWorldFrame(ps::PlanningScenePtr planning_scene,
                                                                  PointSequence const& target_tool_positions)
{
  MPS_ASSERT(planning_scene);
  auto const& start_state = planning_scene->getCurrentState();
  auto const start_tool_transforms = getToolTransforms(start_state);

  // Verify that the start state is collision free
  {
    collision_detection::CollisionRequest request;
    collision_detection::CollisionResult result;
    planning_scene->checkCollision(request, result, start_state);
    if (result.collision)
    {
      request.contacts = true;
      request.verbose = true;
      result.clear();
      planning_scene->checkCollision(request, result, start_state);

      std::cerr << "Collision at start_state:\n" << result << std::endl;
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
  std::vector<PointSequence> tool_paths(num_ees_);
  for (auto idx = 0ul; idx < num_ees_; ++idx)
  {
    tool_paths[idx] = interpolate(start_tool_transforms[idx].translation(), target_tool_positions[idx], steps);
  }

  // Debugging - visualize interpolated path
  {
    visualization_msgs::MarkerArray msg;
    msg.markers.resize(num_ees_);

    auto const stamp = ros::Time::now();
    for (auto tool_idx = 0ul; tool_idx < num_ees_; ++tool_idx)
    {
      auto& path = tool_paths[tool_idx];
      auto& m = msg.markers[tool_idx];
      m.ns = tool_names_[tool_idx] + "_interpolation_path";
      m.id = 1;
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
        m.points[step_idx] = ConvertTo<gm::Point>(path[step_idx]);
        auto const ratio = static_cast<float>(step_idx) / static_cast<float>(std::max(steps - 1, 1));
        m.colors[step_idx] = arc_helpers::InterpolateColor(start_color, end_color, ratio);
      }
    }

    vis_pub_.publish(msg);
  }

  auto const cmd = jacobianPath3d(planning_scene, tool_paths);

  // Debugging - visualize JacobiakIK result tip
  if (true)
  {
    visualization_msgs::MarkerArray msg;
    msg.markers.resize(num_ees_);

    auto const stamp = ros::Time::now();
    for (auto tool_idx = 0ul; tool_idx < num_ees_; ++tool_idx)
    {
      auto& m = msg.markers[tool_idx];
      m.ns = tool_names_[tool_idx] + "_ik_result";
      m.id = 1;
      m.header.frame_id = world_frame_;
      m.header.stamp = stamp;
      m.action = m.ADD;
      m.type = m.POINTS;
      m.points.resize(cmd.points.size());
      m.scale.x = 0.01;
      m.scale.y = 0.01;
      m.colors.resize(cmd.points.size());
    }

    robot_state::RobotState state = planning_scene->getCurrentState();
    auto const start_color = ColorBuilder::MakeFromFloatColors(0, 0, 1, 1);
    auto const end_color = ColorBuilder::MakeFromFloatColors(1, 0, 1, 1);
    for (size_t step_idx = 0; step_idx < cmd.points.size(); ++step_idx)
    {
      state.setJointGroupPositions(jmg_, cmd.points[step_idx].positions);
      state.updateLinkTransforms();
      auto const tool_poses = getToolTransforms(state);
      for (auto tool_idx = 0ul; tool_idx < num_ees_; ++tool_idx)
      {
        auto& m = msg.markers[tool_idx];
        m.points[step_idx] = ConvertTo<gm::Point>(Eigen::Vector3d(tool_poses[tool_idx].translation()));
        auto const ratio = static_cast<float>(step_idx) / static_cast<float>(std::max((cmd.points.size() - 1), 1ul));
        m.colors[step_idx] = arc_helpers::InterpolateColor(start_color, end_color, ratio);
      }
    }
    vis_pub_.publish(msg);
  }

  return cmd;
}

trajectory_msgs::JointTrajectory PlanningInterace::jacobianPath3d(planning_scene::PlanningScenePtr planning_scene,
                                                                std::vector<PointSequence> const& tool_paths)
{
  // Do some preliminary sanity checks
  MPS_ASSERT(planning_scene);
  MPS_ASSERT(num_ees_ >= 1);
  MPS_ASSERT(tool_paths.size() == num_ees_);
  auto const steps = tool_paths[0].size();
  MPS_ASSERT(steps >= 1);
  for (auto const& path : tool_paths)
  {
    MPS_ASSERT(path.size() == steps);
  }

  // Check that the start state of the scene is consisent with the start of the tool paths
  auto const& start_state = planning_scene->getCurrentState();
  auto const start_tool_transforms = getToolTransforms(start_state);
  MPS_ASSERT(start_tool_transforms.size() == num_ees_);
  for (auto idx = 0ul; idx < num_ees_; ++idx)
  {
    MPS_ASSERT(start_tool_transforms[idx].translation().isApprox(tool_paths[idx][0]));
  }

  // Initialize the command with the current state for the first target point
  trajectory_msgs::JointTrajectory cmd;
  cmd.joint_names = jmg_->getActiveJointModelNames();
  cmd.points.resize(steps);
  planning_scene->getCurrentState().copyJointGroupPositions(jmg_, cmd.points[0].positions);
  cmd.points[0].time_from_start = ros::Duration(0);

  // Iteratively follow the Jacobian to each other point in the path
  ROS_INFO("Following Jacobian along path for group %s", jmg_->getName().c_str());
  for (auto step_idx = 1ul; step_idx < steps; ++step_idx)
  {
    // Extract the goal positions and orientations for each tool in robot frame
    PoseSequence robotTtargets(num_ees_);
    for (auto ee_idx = 0ul; ee_idx < num_ees_; ++ee_idx)
    {
      robotTtargets[ee_idx].translation() = robotTworld * tool_paths[ee_idx][step_idx];
      robotTtargets[ee_idx].linear() = robot_nominal_tool_orientations_[ee_idx].toRotationMatrix();
    }

    // Note that jacobianIK is assumed to have modified the state in the planning scene
    const auto iksoln = jacobianIK(planning_scene, robotTtargets);
    if (!iksoln)
    {
      ROS_WARN("IK Stalled at idx %d, returning early", step_idx);
      cmd.points.resize(step_idx);
      cmd.points.shrink_to_fit();
      break;
    }
    planning_scene->getCurrentState().copyJointGroupPositions(jmg_, cmd.points[step_idx].positions);
    cmd.points[step_idx].time_from_start = ros::Duration(static_cast<double>(step_idx));
  }

  ROS_INFO("Jacobian IK path has %d points out of a requested %d", cmd.points.size(), steps);
  return cmd;
}

// Note that robotTtargets is the target points for the tools, measured in robot frame
bool PlanningInterace::jacobianIK(planning_scene::PlanningScenePtr planning_scene, PoseSequence const& robotTtargets)
{
  MPS_ASSERT(planning_scene);
  MPS_ASSERT(robotTtargets.size() == num_ees_);
  constexpr bool bPRINT = false;

  robot_state::RobotState& state = planning_scene->getCurrentStateNonConst();
  Eigen::VectorXd currConfig;
  state.copyJointGroupPositions(jmg_, currConfig);
  Eigen::VectorXd const startConfig = currConfig;
  Eigen::VectorXd prevConfig = currConfig;
  const int ndof = static_cast<int>(startConfig.rows());

  // TODO: correctly handle DOF that have no limits throughout
  //       as well as DOF that are not in R^N, i.e.; continuous revolute, spherical etc.
  // TODO: arm->getVariableLimits()
  Eigen::VectorXd lowerLimit(ndof);
  Eigen::VectorXd upperLimit(ndof);
  {
    auto const& joints = jmg_->getJointModels();
    int nextDofIdx = 0;
    for (auto joint : joints)
    {
      auto const& names = joint->getVariableNames();
      auto const& bounds = joint->getVariableBounds();

      for (size_t var_idx = 0; var_idx < names.size(); ++var_idx)
      {
        if (bounds[var_idx].position_bounded_)
        {
          lowerLimit[nextDofIdx] = bounds[var_idx].min_position_;
          upperLimit[nextDofIdx] = bounds[var_idx].max_position_;
        }
        else
        {
          lowerLimit[nextDofIdx] = std::numeric_limits<double>::lowest() / 4.0;
          upperLimit[nextDofIdx] = std::numeric_limits<double>::max() / 4.0;
        }
        ++nextDofIdx;
      }
    }
    if (bPRINT)
    {
      std::cerr << "lowerLimit:  " << lowerLimit.transpose() << "\n"
                << "upperLimit:  " << upperLimit.transpose() << "\n"
                << "startConfig: " << startConfig.transpose() << "\n\n";
    }
  }

  // Configuration variables (eventually input parameters)
  constexpr auto accuracyThreshold = 0.001;
  constexpr auto movementLimit = std::numeric_limits<double>::max();
  // setting this to true will make the algorithm attempt to move joints
  // that are at joint limits at every iteration back away from the limit
  constexpr bool clearBadJoints = true;
  const auto dampingThreshold = EigenHelpers::SuggestedRcond();
  const auto damping = EigenHelpers::SuggestedRcond();

  collision_detection::CollisionRequest collisionRequest;
  collision_detection::CollisionResult collisionResult;
  VecArrayXb goodJoints = VecArrayXb::Ones(ndof);
  const int maxItr = 200;
  const double maxStepSize = 0.1;
  double stepSize = maxStepSize;
  double prevError = 1000000.0;

  for (int itr = 0; itr < maxItr; itr++)
  {
    auto const robotTcurr = Transform(robotTworld, getToolTransforms(state));
    auto const poseError = calcPoseError(robotTcurr, robotTtargets);
    auto const [posErrorVec, rotErrorVec] = calcVecError(poseError);
    auto currPositionError = posErrorVec.norm();
    auto currRotationError = rotErrorVec.norm();
    if (bPRINT)
    {
      std::cerr << std::endl << "----------- Start of for loop ---------------\n";
      std::cerr << "config = [" << currConfig.transpose() << "]';\n";

      Eigen::MatrixXd matrix_output =
          std::numeric_limits<double>::infinity() * Eigen::MatrixXd::Ones(14, 5 * num_ees_ - 1);
      for (auto idx = 0ul; idx < num_ees_; ++idx)
      {
        matrix_output.block<4, 4>(0, 5 * idx) = robotTcurr[idx].matrix();
      }
      for (auto idx = 0ul; idx < num_ees_; ++idx)
      {
        matrix_output.block<4, 4>(5, 5 * idx) = robotTtargets[idx].matrix();
      }
      for (auto idx = 0ul; idx < num_ees_; ++idx)
      {
        matrix_output.block<4, 4>(10, 5 * idx) = poseError[idx].matrix();
      }
      std::cerr << "         ee idx ----->\n"
                << " current \n"
                << " target  \n"
                << " error   \n";
      std::cerr << "matrix_ouptut = [\n" << matrix_output << "\n];\n";
      std::cerr << "posErrorVec = [" << posErrorVec.transpose() << "]';\n"
                << "rotErrorVec = [" << rotErrorVec.transpose() << "]';\n"
                << "currPositionError: " << currPositionError << std::endl
                << "currRotationError: " << currRotationError << std::endl;
    }
    // Check if we've reached our target
    if (currPositionError < accuracyThreshold)
    {
      if (bPRINT)
      {
        std::cerr << "Projection successful itr: " << itr << ": currPositionError: " << currPositionError << "\n";
      }
      return true;
    }

    // stepSize logic and reporting
    {
      // Take smaller steps if error increases
      // if ((currPositionError >= prevError) || (prevError - currPositionError < accuracyThreshold / 10))
      if (currPositionError >= prevError)
      {
        if (bPRINT)
        {
          std::cerr << "No progress, reducing stepSize: "
                    << "prevError: " << prevError << " currErrror: " << currPositionError << std::endl;
        }
        stepSize = stepSize / 2;
        currPositionError = prevError;
        currConfig = prevConfig;

        // don't let step size get too small
        if (stepSize < accuracyThreshold / 32)
        {
          std::cerr << "stepSize: " << stepSize << std::endl;
          std::cerr << "Projection stalled itr: " << itr << ": stepSize < accuracyThreshold/32\n";
          return false;
        }
      }
      else
      {
        if (bPRINT)
        {
          std::cerr << "Progress, resetting stepSize to max\n";
          std::cerr << "stepSize: " << stepSize << std::endl;
        }
        stepSize = maxStepSize;
      }
    }

    // Check if we've moved too far from the start config
    if (movementLimit != std::numeric_limits<double>::infinity())
    {
      if ((currConfig - startConfig).norm() > movementLimit)
      {
        std::cerr << "Projection hit movement limit at itr " << itr << "\n";
        return false;
      }
    }

    // If we clear bad joint inds, we try to use them again every loop;
    // this makes some sense if we're using the nullspace to servo away from joint limits
    if (clearBadJoints)
    {
      goodJoints = VecArrayXb::Ones(ndof);
    }

    Eigen::MatrixXd const fullJacobian = getJacobianServoFrame(state, robotTcurr);
    Eigen::MatrixXd positionJacobian(3 * num_ees_, ndof);
    Eigen::MatrixXd rotationJacobian(3 * num_ees_, ndof);
    for (auto idx = 0ul; idx < num_ees_; ++idx)
    {
      positionJacobian.block(3 * idx, 0, 3, ndof) = fullJacobian.block(6 * idx, 0, 3, ndof);
      rotationJacobian.block(3 * idx, 0, 3, ndof) = fullJacobian.block(6 * idx + 3, 0, 3, ndof);
    }
    if (bPRINT)
    {
      std::cerr << "fullJacobian = [\n" << fullJacobian << "];\n";
      std::cerr << "positionJacobian = [\n" << positionJacobian << "];\n";
      std::cerr << "rotationJacobian = [\n" << rotationJacobian << "];\n";
    }

    bool newJointAtLimit = false;
    VecArrayXb newBadJoints;
    do
    {
      prevConfig = currConfig;
      prevError = currPositionError;

      // Eliminate bad joint columns from the Jacobian
      ArrayXb const jacobianMask = goodJoints.replicate(1, 3 * num_ees_).transpose();
      Eigen::MatrixXd const partialPositionJacobian = jacobianMask.select(positionJacobian, 0.0);
      Eigen::MatrixXd const partialRotationJacobian = jacobianMask.select(rotationJacobian, 0.0);

      // Converts the position error vector into a unit vector if the step is too large
      const auto positionMagnitude = (currPositionError > stepSize) ? stepSize / currPositionError : 1.0;
      const Eigen::VectorXd positionCorrectionStep = positionMagnitude * EigenHelpers::UnderdeterminedSolver(partialPositionJacobian, posErrorVec, dampingThreshold, damping);
      const Eigen::VectorXd drotTransEffect = rotationJacobian * positionCorrectionStep;

      const Eigen::VectorXd drotEffective = rotErrorVec - drotTransEffect;
      const double effectiveRotationError = drotEffective.norm();
      // Converts the rotation error vector into a unit vector if the step is too large
      const auto rotationMagnitude = (effectiveRotationError > stepSize) ? stepSize / effectiveRotationError : 1.0;
      const Eigen::VectorXd rotationCorrectionStep = rotationMagnitude * EigenHelpers::UnderdeterminedSolver(partialRotationJacobian, drotEffective, dampingThreshold, damping);

      // Build the nullspace constraint matrix:
      // Jpos*q = dpos
      // [0 ... 0 1 0 ... 0]*q = 0 for bad joints
      const int ndof_at_limits = ndof - goodJoints.cast<int>().sum();
      Eigen::MatrixXd nullspaceConstraintMatrix(3 * num_ees_ + ndof_at_limits, ndof);
      nullspaceConstraintMatrix.topRows(3 * num_ees_) = positionJacobian;
      nullspaceConstraintMatrix.bottomRows(ndof_at_limits).setConstant(0);
      int nextMatrixRowIdx = 3 * (int)num_ees_;
      for (int dof_idx = 0; dof_idx < ndof; ++dof_idx)
      {
        if (!goodJoints(dof_idx))
        {
          nullspaceConstraintMatrix(nextMatrixRowIdx, dof_idx) = 1;
          ++nextMatrixRowIdx;
        }
      }
      MPS_ASSERT(nextMatrixRowIdx == nullspaceConstraintMatrix.rows());

      // Project the rotation step into the nullspace of position servoing
      const Eigen::MatrixXd nullspaceConstraintMatrixPinv = EigenHelpers::Pinv(nullspaceConstraintMatrix, EigenHelpers::SuggestedRcond());
      const Eigen::MatrixXd nullspaceProjector = Eigen::MatrixXd::Identity(ndof, ndof) - (nullspaceConstraintMatrixPinv * nullspaceConstraintMatrix);
      const Eigen::VectorXd nullspaceRotationStep = nullspaceProjector * rotationCorrectionStep;
      const Eigen::VectorXd step = positionCorrectionStep + nullspaceRotationStep;
      if (bPRINT)
      {
        std::cerr << "\n\n";
        std::cerr << "fullJacobian                  = [\n" << fullJacobian << "];\n";
        std::cerr << "nullspaceConstraintMatrix     = [\n" << nullspaceConstraintMatrix << "];\n";
        std::cerr << "nullspaceConstraintMatrixPinv = [\n" << nullspaceConstraintMatrixPinv << "];\n";
        std::cerr << "posErrorVec        = [" << posErrorVec.transpose() << "]';\n";
        std::cerr << "rotErrorVec        = [" << rotErrorVec.transpose() << "]';\n";
        std::cerr << "drotTransEffect    = [" << drotTransEffect.transpose() << "]';\n";
        std::cerr << "drotEffective      = [" << drotEffective.transpose() << "]';\n";
        std::cerr << "positionCorrectionStep = [" << positionCorrectionStep.transpose() << "]';\n";
        std::cerr << "rotationCorrectionStep = [" << rotationCorrectionStep.transpose() << "]';\n";
        std::cerr << "nullspaceRotationStep  = [" << nullspaceRotationStep.transpose() << "]';\n";
        std::cerr << "step                   = [" << step.transpose() << "]';\n\n";
      }

      // add step and check for joint limits
      newJointAtLimit = false;
      currConfig += step;
      newBadJoints =
          ((currConfig.array() < lowerLimit.array()) || (currConfig.array() > upperLimit.array())) && goodJoints;
      newJointAtLimit = newBadJoints.any();
      goodJoints = goodJoints && !newBadJoints;

      if (bPRINT)
      {
        std::cerr << "lowerLimit      = [" << lowerLimit.transpose() << "]';\n";
        std::cerr << "upperLimit      = [" << upperLimit.transpose() << "]';\n";
        std::cerr << "currConfig      = [" << currConfig.transpose() << "]';\n";
        std::cerr << "newBadJoints    = [" << newBadJoints.transpose() << "]';\n";
        std::cerr << "goodJoints      = [" << goodJoints.transpose() << "]';\n";
        std::cerr << "newJointAtLimit = [" << newJointAtLimit << "]';\n";
      }

      // move back to previous point if any joint limits
      if (newJointAtLimit)
      {
        currConfig = prevConfig;
      }
    }
    // Exit the loop if we did not reach a new joint limit
    while (newJointAtLimit);

    // Set the robot to the current estimate and update collision info
    state.setJointGroupPositions(jmg_, currConfig);
    state.update();
    planning_scene->checkCollision(collisionRequest, collisionResult, state);
    if (collisionResult.collision)
    {
      ROS_WARN("Projection stalled at itr %d: collision", itr);
      return false;
    }
    collisionResult.clear();
  }

  std::cerr << "Iteration limit reached\n";
  return false;
}

Eigen::MatrixXd PlanningInterace::getJacobianServoFrame(robot_state::RobotState const& state,
                                                      PoseSequence const& robotTservo)
{
  assert(robotTservo.size() == num_ees_);
  const int rows = 6 * (int)num_ees_;
  const int columns = jmg_->getVariableCount();
  auto const& ees = jmg_->getAttachedEndEffectorNames();

  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(rows, columns);
  for (auto idx = 0ul; idx < num_ees_; ++idx)
  {
    auto const ee_baselink = model_->getEndEffector(ees[idx])->getLinkModels().front();
    jacobian.block(idx * 6, 0, 6, columns) = getJacobianServoFrame(state, ee_baselink, robotTservo[idx]);
  }
  return jacobian;
}

// See MLS Page 115-121
// https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-complete.pdf
Matrix6Xd PlanningInterace::getJacobianServoFrame(robot_state::RobotState const& state,
                                                robot_model::LinkModel const* link, Pose const& robotTservo)
{
  const Pose reference_transform = robotTservo.inverse(Eigen::Isometry);
  const robot_model::JointModel* root_joint_model = jmg_->getJointModels()[0];

  const int rows = 6;
  const int columns = jmg_->getVariableCount();
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(rows, columns);

  Eigen::Vector3d joint_axis;
  Pose joint_transform;
  while (link)
  {
    const robot_model::JointModel* pjm = link->getParentJointModel();
    if (pjm->getVariableCount() > 0)
    {
      // TODO: confirm that variables map to unique joint indices
      const unsigned int joint_index = jmg_->getVariableGroupIndex(pjm->getName());
      if (pjm->getType() == robot_model::JointModel::REVOLUTE)
      {
        joint_transform = reference_transform * state.getGlobalLinkTransform(link);
        joint_axis = joint_transform.rotation() * static_cast<const robot_model::RevoluteJointModel*>(pjm)->getAxis();
        jacobian.block<3, 1>(0, joint_index) = -joint_axis.cross(joint_transform.translation());
        jacobian.block<3, 1>(3, joint_index) = joint_axis;
      }
      else if (pjm->getType() == robot_model::JointModel::PRISMATIC)
      {
        joint_transform = reference_transform * state.getGlobalLinkTransform(link);
        joint_axis = joint_transform.rotation() * static_cast<const robot_model::PrismaticJointModel*>(pjm)->getAxis();
        jacobian.block<3, 1>(0, joint_index) = joint_axis;
      }
      else if (pjm->getType() == robot_model::JointModel::PLANAR)
      {
        // This is an SE(2) joint
        joint_transform = reference_transform * state.getGlobalLinkTransform(link);
        joint_axis = joint_transform * Eigen::Vector3d::UnitX();
        jacobian.block<3, 1>(0, joint_index) = joint_axis;
        joint_axis = joint_transform * Eigen::Vector3d::UnitY();
        jacobian.block<3, 1>(0, joint_index + 1) = joint_axis;
        joint_axis = joint_transform * Eigen::Vector3d::UnitZ();
        jacobian.block<3, 1>(0, joint_index + 2) = -joint_axis.cross(joint_transform.translation());
        jacobian.block<3, 1>(3, joint_index + 2) = joint_axis;
      }
      else
      {
        ROS_ERROR("Unknown type of joint in Jacobian computation");
      }
    }
    // NB: this still works because we all joints that are not directly in
    // the kinematic chain have 0s in the Jacobian as they should
    if (pjm == root_joint_model)
    {
      break;
    }
    // NB: this still works because we all joints that are not directly in
    // the kinematic chain have 0s in the Jacobian as they should
    link = pjm->getParentLinkModel();
  }
  return jacobian;
}