//
// Created by arprice on 9/3/18.
//

#include "physical_robot_3d_rope_shim/manipulator.hpp"
#include "physical_robot_3d_rope_shim/assert.h"
#include "eigen_std_conversions.hpp"

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <arc_utilities/pretty_print.hpp>
#include <memory>

using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>;
using VecArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;

template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

std::vector<std::string> getBodyNames(planning_scene::PlanningSceneConstPtr const& ps)
{
  auto bodies = ps->getRobotModel()->getLinkModelNames();
  auto const obstacles = ps->getWorld()->getObjectIds();
  bodies.insert(bodies.end(), obstacles.begin(), obstacles.end());
  return bodies;
}

Manipulator::Manipulator(robot_model::RobotModelPtr _pModel, robot_model::JointModelGroup* _arm,
                         robot_model::JointModelGroup* _gripper, std::string _palmName)
  : pModel(std::move(_pModel)), arm(_arm), gripper(_gripper), palmName(std::move(_palmName))
{
  MPS_ASSERT(arm->getVariableCount() == arm->getActiveJointModels().size());
  qMin.resize(arm->getVariableCount());
  qMax.resize(arm->getVariableCount());
  qMid.resize(arm->getVariableCount());
  auto const& bounds = arm->getActiveJointModelsBounds();

  for (size_t j = 0; j < bounds.size(); ++j)
  {
    MPS_ASSERT(bounds[j]->size() == 1);
    const auto& b = bounds[j]->front();

    qMin[j] = b.min_position_;
    qMax[j] = b.max_position_;
    qMid[j] = (qMin[j] + qMax[j]) / 2.0;
  }

  linkNames.insert(linkNames.end(), arm->getLinkModelNames().begin(), arm->getLinkModelNames().end());
  linkNames.insert(linkNames.end(), gripper->getLinkModelNames().begin(), gripper->getLinkModelNames().end());
}

bool Manipulator::interpolate(const Eigen::Vector3d& from, const Eigen::Vector3d& to, PointSequence& sequence,
                              const int INTERPOLATE_STEPS)
{
  MPS_ASSERT(INTERPOLATE_STEPS > 1);
  sequence.resize(INTERPOLATE_STEPS);
  for (int i = 0; i < INTERPOLATE_STEPS; ++i)
  {
    const double t = i / static_cast<double>(INTERPOLATE_STEPS - 1);
    sequence[i] = ((1.0 - t) * from) + (t * to);
  }
  return true;
}

bool Manipulator::interpolate(const Pose& from, const Pose& to, PoseSequence& sequence,
                              const int INTERPOLATE_STEPS) const
{
  MPS_ASSERT(INTERPOLATE_STEPS > 1);
  sequence.resize(INTERPOLATE_STEPS);
  const Eigen::Quaterniond qStart(from.linear());
  const Eigen::Quaterniond qEnd(to.linear());
  for (int i = 0; i < INTERPOLATE_STEPS; ++i)
  {
    const double t = i / static_cast<double>(INTERPOLATE_STEPS - 1);
    sequence[i].translation() = ((1.0 - t) * from.translation()) + (t * to.translation());
    sequence[i].linear() = qStart.slerp(t, qEnd).matrix();
  }
  return true;
}

bool Manipulator::interpolate(const robot_state::RobotState& currentState, const robot_state::RobotState& toState,
                              planning_scene::PlanningSceneConstPtr scene, trajectory_msgs::JointTrajectory& cmd,
                              const int INTERPOLATE_STEPS) const
{
  MPS_ASSERT(INTERPOLATE_STEPS > 1);
  if (!scene)
  {
    scene = std::make_shared<planning_scene::PlanningScene>(pModel);
  }

  collision_detection::CollisionRequest collision_request;
  collision_detection::CollisionResult collision_result;

  scene->checkCollision(collision_request, collision_result, currentState);
  if (collision_result.collision)
  {
    std::cerr << "currentState in collision\n\n";
    return false;
  }

  collision_result.clear();
  scene->checkCollision(collision_request, collision_result, toState);
  if (collision_result.collision)
  {
    std::cerr << "toState in collision\n\n";
    return false;
  }

  cmd.joint_names = arm->getActiveJointModelNames();
  cmd.points.resize(cmd.points.size() + INTERPOLATE_STEPS);

  robot_state::RobotState interpState(currentState);
  interpState.setToDefaultValues();
  // Verify the "plan" is collision-free
  for (int i = 0; i < INTERPOLATE_STEPS; ++i)
  {
    const double t = i / static_cast<double>(INTERPOLATE_STEPS - 1);
    currentState.interpolate(toState, t, interpState, arm);
    interpState.update();

    collision_result.clear();
    scene->checkCollision(collision_request, collision_result, interpState);
    if (collision_result.collision)
    {
      std::cerr << "state " << i << " in collision\n\n";
      return false;
    }

    interpState.copyJointGroupPositions(arm, cmd.points[i].positions);
    cmd.points[i].time_from_start = ros::Duration(t * 3.0);
  }

  return true;
}

std::vector<std::vector<double>> Manipulator::IK(const Pose& worldGoalPose, const Pose& robotTworld,
                                                 const robot_state::RobotState& currentState,
                                                 planning_scene::PlanningSceneConstPtr scene) const
{
  if (!scene)
  {
    scene = std::make_shared<planning_scene::PlanningScene const>(pModel);
  }

  const kinematics::KinematicsBaseConstPtr& solver = arm->getSolverInstance();
  MPS_ASSERT(solver.get());

  // NB: The (possibly dirty) frames in RobotState are not marked mutable, hence the const casting.
  Pose solverbaseTrobot = Pose::Identity();
  const_cast<robot_state::RobotState&>(currentState).updateLinkTransforms();
  const_cast<robot_state::RobotState&>(currentState).setToIKSolverFrame(solverbaseTrobot, solver);

  const Pose solvertipTgoal = currentState.getFrameTransform(solver->getTipFrame()).inverse(Eigen::Isometry) *
                              currentState.getFrameTransform(this->palmName);

  // Convert to solver frame
  const Pose pt_solver = solverbaseTrobot * robotTworld * worldGoalPose * solvertipTgoal.inverse(Eigen::Isometry);

  const Eigen::Quaterniond q(pt_solver.linear());
  geometry_msgs::Pose pose;
  pose.position.x = pt_solver.translation().x();
  pose.position.y = pt_solver.translation().y();
  pose.position.z = pt_solver.translation().z();
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  const std::vector<geometry_msgs::Pose> targetPoses = { pose };

  std::vector<double> seed(arm->getVariableCount(), 0.0);
  currentState.copyJointGroupPositions(arm, seed);
  kinematics::KinematicsResult result;  // NOLINT(cppcoreguidelines-pro-type-member-init)
  kinematics::KinematicsQueryOptions options;
  options.discretization_method = kinematics::DiscretizationMethod::ALL_DISCRETIZED;
  std::vector<std::vector<double>> solutions;
  solver->getPositionIK(targetPoses, seed, solutions, result, options);

  // Filter out any solutions that are in collision
  collision_detection::CollisionRequest collisionRequest;
  collision_detection::CollisionResult collisionResult;
  std::vector<std::vector<double>> validSolutions;
  validSolutions.reserve(solutions.size());
  for (size_t idx = 0; idx < solutions.size(); ++idx)
  {
    // TODO: ensure that this is taking the current robot configuration into account correctly
    robot_state::RobotState collisionState(currentState);
    collisionState.setJointGroupPositions(arm, solutions[idx]);
    collisionState.update();
    scene->checkCollision(collisionRequest, collisionResult, collisionState);
    if (!collisionResult.collision)
    {
      validSolutions.push_back(solutions[idx]);
    }
    collisionResult.clear();
  }
  validSolutions.shrink_to_fit();
  return validSolutions;
}

// See MLS Page 115-121
// https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-complete.pdf
Matrix6Xd Manipulator::getJacobianServoFrame(const robot_model::RobotState& state,
                                             const robot_model::LinkModel* link,
                                             const Pose& robotTservo) const
{
  const Pose reference_transform = robotTservo.inverse(Eigen::Isometry);
  const robot_model::JointModel* root_joint_model = arm->getJointModels()[0];

  const int rows = 6;
  const int columns = arm->getVariableCount();
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(rows, columns);

  Eigen::Vector3d joint_axis;
  Pose joint_transform;
  while (link)
  {
    const robot_model::JointModel* pjm = link->getParentJointModel();
    if (pjm->getVariableCount() > 0)
    {
      // TODO: confirm that variables map to unique joint indices
      const unsigned int joint_index = arm->getVariableGroupIndex(pjm->getName());
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
    if (pjm == root_joint_model)
    {
      break;
    }
    link = pjm->getParentLinkModel();
  }
  return jacobian;
}

Matrix6Xd Manipulator::getJacobianRobotFrame(const robot_model::RobotState& state, const robot_model::LinkModel* link,
                                             const Pose& robotTservo) const
{
  const auto adjoint = EigenHelpers::AdjointFromTransform(robotTservo);
  const auto jacobian = getJacobianServoFrame(state, link, robotTservo);
  return adjoint * jacobian;
}

// Implementation based on comps::GeneralIK
// https://github.com/UM-ARM-Lab/comps/blob/master/generalik/GeneralIK.cpp
// TODO: handle servoing away from joint limits
std::optional<Eigen::VectorXd> Manipulator::jacobianIK(const Pose& robotTgoal, const Pose& flangeTservo,
                                                       const robot_state::RobotState& currentState,
                                                       planning_scene::PlanningSceneConstPtr scene) const
{
  constexpr bool bPRINT = false;

  if (!scene)
  {
    scene = std::make_shared<planning_scene::PlanningScene const>(pModel);
  }

  robot_state::RobotState state = currentState;
  Eigen::VectorXd currConfig;
  state.copyJointGroupPositions(arm, currConfig);
  Eigen::VectorXd const startConfig = currConfig;
  Eigen::VectorXd prevConfig = currConfig;
  const int ndof = static_cast<int>(startConfig.rows());

  // TODO: correctly handle DOF that have no limits throughout
  //       as well as DOF that are not in R^N, i.e.; continuous revolute, spherical etc.
  // TODO: arm->getVariableLimits()
  Eigen::VectorXd lowerLimit(ndof);
  Eigen::VectorXd upperLimit(ndof);
  {
    auto const& joints = arm->getJointModels();
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
  double currPositionError;
  double currRotationError;
  Pose robotTflange = state.getGlobalLinkTransform(arm->getLinkModels().back());
  for (int itr = 0; itr < maxItr; itr++)
  {
    // Set the robot to the current estimate and update collision info
    state.setJointGroupPositions(arm, currConfig);
    state.update();
    scene->checkCollision(collisionRequest, collisionResult, state);
    if (collisionResult.collision)
    {
      ROS_WARN_STREAM("Projection stalled at itr: " << itr << " for arm " << palmName << ": collision");
      return std::nullopt;
    }
    collisionResult.clear();

    // Update error information
    robotTflange = state.getGlobalLinkTransform(arm->getLinkModels().back());
    const Pose robotTcurr = robotTflange * flangeTservo;
    const Pose poseError = robotTcurr.inverse(Eigen::Isometry) * robotTgoal;
    // TODO: replace twistError with matrix log + unhat of just the relative rotation matrix?
    const auto twistError = EigenHelpers::TwistBetweenTransforms(robotTcurr, robotTgoal);
    const Eigen::Vector3d dtrans = poseError.translation();
    const Eigen::Vector3d drot = twistError.tail<3>();
    currPositionError = dtrans.norm();
    currRotationError = drot.norm();
    if (bPRINT)
    {
      std::cerr << std::endl << "----------- Start of for loop ---------------\n";
      std::cerr << "config = [" << currConfig.transpose() << "]';\n"
                << "target = [\n"
                << robotTgoal.matrix() << "\n];\n"
                << "current = [\n"
                << robotTcurr.matrix() << "];\n"
                << "poseError = [\n"
                << poseError.matrix() << "];\n"
                << "twistError = [" << twistError.transpose() << "]';\n"
                << "dtrans = [" << dtrans.transpose() << "]';\n"
                << "drot   = [" << drot.transpose() << "]';\n\n"
                << "currPositionError: " << currPositionError << std::endl
                << "currRotationError: " << currRotationError << std::endl;
    }
    if (currPositionError < accuracyThreshold)
    {
      if (bPRINT)
      {
        std::cerr << "Projection successful itr: " << itr << ": currPositionError: " << currPositionError << "\n";
      }
      return currConfig;
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
        if (stepSize < accuracyThreshold / 1024)
        {
          std::cerr << "stepSize: " << stepSize << std::endl;
          std::cerr << "Projection stalled itr: " << itr << ": stepSize < accuracyThreshold/32\n";
          return std::nullopt;
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
        return std::nullopt;
      }
    }

    // If we clear bad joint inds, we try to use them again every loop;
    // this makes some sense if we're using the nullspace to servo away from joint limits
    if (clearBadJoints)
    {
      goodJoints = VecArrayXb::Ones(ndof);
    }

    // NB: The jacobian is with respect to the last link in the group (victor_left_arm_flange)
    // const Eigen::MatrixXd fullJacobian = state.getJacobian(arm, flangeReferencePoint);
    const Eigen::MatrixXd fullJacobian = getJacobianServoFrame(state, arm->getLinkModels().back(), robotTcurr);
    // getJacobian(state, arm, arm->getLinkModels().back(), flangeReferencePoint);
    bool newJointAtLimit = false;
    VecArrayXb newBadJoints;
    do
    {
      prevConfig = currConfig;
      prevError = currPositionError;

      // Eliminate bad joint columns from the Jacobian
      const ArrayXb jacobianMask = goodJoints.replicate(1, 3).transpose();
      const Eigen::MatrixXd partialPositionJacobian = jacobianMask.select(fullJacobian.topRows<3>(), 0.0);
      const Eigen::MatrixXd partialRotationJacobian = jacobianMask.select(fullJacobian.bottomRows<3>(), 0.0);

      // Converts the position error vector into a unit vector if the step is too large
      const auto positionMagnitude = (currPositionError > stepSize) ? stepSize / currPositionError : 1.0;
      const Eigen::VectorXd positionCorrectionStep =
          positionMagnitude *
          EigenHelpers::UnderdeterminedSolver(partialPositionJacobian, dtrans, dampingThreshold, damping);
      const Eigen::VectorXd drotTransEffect = fullJacobian.bottomRows<3>() * positionCorrectionStep;

      const Eigen::Vector3d drotEffective = drot - drotTransEffect;
      const double effectiveRotationError = drotEffective.norm();
      // Converts the rotation error vector into a unit vector if the step is too large
      const auto rotationMagnitude = (effectiveRotationError > stepSize) ? stepSize / effectiveRotationError : 1.0;
      const Eigen::VectorXd rotationCorrectionStep =
          rotationMagnitude *
          EigenHelpers::UnderdeterminedSolver(partialRotationJacobian, drotEffective, dampingThreshold, damping);

      // Build the nullspace constraint matrix:
      // Jpos*q = dpos
      // [0 ... 0 1 0 ... 0]*q = 0 for bad joints
      const int ndof_at_limits = ndof - goodJoints.cast<int>().sum();
      Eigen::MatrixXd nullspaceConstraintMatrix(3 + ndof_at_limits, ndof);
      nullspaceConstraintMatrix.topRows<3>() = fullJacobian.topRows<3>();
      nullspaceConstraintMatrix.bottomRows(ndof_at_limits).setConstant(0);
      int nextMatrixRowIdx = 3;
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
      const Eigen::MatrixXd nullspaceConstraintMatrixPinv =
          EigenHelpers::Pinv(nullspaceConstraintMatrix, EigenHelpers::SuggestedRcond());
      const Eigen::MatrixXd nullspaceProjector =
          Eigen::MatrixXd::Identity(ndof, ndof) - (nullspaceConstraintMatrixPinv * nullspaceConstraintMatrix);
      const Eigen::VectorXd nullspaceRotationStep = nullspaceProjector * rotationCorrectionStep;
      const Eigen::VectorXd step = positionCorrectionStep + nullspaceRotationStep;
      if (bPRINT)
      {
        std::cerr << "\nfullJacobian                = [\n" << fullJacobian << "];\n";
        std::cerr << "nullspaceConstraintMatrix     = [\n" << nullspaceConstraintMatrix << "];\n";
        std::cerr << "nullspaceConstraintMatrixPinv = [\n" << nullspaceConstraintMatrixPinv << "];\n";
        std::cerr << "dtrans             = [" << dtrans.transpose() << "]';\n";
        std::cerr << "drot               = [" << drot.transpose() << "]';\n";
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
  }

  std::cerr << "Iteration limit reached\n";
  return std::nullopt;
}

bool Manipulator::jacobianPath3D(
  const PointSequence& worldGoalPoints,
  const Eigen::Matrix3d& worldNominalToolOrientation,
  const Pose& robotTworld,
  const Pose& flangeTservo,
  const robot_state::RobotState& currentState,
  planning_scene::PlanningSceneConstPtr scene,
  trajectory_msgs::JointTrajectory& cmd) const
{
  MPS_ASSERT(worldGoalPoints.size() > 1);
  if (!scene)
  {
    scene = std::make_shared<planning_scene::PlanningScene const>(pModel);
  }

  robot_state::RobotState state = currentState;
  state.update();

  const Eigen::Quaterniond robotNominalToolOrientation(
      (robotTworld * Eigen::Quaterniond(worldNominalToolOrientation)).rotation());

  // Initialize the command with the current state for the first target point
  cmd.joint_names = arm->getActiveJointModelNames();
  cmd.points.resize(worldGoalPoints.size());
  state.copyJointGroupPositions(arm, cmd.points[0].positions);
  cmd.points[0].time_from_start = ros::Duration(0);

  // Iteratively follow the Jacobian to each other point in the path
  ROS_INFO_STREAM("Following Jacobian along path for manipulator " << palmName);
  for (size_t idx = 1; idx < worldGoalPoints.size(); ++idx)
  {
    const Pose robotTgoal = Eigen::Translation3d(robotTworld * worldGoalPoints[idx]) * robotNominalToolOrientation;
    const auto ikSoln = jacobianIK(robotTgoal, flangeTservo, state, scene);
    if (!ikSoln)
    {
      ROS_WARN_STREAM("IK Stalled at idx " << idx << " for " << palmName << "; returning early");
      cmd.points.resize(idx);
      cmd.points.shrink_to_fit();
      break;
    }
    cmd.points[idx].positions = ConvertTo<std::vector<double>>(*ikSoln);
    cmd.points[idx].time_from_start = ros::Duration(static_cast<double>(idx));
    state.setJointGroupPositions(arm, *ikSoln);
    state.update();
  }

  ROS_INFO_STREAM("Jacobian IK path has " << cmd.points.size() << " points");
  return true;
}
