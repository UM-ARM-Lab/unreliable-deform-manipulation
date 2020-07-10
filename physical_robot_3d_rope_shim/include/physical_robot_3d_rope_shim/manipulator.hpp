//
// Created by arprice on 9/3/18.
//

#ifndef MPS_MANIPULATOR_HPP
#define MPS_MANIPULATOR_HPP

#include <optional>
#include <Eigen/StdVector>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/planning_scene/planning_scene.h>
#include <trajectory_msgs/JointTrajectory.h>

#include "physical_robot_3d_rope_shim/moveit_pose_type.hpp"

using Matrix6Xd = Eigen::Matrix<double, 6, Eigen::Dynamic>;

class Manipulator
{
public:
	using PoseSequence = std::vector<Pose, Eigen::aligned_allocator<Pose>>;
	using PointSequence = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;

	robot_model::RobotModelPtr pModel;
	robot_model::JointModelGroup* arm;
	robot_model::JointModelGroup* gripper;
	std::string palmName;
	std::vector<std::string> linkNames;

	Manipulator(robot_model::RobotModelPtr _pModel,
	            robot_model::JointModelGroup* _arm,
	            robot_model::JointModelGroup* _gripper,
	            std::string _palmName);

	virtual
	bool configureHardware() { return true; }

	std::optional<PointSequence> interpolate(
		const Eigen::Vector3d& from,
		const Eigen::Vector3d& to,
		const int INTERPOLATE_STEPS = 15);

	std::optional<PoseSequence> interpolate(
		const Pose& from,
		const Pose& to,
		const int INTERPOLATE_STEPS = 15) const;

	std::optional<trajectory_msgs::JointTrajector> interpolate(
		const robot_state::RobotState& currentState,
		const robot_state::RobotState& toState,
		planning_scene::PlanningSceneConstPtr scene,
		const int INTERPOLATE_STEPS = 15) const;

	Matrix6Xd getJacobianServoFrame(
		const robot_model::RobotState& state,
		const robot_model::LinkModel* link,
		const Pose& robotTservo) const;

	Matrix6Xd getJacobianRobotFrame(
		const robot_model::RobotState& state,
		const robot_model::LinkModel* link,
		const Pose& robotTservo) const;

	// Performs Jacobian based IK to reach the goal position in robotTgoal,
	//     while attempting to reach the orientation in robotTgoal with any extra
	//     movement available in the nullspace/joint limits of position servoing
	std::optional<Eigen::VectorXd> jacobianIK(
		const Pose& robotTgoal,
		const Pose& flangeTservo,
		const robot_state::RobotState& currentState,
		planning_scene::PlanningSceneConstPtr scene) const;

	// Attempts to move the servo frame to each goal point in turn, while
	//     maintaining the nominal tool orientation as best as possible
	std::optional<trajectory_msgs::JointTrajectory> jacobianPath3D(
		const PointSequence& worldGoalPoints,
		const Eigen::Matrix3d& worldNominalToolOrientation,
		const Pose& robotTworld,
		const Pose& flangeTservo,
		const robot_state::RobotState& currentState,
		planning_scene::PlanningSceneConstPtr scene) const;
};

#endif // MPS_MANIPULATOR_HPP
