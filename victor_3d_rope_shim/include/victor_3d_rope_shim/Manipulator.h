//
// Created by arprice on 9/3/18.
//

#ifndef MPS_MANIPULATOR_H
#define MPS_MANIPULATOR_H

#include <optional>
#include <Eigen/StdVector>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/collision_detection/world.h>
#include <moveit/planning_scene/planning_scene.h>
#include <trajectory_msgs/JointTrajectory.h>

#include "victor_3d_rope_shim/moveit_pose_type.h"

using Pose = moveit::Pose;
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

	void setAcmThisArmOnly(planning_scene::PlanningScenePtr const ps);

	double stateCost(const std::vector<double>& q1) const;

	double transitionCost(
		const std::vector<double>& q1, const double t1,
		const std::vector<double>& q2, const double t2) const;

	bool interpolate(
		const Eigen::Vector3d& from,
		const Eigen::Vector3d& to,
		PointSequence& sequence,
		const int INTERPOLATE_STEPS = 15);

	bool interpolate(
		const Pose& from,
		const Pose& to,
		PoseSequence& sequence,
		const int INTERPOLATE_STEPS = 15) const;

	bool interpolate(
		const robot_state::RobotState& currentState,
		const robot_state::RobotState& toState,
		planning_scene::PlanningSceneConstPtr scene,
		trajectory_msgs::JointTrajectory& cmd,
		const int INTERPOLATE_STEPS = 15) const;

	std::vector<std::vector<double>> IK(
		const Pose& worldGoalPose,
		const Pose& robotTworld,
		const robot_state::RobotState& currentState,
		planning_scene::PlanningSceneConstPtr scene) const;

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

	// Returns a config, and a distance that was moved in radians from the start config
	std::pair<Eigen::VectorXd, double> servoNullSpace(
		const robot_state::RobotState& currentState,
		const Eigen::Vector3d& flangeReferencePoint,
		const Pose& referencePose,
		planning_scene::PlanningSceneConstPtr scene,
		const double displacement) const;

	bool cartesianPath(
		const PoseSequence& worldGoalPoses,
		const Pose& robotTworld,
		const robot_state::RobotState& currentState,
		planning_scene::PlanningSceneConstPtr scene,
		trajectory_msgs::JointTrajectory& cmd) const;

	// Attempts to move the servo frame to each goal point in turn, while
	//     maintaining the nominal tool orientation as best as possible
	bool jacobianPath3D(
		const PointSequence& worldGoalPoints,
		const Eigen::Matrix3d& worldNominalToolOrientation,
		const Pose& robotTworld,
		const Pose& flangeTservo,
		const robot_state::RobotState& currentState,
		planning_scene::PlanningSceneConstPtr scene,
		trajectory_msgs::JointTrajectory& cmd) const;

	bool nullspacePath(
		const robot_state::RobotState& startingState,
		const Eigen::Vector3d& flangeReferencePoint,
		planning_scene::PlanningSceneConstPtr scene,
		const double displacement,
		trajectory_msgs::JointTrajectory& cmd,
		const int INTERPOLATE_STEPS = 15) const;

	virtual
	std::vector<double> getGripperOpenJoints() const { return {}; }

protected:
	std::vector<double> qMin;
	std::vector<double> qMax;
	std::vector<double> qMid;
};

#endif // MPS_MANIPULATOR_H
