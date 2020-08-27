from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, MoveGroupGoal


def make_moveit_action_goal(joint_names, joint_positions):
    goal_config_constraint = Constraints()
    for name, position in zip(joint_names, joint_positions):
        joint_constraint = JointConstraint()
        joint_constraint.joint_name = name
        joint_constraint.position = position
        goal_config_constraint.joint_constraints.append(joint_constraint)

    req = MotionPlanRequest()
    req.group_name = 'both_arms'
    req.goal_constraints.append(goal_config_constraint)

    goal = MoveGroupGoal()
    goal.request = req
    return goal
