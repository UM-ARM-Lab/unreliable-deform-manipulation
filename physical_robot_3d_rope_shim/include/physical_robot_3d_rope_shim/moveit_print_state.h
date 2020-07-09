#ifndef MPS_PRINT_STATE_H

#include "victor_3d_rope_shim/moveit_pose_type.h"

#if MOVEIT_VERSION_AT_LEAST(1, 0, 1)
    #define PRINT_STATE_POSITIONS_WITH_JOINT_LIMITS(state, jmg, out) state.printStatePositionsWithJointLimits(jmg, out)
#else
    #include <moveit/macros/console_colors.h>
    #include <moveit/robot_state/robot_state.h>
    #define PRINT_STATE_POSITIONS_WITH_JOINT_LIMITS(state, jmg, out) printStatePositionsWithJointLimits(state, jmg, out)

    // Taken from http://docs.ros.org/melodic/api/moveit_core/html/robot__state_8cpp_source.html#l02189
    inline void printStatePositionsWithJointLimits(const robot_state::RobotState& state, const moveit::core::JointModelGroup* jmg, std::ostream& out)
    {
        // TODO(davetcoleman): support joints with multiple variables / multiple DOFs such as floating joints
        // TODO(davetcoleman): support unbounded joints

        const std::vector<const moveit::core::JointModel*>& joints = jmg->getActiveJointModels();

        // Loop through joints
        for (std::size_t i = 0; i < joints.size(); ++i)
        {
            // Ignore joints with more than one variable
            if (joints[i]->getVariableCount() > 1)
                continue;

            double current_value = state.getVariablePosition(joints[i]->getName());

            // check if joint is beyond limits
            bool out_of_bounds = !state.satisfiesBounds(joints[i]);

            const moveit::core::VariableBounds& bound = joints[i]->getVariableBounds()[0];

            if (out_of_bounds)
                out << MOVEIT_CONSOLE_COLOR_RED;

            out << "   " << std::fixed << std::setprecision(5) << bound.min_position_ << "\t";
            double delta = bound.max_position_ - bound.min_position_;
            double step = delta / 20.0;

            bool marker_shown = false;
            for (double value = bound.min_position_; value < bound.max_position_; value += step)
            {
                // show marker of current value
                if (!marker_shown && current_value < value)
                {
                    out << "|";
                    marker_shown = true;
                }
                else
                    out << "-";
            }
            if (!marker_shown)
                out << "|";

            // show max position
            out << " \t" << std::fixed << std::setprecision(5) << bound.max_position_ << "  \t" << joints[i]->getName()
                << " current: " << std::fixed << std::setprecision(5) << current_value << std::endl;

            if (out_of_bounds)
                out << MOVEIT_CONSOLE_COLOR_RESET;
        }
    }
#endif

#define MPS_PRINT_STATE_H
#endif // MPS_PRINT_STATE_H