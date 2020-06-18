#ifndef LBV_EIGEN_TRANSFORMS_HPP
#define LBV_EIGEN_TRANSFORMS_HPP

#include <arc_utilities/eigen_typedefs.hpp>
#include "victor_3d_rope_shim/moveit_pose_type.h"

template <typename T1, typename T2>
std::pair<T1, T2> Transform(moveit::Pose const& transform, std::pair<T1, T2> const& input)
{
  return { transform * input.first, transform * input.second };
}

std::pair<moveit::Pose, moveit::Pose> Transform(moveit::Pose const& transform,
                                                std::pair<Eigen::Translation3d, Eigen::Translation3d> const& input)
{
  return { transform * input.first, transform * input.second };
}

#endif