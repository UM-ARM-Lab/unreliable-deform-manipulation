#pragma once

#include <ignition/math.hh>
#include <link_bot_gazebo/mymath.hpp>

// FIXME: move these to arc_utilities

ignition::math::Vector3d angle_error(ignition::math::Vector3d const y1, ignition::math::Vector3d const y2)
{
  auto diff = y2 - y1;
  for (auto i = 0; i < 3; ++i)
  {
    if (diff[i] > M_PI)
    {
      diff[i] = diff[i] - M_PI * 2;
    } else if (diff[i] < -M_PI)
    {
      diff[i] = diff[i] + M_PI * 2;
    }
  }
  return diff;
}
