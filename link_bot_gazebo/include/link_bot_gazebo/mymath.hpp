#pragma once

#include <ignition/math.hh>

/** \brief Computes the signed shorted angle between y2 and y1. Check CommonTest.cpp to see examples
 *
 * @param y1 the second angle in the subtraction
 * @param y2 the first angle in the subtraction
 * @return the signed shorted angle between y2 and y1.
 */
constexpr double angle_error(double const y1, double const y2)
{
  double diff = y2 - y1;
  if (diff > M_PI) return diff - M_PI * 2;
  if (diff < -M_PI) return diff + M_PI * 2;
  return diff;
}

ignition::math::Vector3d angle_error(ignition::math::Vector3d const y1, ignition::math::Vector3d const y2)
{
  auto diff = y2 - y1;
  for (auto i = 0; i < 3; ++i) {
    if (diff[i] > M_PI) {
      diff[i] = diff[i] - M_PI * 2;
    }
    else if (diff[i] < -M_PI) {
      diff[i] = diff[i] + M_PI * 2;
    }
  }
  return diff;
}
