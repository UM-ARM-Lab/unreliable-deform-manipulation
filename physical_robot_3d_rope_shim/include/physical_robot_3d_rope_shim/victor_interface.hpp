#ifndef LBV_VICTOR_INTERFACE_HPP
#define LBV_VICTOR_INTERFACE_HPP

#include "physical_robot_3d_rope_shim/planning_interface.hpp"

class VictorInterface : public PlanningInterface
{
public:
  VictorInterface(ros::NodeHandle nh, ros::NodeHandle ph, std::shared_ptr<tf2_ros::Buffer> tf_buffer,
                  std::string const& group);

  virtual Eigen::VectorXd lookupQHome() override;
  virtual void updateAllowedCollisionMatrix(collision_detection::AllowedCollisionMatrix& acm) override;
};

#endif  // LBV_VICTOR_INTERFACE_HPP
