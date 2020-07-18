#ifndef LBV_VAL_INTERFACE_HPP
#define LBV_VAL_INTERFACE_HPP

#include "physical_robot_3d_rope_shim/planning_interface.hpp"

class ValInterface : public PlanningInterface
{
public:
  ValInterface(ros::NodeHandle nh, ros::NodeHandle ph, std::shared_ptr<tf2_ros::Buffer> tf_buffer,
               std::string const& group);

  virtual Eigen::VectorXd lookupQHome() override;
  virtual void updateAllowedCollisionMatrix(collision_detection::AllowedCollisionMatrix& acm) override;
};

#endif  // LBV_VAL_INTERFACE_HPP
