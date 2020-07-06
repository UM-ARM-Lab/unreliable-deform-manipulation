#include <ros/ros.h>
#include "victor_3d_rope_shim/victor_shim.hpp"

int main(int argc, char* argv[])
{
  // Read in all ROS parameters
  ros::init(argc, argv, "world_interaction");

  ros::NodeHandle nh;
  ros::NodeHandle ph("~");
  ros::AsyncSpinner spinner(1);
  spinner.start();

  VictorShim vs(nh, ph);
  vs.victor_->updatePlanningScene();
  // vs.victor_->test();
  // vs.victor_->gotoHome();
  vs.enableServices();

  ros::Rate rate(1);
  while (ros::ok())
  {
    vs.victor_->updatePlanningScene();
    rate.sleep();
  }

  return EXIT_SUCCESS;
}
