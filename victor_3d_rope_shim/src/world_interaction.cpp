#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <arc_utilities/geometry_msgs_builders.hpp>
#include "victor_3d_rope_shim/assert.h"
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
    vs.victor_->visualizePlanningScene();
    vs.victor_->test();
    vs.victor_->gotoHome();

    // Test moving the hands to a specific location relative to the table
    // peter_msgs::DualGripperTrajectory::Request req;
    // peter_msgs::DualGripperTrajectory::Response res;
    // req.gripper1_points = {arc_utilities::rmb::MakePoint(1.1,  0.4, 1.05)};
    // req.gripper2_points = {arc_utilities::rmb::MakePoint(1.1, -0.4, 1.05)};
    // vs.executeTrajectory(req, res);

    ROS_INFO("Publishing planning scene");
    ros::Rate r(2);
    while (ros::ok())
    {
        // ROS_INFO_THROTTLE(10, "Publishing planning scene");
        vs.victor_->visualizePlanningScene();
        r.sleep();
    }

    return EXIT_SUCCESS;
}
