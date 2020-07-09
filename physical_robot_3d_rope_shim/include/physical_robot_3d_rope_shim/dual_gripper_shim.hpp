#ifndef DUAL_GRIPPER_SHIM_HPP
#define DUAL_GRIPPER_SHIM_HPP

#include <memory>

#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <peter_msgs/DualGripperTrajectory.h>

#include "victor_3d_rope_shim/Listener.hpp"
#include "victor_3d_rope_shim/Scene.h"

template <class RobotInterface>
class DualGripperShim
{
public:
  ros::NodeHandle nh_;
  ros::NodeHandle ph_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<typename RobotInterface> robot_;
  ros::ServiceServer execute_traj_srv_;

  // Extra static objects only
  std::shared_ptr<Scene> scene_;

  std::mutex planning_scene_mtx_;
  ros::ServiceClient get_planning_scene_client_;
  ros::ServiceServer update_planning_scene_server_;
  planning_scene::PlanningScenePtr planning_scene_;
  ros::Publisher planning_scene_publisher_;

  DualGripperShim(ros::NodeHandle nh, ros::NodeHandle ph);

  // Control/exection
  void enableServices();
  bool executeTrajectory(peter_msgs::DualGripperTrajectory::Request& req,
                         peter_msgs::DualGripperTrajectory::Response& res);

  bool UpdatePlanningSceneCallback(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res);
  void updatePlanningScene();
};

#endif
