//
// Created by dmcconac on 12/11/18.
//

#ifndef MPS_SCENE_H
#define MPS_SCENE_H

#include <moveit/collision_detection/world.h>
#include <moveit/planning_scene/planning_scene.h>
#include <std_srvs/Empty.h>
#include <mutex>

#include "physical_robot_3d_rope_shim/listener.hpp"
#include "physical_robot_3d_rope_shim/moveit_pose_type.hpp"
#include "physical_robot_3d_rope_shim/planning_interface.hpp"

class Scene
{
public:
  static auto constexpr OBSTACLES_NAME = "static_obstacles";

  ros::NodeHandle nh_;
  ros::NodeHandle ph_;
  std::shared_ptr<PlanningInterace> robot_;

  std::recursive_mutex planning_scene_mtx_;
  std::vector<std::pair<std::shared_ptr<shapes::Shape>, Pose>> static_obstacles_;
  ros::ServiceClient get_planning_scene_client_;
  ros::ServiceServer update_planning_scene_server_;
  std::shared_ptr<Listener<sensor_msgs::JointState>> joint_states_listener_;
  planning_scene::PlanningScenePtr planning_scene_;
  ros::Publisher planning_scene_publisher_;

  //////////////////////////////////////////////////////////////////////////////

  Scene(ros::NodeHandle nh, ros::NodeHandle ph, std::shared_ptr<PlanningInterace> robot);

  robot_state::RobotState getCurrentRobotState() const;

  collision_detection::WorldPtr computeCollisionWorld();
  bool UpdatePlanningSceneCallback(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res);
  void updatePlanningScene();
  planning_scene::PlanningScenePtr clonePlanningScene();
};

#endif  // MPS_SCENE_H
