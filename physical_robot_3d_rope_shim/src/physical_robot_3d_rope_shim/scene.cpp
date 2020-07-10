//
// Created by arprice on 12/11/18.
//

#include "physical_robot_3d_rope_shim/moveit_pose_type.hpp"
#include "physical_robot_3d_rope_shim/scene.hpp"

#include <moveit/robot_state/conversions.h>
#include <moveit_msgs/GetPlanningScene.h>
#include <arc_utilities/ros_helpers.hpp>

Scene::Scene(ros::NodeHandle nh, ros::NodeHandle ph, std::shared_ptr<PlanningInterace> robot)
  : nh_(nh)
  , ph_(ph)
  , robot_(robot)
  , joint_states_listener_(std::make_shared<Listener<sensor_msgs::JointState>>(nh_, "joint_states", true))
  , planning_scene_publisher_(nh.advertise<moveit_msgs::PlanningScene>("planning_scene", 1, true))
{
  // Retrieve the planning scene obstacles if possible, otherwise default to a saved set
  {
    auto const topic = ROSHelpers::GetParam<std::string>(ph_, "get_planning_scene_topic", "get_planning_scene");
    get_planning_scene_client_ = nh_.serviceClient<moveit_msgs::GetPlanningScene>(topic);
    if (get_planning_scene_client_.waitForExistence(ros::Duration(3)))
    {
      planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_->model_, computeCollisionWorld());
      updatePlanningScene();
    }
    else
    {
      ROS_WARN_STREAM("Service [" << nh_.getNamespace() << topic << " was not available. Defaulting to a saved set");

      // NB: PlannningScene assumes everything is defined relative to the
      //     robot base frame, so we have to deal with that here
      auto constexpr wall_width = 0.1;
      // Protect Andrew's monitors
      {
        auto const wall = std::make_shared<shapes::Box>(5, wall_width, 3);
        Pose const pose(Eigen::Translation3d(2.8 + wall_width / 2, 1.1, 1.5));
        static_obstacles_.push_back({ wall, robot_->robotTworld * pose });
      }
      // Protect Dale's monitors
      {
        auto const wall = std::make_shared<shapes::Box>(5, wall_width, 3);
        Pose const pose(Eigen::Translation3d(2.8 + wall_width / 2, -1.1, 1.5));
        static_obstacles_.push_back({ wall, robot_->robotTworld * pose });
      }
      // Protect stuff behind Victor
      {
        auto const wall = std::make_shared<shapes::Box>(0.1, 2.2 + wall_width, 3);
        Pose const pose(Eigen::Translation3d(0.3, 0.0, 1.5));
        static_obstacles_.push_back({ wall, robot_->robotTworld * pose });
      }

      planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_->model_, computeCollisionWorld());
    }
  }
}

collision_detection::WorldPtr Scene::computeCollisionWorld()
{
  auto const world = std::make_shared<collision_detection::World>();
  for (const auto& obstacle : static_obstacles_)
  {
    world->addToObject(OBSTACLES_NAME, obstacle.first, obstacle.second);
  }
  return world;
}

robot_state::RobotState Scene::getCurrentRobotState() const
{
  auto state = robot_state::RobotState(robot_->model_);
  auto const joints = joint_states_listener_->waitForNew(1000.0);
  if (joints)
  {
    moveit::core::jointStateToRobotState(*joints, state);
  }
  else
  {
    ROS_ERROR_STREAM("getCurrentRobotState() has no data, returning default values. This ought to be impossible.");
  }
  state.update();
  return state;
}

void Scene::updatePlanningScene()
{
  std::lock_guard lock(planning_scene_mtx_);

  // TODO: Is this request really what we want for a more generic task?
  moveit_msgs::GetPlanningSceneRequest req;
  req.components.components =
      moveit_msgs::PlanningSceneComponents::WORLD_OBJECT_NAMES |
      moveit_msgs::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY | moveit_msgs::PlanningSceneComponents::OCTOMAP |
      moveit_msgs::PlanningSceneComponents::TRANSFORMS | moveit_msgs::PlanningSceneComponents::OBJECT_COLORS;
  moveit_msgs::GetPlanningSceneResponse resp;
  get_planning_scene_client_.call(req, resp);
  planning_scene_->processPlanningSceneWorldMsg(resp.scene.world);
  planning_scene_->setCurrentState(getCurrentRobotState());

  moveit_msgs::PlanningScene scene_msg;
  planning_scene_->getPlanningSceneMsg(scene_msg);
  planning_scene_publisher_.publish(scene_msg);
}

planning_scene::PlanningScenePtr Scene::clonePlanningScene()
{
  std::lock_guard lock(planning_scene_mtx_);
  updatePlanningScene();
  return planning_scene::PlanningScene::clone(planning_scene_);
}
