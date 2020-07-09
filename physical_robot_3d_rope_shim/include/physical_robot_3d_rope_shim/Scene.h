//
// Created by arprice on 12/11/18.
//

#ifndef MPS_SCENE_H
#define MPS_SCENE_H

#include <Eigen/StdVector>
#include <moveit/collision_detection/world.h>

#include "victor_3d_rope_shim/Manipulator.h"


class Scene
{
public:
	static auto constexpr OBSTACLES_NAME = "static_obstacles";

	robot_model::RobotModelPtr robotModel;
	std::vector<std::shared_ptr<Manipulator>> manipulators;
	std::map<std::string, std::shared_ptr<Manipulator>> jointToManipulator;
	std::vector<std::pair<std::shared_ptr<shapes::Shape>, Pose>> staticObstacles;

	bool loadManipulators(robot_model::RobotModelPtr& pModel);
	collision_detection::WorldPtr computeCollisionWorld();
};

#endif // MPS_SCENE_H
