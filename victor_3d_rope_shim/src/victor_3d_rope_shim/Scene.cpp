//
// Created by arprice on 12/11/18.
//

#include "victor_3d_rope_shim/Scene.h"

#include "victor_3d_rope_shim/VictorManipulator.h"

bool Scene::loadManipulators(robot_model::RobotModelPtr& pModel)
{
	ros::NodeHandle nh;

	if (robotModel && robotModel.get() != pModel.get())
	{
		ROS_WARN("Overwriting Robot Model.");
	}
	robotModel = pModel;

	for (robot_model::JointModelGroup* jmg : pModel->getJointModelGroups())
	{
		if (jmg->isChain() && jmg->getSolverInstance())
		{
			jmg->getSolverInstance()->setSearchDiscretization(0.1); // 0.1 = ~5 degrees
			const auto& ees = jmg->getAttachedEndEffectorNames();
			// std::cerr << "Loaded jmg '" << jmg->getName() << "' " << jmg->getSolverInstance()->getBaseFrame() << std::endl;
			for (const std::string& eeName : ees)
			{
				robot_model::JointModelGroup* ee = pModel->getEndEffector(eeName);
				ee->getJointRoots();
				const robot_model::JointModel* rootJoint = ee->getCommonRoot();
				rootJoint->getNonFixedDescendantJointModels();

				// std::cerr << "\t-" << eeName << "\t" << ee->getFixedJointModels().size() << std::endl;
				for (const std::string& eeSubName : ee->getAttachedEndEffectorNames())
				{
					// std::cerr << "\t\t-" << eeSubName << "\t" << pModel->getEndEffector(eeSubName)->getFixedJointModels().size() << std::endl;

					auto const palmName = pModel->getEndEffector(eeSubName)->getLinkModelNames().front();
					auto manip = std::make_shared<VictorManipulator>(nh, pModel, jmg, ee, palmName);
					manipulators.emplace_back(manip);
					if (!manip->configureHardware())
					{
						ROS_FATAL_STREAM("Failed to setup hardware '" << manip->arm->getName() << "'");
					}
					for (const std::string& jName : manip->arm->getJointModelNames())
					{
						jointToManipulator[jName] = manip;
					}
					for (const std::string& jName : manip->gripper->getJointModelNames())
					{
						jointToManipulator[jName] = manip;
					}
				}
			}
		}
		else
		{
			// std::cerr << "Did not load jmg '" << jmg->getName() << "'" << std::endl;
			// std::cerr << "\t is " << (jmg->isEndEffector()?"":"not ") << "end-effector." << std::endl;
		}
	}

	return true;
}

collision_detection::WorldPtr Scene::computeCollisionWorld()
{
	auto const world = std::make_shared<collision_detection::World>();
	for (const auto& obstacle : staticObstacles)
	{
		world->addToObject(OBSTACLES_NAME, obstacle.first, obstacle.second);
	}
	return world;
}
