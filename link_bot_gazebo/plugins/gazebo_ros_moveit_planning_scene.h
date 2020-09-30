#pragma once
/*
 * Copyright (C) 2012-2014 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
/*
 * Desc: A plugin which publishes the gazebo world state as a MoveIt! planning scene
 * Author: Jonathan Bohren
 * Date: 15 May 2014
 */
#pragma once

#include <atomic>
#include <string>

#include <geometry_msgs/Pose.h>
#include <ros/callback_queue.h>
#include <ros/subscribe_options.h>

#include <ros/ros.h>
#include <boost/thread.hpp>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>

#include <moveit_msgs/GetPlanningScene.h>
#include <moveit_msgs/ObjectColor.h>
#include <moveit_msgs/PlanningScene.h>
#include <std_msgs/ColorRGBA.h>
#include <std_srvs/Empty.h>

namespace gazebo
{
class GazeboRosMoveItPlanningScene : public ModelPlugin
{
public:
  virtual ~GazeboRosMoveItPlanningScene();

  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);

  moveit_msgs::PlanningScene BuildMessage();

  void PeriodicUpdate();

  void QueueThread();

  physics::WorldPtr world_;
  physics::ModelPtr model_;

  boost::scoped_ptr<ros::NodeHandle> rosnode_;

  ros::Publisher planning_scene_pub_;

  std::string topic_name_;
  std::string scene_name_;
  std::string frame_id_;
  std::string robot_name_;
  std::string model_name_;
  std::string robot_namespace_;
  std::vector<std::string> excluded_model_names;

  // We need a separate queue and thread so we can handle messages and services when gazebo is paused
  ros::CallbackQueue queue_;
  boost::thread callback_queue_thread_;
  // Protects against multiple ROS callbacks or publishers accessing/changing data out of order
  std::mutex ros_mutex_;

  std::thread periodic_event_thread_;
  double scale_primitives_factor_{ 1.0 };
};

}  // namespace gazebo
