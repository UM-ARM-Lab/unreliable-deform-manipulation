#pragma once
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <string>

gazebo::physics::LinkPtr GetLink(char const* plugin_name, gazebo::physics::ModelPtr model, std::string const link_name);

gazebo::physics::JointPtr GetJoint(char const* plugin_name, gazebo::physics::ModelPtr model,
                                   std::string const link_name);

gazebo::physics::LinkPtr GetLink(char const *plugin_name,
                                 gazebo::physics::WorldPtr world,
                                 std::string const scoped_link_name);
