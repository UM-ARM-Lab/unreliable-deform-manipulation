#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include <ros/ros.h>

gazebo::physics::LinkPtr GetLink(char const *plugin_name, gazebo::physics::ModelPtr model, std::string const link_name)
{
  auto const scoped_model_name = model->GetScopedName();
  auto const full_link_name = scoped_model_name + "::" + link_name;
  gazebo::physics::LinkPtr link = model->GetLink(full_link_name);
  if (!link)
  {
    ROS_ERROR_STREAM_NAMED(plugin_name, "No link " << full_link_name << " found");
    ROS_WARN_STREAM_NAMED(plugin_name, "Available links are:");
    for (const auto &l : model->GetLinks())
    {
      ROS_WARN_STREAM_NAMED(plugin_name, l->GetName());
    }
  }
  return link;
}

gazebo::physics::JointPtr GetJoint(char const *plugin_name, gazebo::physics::ModelPtr model,
                                   std::string const joint_name)
{
  auto const scoped_model_name = model->GetScopedName();
  auto const full_joint_name = scoped_model_name + "::" + joint_name;
  gazebo::physics::JointPtr joint = model->GetJoint(full_joint_name);
  if (!joint)
  {
    ROS_ERROR_STREAM_NAMED(plugin_name, "No joint " << full_joint_name << " found");
    ROS_WARN_STREAM_NAMED(plugin_name, "Available joints are:");
    for (const auto &j : model->GetJoints())
    {
      ROS_WARN_STREAM_NAMED(plugin_name, j->GetName());
    }
  }
  return joint;
}