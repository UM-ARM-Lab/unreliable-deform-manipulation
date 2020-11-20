#include <link_bot_gazebo/base_link_position_controller.h>
#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include <ros/console.h>

namespace gazebo
{

BaseLinkPositionController::BaseLinkPositionController(char const *plugin_name, physics::LinkPtr link, std::string const type) :
    plugin_name_(plugin_name),
    link_(link),
    scoped_link_name_(link->GetScopedName()),
    tf_listener_(tf_buffer_),
    setpoint_(link->WorldPose().Pos()),
    type(type)
{
}

std::optional<ignition::math::Vector3d> BaseLinkPositionController::Get() const
{
  return link_->WorldPose().Pos();
}

void BaseLinkPositionController::OnFollow(std::string const &frame_id)
{
  OnEnable(true);
  following_ = true;
  following_frame_id_ = frame_id;
  speed_mps_ = std::numeric_limits<typeof(speed_mps_)>::max();
}

void BaseLinkPositionController::OnUpdate()
{
  if (not enabled_)
  {
    return;
  }

  if (following_)
  {
    try
    {
      auto const transform_stamped = tf_buffer_.lookupTransform("world", following_frame_id_, ros::Time(0));
      auto const trans = transform_stamped.transform.translation;
      Update({trans.x, trans.y, trans.z});
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN_STREAM_NAMED(plugin_name_, ex.what());
    }
  } else
  {
    Update(setpoint_);
  }
}

void BaseLinkPositionController::Set(peter_msgs::Position3DActionRequest action)
{
  OnEnable(true);
  following_ = false;
  setpoint_ = point_to_ign_vector_3d(action.position);
  timeout_s_ = action.timeout_s;
  speed_mps_ = action.speed_mps;
}


void BaseLinkPositionController::OnStop()
{
  auto const pos = Get();
  if (pos)
  {
    setpoint_ = *pos;
  }
}

void BaseLinkPositionController::OnEnable(bool enable)
{
  enabled_ = enable;
  OnStop();
}

}  // namespace gazebo
