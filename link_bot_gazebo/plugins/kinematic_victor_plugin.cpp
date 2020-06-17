#include "kinematic_victor_plugin.h"

#include <functional>

#define create_service_options(type, name, bind) \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

#define create_service_options_private(type, name, bind) \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &private_queue_)

namespace gazebo {
GZ_REGISTER_MODEL_PLUGIN(KinematicVictorPlugin)

KinematicVictorPlugin::~KinematicVictorPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  private_ros_node_->shutdown();
  ros_queue_thread_.join();
  private_ros_queue_thread_.join();
}

void KinematicVictorPlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf)
{
  model_ = parent;
  std::string joint_name{"victor::victor_right_arm_joint_1"};
  joint_ = model_->GetJoint(joint_name);
  if (!joint_) {
    gzerr << "No joint " << joint_name << '\n';
    gzerr << "Possible Joint Nams:" << '\n';
    for (auto const j : model_->GetJoints()) {
      gzerr << j->GetName() << '\n';
    }
  }

  // setup ROS stuff
  if (!ros::isInitialized()) {
    int argc = 0;
    ros::init(argc, nullptr, model_->GetScopedName(), ros::init_options::NoSigintHandler);
  }

  auto pos_action_bind = [this](peter_msgs::JointTrajRequest &req, peter_msgs::JointTrajResponse &res) {
    return OnAction(req, res);
  };
  auto action_so = create_service_options_private(peter_msgs::JointTraj, "joint_traj", pos_action_bind);

  private_ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
  action_service_ = private_ros_node_->advertiseService(action_so);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
  private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });
}

bool KinematicVictorPlugin::OnAction(peter_msgs::JointTrajRequest &req, peter_msgs::JointTrajResponse &res)
{
  if (joint_) {
    joint_->SetPosition(0, req.joint_angle);
  }
  //  model_->GetWorld()->Step(1);
  return true;
}

void KinematicVictorPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void KinematicVictorPlugin::PrivateQueueThread()
{
  double constexpr timeout = 0.01;
  while (private_ros_node_->ok()) {
    private_queue_.callAvailable(ros::WallDuration(timeout));
  }
}
}  // namespace gazebo
