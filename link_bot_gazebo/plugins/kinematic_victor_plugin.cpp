#include "kinematic_victor_plugin.h"

#include <sensor_msgs/JointState.h>
#include <std_msgs/Empty.h>

#include <functional>

#include "enumerate.h"

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
  world_ = parent->GetWorld();

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
  action_service_ = ros_node_.advertiseService(action_so);
  joint_states_pub_ = ros_node_.advertise<sensor_msgs::JointState>("joint_states", 10);
  auto interrupt_callback = [this](std_msgs::EmptyConstPtr const &msg) { this->interrupted_ = true; };
  interrupt_sub_ = ros_node_.subscribe<std_msgs::Empty>("interrupt_trajectory", 10, interrupt_callback);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
  private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

  auto update = [this](common::UpdateInfo const &info) { OnUpdate(); };
  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(update);
}

void KinematicVictorPlugin::OnUpdate()
{
  sensor_msgs::JointState msg;
  for (auto const &j : model_->GetJoints()) {
    // FIXME: why is this not equal to physics::Joint::HINGE_JOINT??
    if (j->GetType() == 576) {  // revolute
      msg.name.push_back(j->GetName());
      msg.position.push_back(j->Position(0));
      msg.velocity.push_back(j->GetVelocity(0));
      msg.effort.push_back(j->GetForce(0));
    }
  }
  joint_states_pub_.publish(msg);
}

bool KinematicVictorPlugin::OnAction(peter_msgs::JointTrajRequest &req, peter_msgs::JointTrajResponse &res)
{
  interrupted_ = false;
  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const steps = static_cast<unsigned int>(req.settling_time_seconds / seconds_per_step);
  for (auto const &point : req.traj.points) {
    for (auto pair : enumerate(req.traj.joint_names)) {
      auto const &[joint_idx, joint_name] = pair;
      for (auto t{0}; t <= steps; ++t) {
        world_->Step(1);
        if (interrupted_) {
          return true;
        }
      }
      auto joint = model_->GetJoint(joint_name);
      if (joint) {
        joint->SetPosition(0, point.positions[joint_idx]);
      }
      else {
        gzerr << "Joint trajectory message set position for non-existant joint " << joint_name << "\n";
      }
    }
  }
  return true;
}  // namespace gazebo

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
