#include "car_plugin.h"

#include <peter_msgs/GetObject.h>
#include <peter_msgs/GetObjects.h>
#include <peter_msgs/WheelSpeeds.h>
#include <std_msgs/String.h>

#include <gazebo/common/Events.hh>
#include <gazebo/gazebo_core.hh>
#include <gazebo/physics/PhysicsTypes.hh>

void CarPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
{
  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "car_plugin", ros::init_options::NoSigintHandler);
  }

  model_ = model;

  body_ = model_->GetLink("base");
  right_wheel_ = model_->GetJoint("right_wheel");
  left_wheel_ = model_->GetJoint("left_wheel");

  auto action_bind = boost::bind(&CarPlugin::ExecuteAction, this, _1, _2);
  auto action_so = ros::AdvertiseServiceOptions::create<peter_msgs::ExecuteAction>("execute_action", action_bind,
                                                                                   ros::VoidPtr(), &queue_);
  constexpr auto car_object_service_name{"car"};
  auto get_object_car_bind = boost::bind(&CarPlugin::GetObjectCarCallback, this, _1, _2);
  auto get_object_car_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetObject>(
      car_object_service_name, get_object_car_bind, ros::VoidPtr(), &queue_);
  auto reset_bind = boost::bind(&CarPlugin::ResetRobot, this, _1, _2);
  auto reset_so = ros::AdvertiseServiceOptions::create<peter_msgs::LinkBotReset>("reset_robot", reset_bind,
                                                                                 ros::VoidPtr(), &queue_);

  execute_action_service_ = ros_node_.advertiseService(action_so);
  register_car_pub_ = ros_node_.advertise<std_msgs::String>("register_object", 10, true);
  wheel_seed_pub_ = ros_node_.advertise<peter_msgs::WheelSpeeds>("wheel_speeds", 10, true);
  get_object_car_service_ = ros_node_.advertiseService(get_object_car_so);
  objects_service_ = ros_node_.serviceClient<peter_msgs::GetObjects>("objects");
  reset_service_ = ros_node_.advertiseService(reset_so);

  ros_queue_thread_ = std::thread(std::bind(&CarPlugin::QueueThread, this));

  while (register_car_pub_.getNumSubscribers() < 1) {
  }

  std_msgs::String register_object;
  register_object.data = car_object_service_name;
  register_car_pub_.publish(register_object);

  {
    if (!sdf->HasElement("kP_vel")) {
      printf("using default kP_vel=%f\n", kP_vel_);
    }
    else {
      kP_vel_ = sdf->GetElement("kP_vel")->Get<double>();
    }

    if (!sdf->HasElement("kI_vel")) {
      printf("using default kI_vel=%f\n", kI_vel_);
    }
    else {
      kI_vel_ = sdf->GetElement("kI_vel")->Get<double>();
    }

    if (!sdf->HasElement("kD_vel")) {
      printf("using default kD_vel=%f\n", kD_vel_);
    }
    else {
      kD_vel_ = sdf->GetElement("kD_vel")->Get<double>();
    }

    if (!sdf->HasElement("kFF_vel")) {
      printf("using default kFF_vel=%f\n", kFF_vel_);
    }
    else {
      kFF_vel_ = sdf->GetElement("kFF_vel")->Get<double>();
    }

    if (!sdf->HasElement("max_force")) {
      printf("using default max_force=%f\n", max_force_);
    }
    else {
      max_force_ = sdf->GetElement("max_force")->Get<double>();
    }
  }

  ros_node_.setParam("n_action", 2);
  ros_node_.setParam("max_speed", max_speed_);
  constexpr auto max_integral{1};
  left_wheel_vel_pid_ = common::PID(kP_vel_, kI_vel_, kD_vel_, max_integral, -max_integral, max_force_, -max_force_);
  right_wheel_vel_pid_ = common::PID(kP_vel_, kI_vel_, kD_vel_, max_integral, -max_integral, max_force_, -max_force_);

  // Connect to the world update event.
  // This will trigger the Update function every Gazebo iteration
  update_connection_ = event::Events::ConnectWorldUpdateBegin(boost::bind(&CarPlugin::Update, this, _1));
}

void CarPlugin::Update(const common::UpdateInfo &info)
{
  UpdateControl();
  left_wheel_->SetForce(0, left_force_);
  right_wheel_->SetForce(0, right_force_);
}

void CarPlugin::UpdateControl()
{
  std::lock_guard<std::mutex> guard(control_mutex_);
  auto const dt = model_->GetWorld()->Physics()->GetMaxStepSize();

  auto const current_left_wheel_velocity = left_wheel_->GetVelocity(0);
  auto const left_error = current_left_wheel_velocity - left_wheel_target_velocity_;
  auto const delta_left_force = left_wheel_vel_pid_.Update(left_error, dt);

  auto const current_right_wheel_velocity = right_wheel_->GetVelocity(0);
  auto const right_error = current_right_wheel_velocity - right_wheel_target_velocity_;
  auto const delta_right_force = right_wheel_vel_pid_.Update(right_error, dt);

  left_force_ = kFF_vel_ * left_wheel_target_velocity_ + delta_left_force;
  right_force_ = kFF_vel_ * right_wheel_target_velocity_ + delta_right_force;

  peter_msgs::WheelSpeeds wheel_speeds;
  wheel_speeds.left_speed = current_left_wheel_velocity;
  wheel_speeds.right_speed = current_right_wheel_velocity;
  wheel_speeds.left_force = left_force_;
  wheel_speeds.right_force = right_force_;
  wheel_seed_pub_.publish(wheel_speeds);
}

bool CarPlugin::GetObjectCarCallback(peter_msgs::GetObjectRequest &req, peter_msgs::GetObjectResponse &res)
{
  auto const pose = body_->WorldCoGPose();
  float const x = pose.Pos()[0];
  float const y = pose.Pos()[1];
  float const yaw = pose.Rot().Euler().Z();

  auto const linear_velocity = body_->WorldCoGLinearVel();
  float const xdot = linear_velocity.X();
  float const ydot = linear_velocity.Y();
  float const yawdot = body_->WorldAngularVel().Z();

  std::vector<float> state{x, y, yaw, xdot, ydot, yawdot};
  res.object.state_vector = state;
  res.object.name = "car";

  return true;
}

bool CarPlugin::ExecuteAction(peter_msgs::ExecuteActionRequest &req, peter_msgs::ExecuteActionResponse &res)
{
  left_wheel_target_velocity_ = req.action.action[0];
  right_wheel_target_velocity_ = req.action.action[1];

  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const steps = static_cast<unsigned int>(req.action.max_time_per_step / seconds_per_step);
  // Wait until the setpoint is reached
  model_->GetWorld()->Step(steps);

  // set setpoint to zero after
  left_wheel_target_velocity_ = 0;
  right_wheel_target_velocity_ = 0;

  return true;
}

bool CarPlugin::ResetRobot(peter_msgs::LinkBotResetRequest &req, peter_msgs::LinkBotResetResponse &res) {}

void CarPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

CarPlugin::~CarPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  ros_queue_thread_.join();
}

GZ_REGISTER_MODEL_PLUGIN(CarPlugin)
