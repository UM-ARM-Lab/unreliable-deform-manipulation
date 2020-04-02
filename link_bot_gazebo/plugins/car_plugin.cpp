#include "car_plugin.h"

#include <gazebo/common/Events.hh>
#include <gazebo/gazebo_core.hh>
#include <gazebo/physics/PhysicsTypes.hh>

void CarPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
{
  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "multi_link_bot_model_plugin", ros::init_options::NoSigintHandler);
  }

  model_ = model;

  body_ = model_->GetLink(sdf->Get<std::string>("base"));
  right_wheel_ = model_->GetJoint("right_wheel_joint");
  left_wheel_ = model_->GetJoint("left_wheel_joint");

  auto action_bind = boost::bind(&CarPlugin::ExecuteAction, this, _1, _2);
  auto action_so = ros::AdvertiseServiceOptions::create<peter_msgs::ExecuteAction>("execute_action", action_bind,
                                                                                   ros::VoidPtr(), &queue_);
  auto action_mode_bind = boost::bind(&CarPlugin::OnActionMode, this, _1);
  auto action_mode_so = ros::SubscribeOptions::create<std_msgs::String>("link_bot_action_mode", 1, action_mode_bind,
                                                                        ros::VoidPtr(), &queue_);
  auto state_bind = boost::bind(&CarPlugin::StateServiceCallback, this, _1, _2);
  auto service_so = ros::AdvertiseServiceOptions::create<peter_msgs::LinkBotState>("link_bot_state", state_bind,
                                                                                   ros::VoidPtr(), &queue_);
  constexpr auto gripper_service_name{"gripper"};
  auto get_object_gripper_bind = boost::bind(&CarPlugin::GetObjectGripperCallback, this, _1, _2);
  auto get_object_gripper_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetObject>(
      gripper_service_name, get_object_gripper_bind, ros::VoidPtr(), &queue_);
  constexpr auto link_bot_service_name{"link_bot"};
  auto get_object_link_bot_bind = boost::bind(&CarPlugin::GetObjectLinkBotCallback, this, _1, _2);
  auto get_object_link_bot_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetObject>(
      link_bot_service_name, get_object_link_bot_bind, ros::VoidPtr(), &queue_);
  auto reset_bind = boost::bind(&CarPlugin::LinkBotReset, this, _1, _2);
  auto reset_so = ros::AdvertiseServiceOptions::create<peter_msgs::LinkBotReset>("link_bot_reset", reset_bind,
                                                                                 ros::VoidPtr(), &queue_);

  joy_sub_ = ros_node_.subscribe(joy_so);
  execute_action_service_ = ros_node_.advertiseService(action_so);
  execute_absolute_action_service_ = ros_node_.advertiseService(execute_abs_action_so);
  register_object_pub_ = ros_node_.advertise<std_msgs::String>("register_object", 10, true);
  reset_service_ = ros_node_.advertiseService(reset_so);
  action_mode_sub_ = ros_node_.subscribe(action_mode_so);
  state_service_ = ros_node_.advertiseService(service_so);
  get_object_gripper_service_ = ros_node_.advertiseService(get_object_gripper_so);
  get_object_link_bot_service_ = ros_node_.advertiseService(get_object_link_bot_so);
  execute_traj_service_ = ros_node_.advertiseService(execute_trajectory_so);
  objects_service_ = ros_node_.serviceClient<peter_msgs::GetObjects>("objects");

  ros_queue_thread_ = std::thread(std::bind(&CarPlugin::QueueThread, this));
  execute_trajs_ros_queue_thread_ = std::thread(std::bind(&CarPlugin::QueueThread, this));

  while (register_object_pub_.getNumSubscribers() < 1) {
  }

  {
    std_msgs::String register_object;
    register_object.data = link_bot_service_name;
    register_object_pub_.publish(register_object);
  }

  {
    std_msgs::String register_object;
    register_object.data = gripper_service_name;
    register_object_pub_.publish(register_object);
  }

  model_ = parent;

  {
    if (!sdf->HasElement("rope_length")) {
      printf("using default rope length=%f\n", length_);
    }
    else {
      length_ = sdf->GetElement("rope_length")->Get<double>();
    }

    if (!sdf->HasElement("num_links")) {
      printf("using default num_links=%u\n", num_links_);
    }
    else {
      num_links_ = sdf->GetElement("num_links")->Get<unsigned int>();
    }

    if (!sdf->HasElement("kP_pos")) {
      printf("using default kP_pos=%f\n", kP_pos_);
    }
    else {
      kP_pos_ = sdf->GetElement("kP_pos")->Get<double>();
    }

    if (!sdf->HasElement("kI_pos")) {
      printf("using default kI_pos=%f\n", kI_pos_);
    }
    else {
      kI_pos_ = sdf->GetElement("kI_pos")->Get<double>();
    }

    if (!sdf->HasElement("kD_pos")) {
      printf("using default kD_pos=%f\n", kD_pos_);
    }
    else {
      kD_pos_ = sdf->GetElement("kD_pos")->Get<double>();
    }

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

    if (!sdf->HasElement("gripper1_link")) {
      throw std::invalid_argument("no gripper1_link tag provided");
    }

    if (!sdf->HasElement("max_force")) {
      printf("using default max_force=%f\n", max_force_);
    }
    else {
      max_force_ = sdf->GetElement("max_force")->Get<double>();
    }

    if (!sdf->HasElement("max_vel")) {
      printf("using default max_vel=%f\n", max_vel_);
    }
    else {
      max_vel_ = sdf->GetElement("max_vel")->Get<double>();
    }
  }

  ros_node_.setParam("n_action", 2);
  ros_node_.setParam("link_bot/rope_length", length_);
  ros_node_.setParam("max_speed", max_speed_);

  left_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);
  right_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);

  // Connect to the world update event.
  // This will trigger the Update function every Gazebo iteration
  update_connection_ = event::Events::ConnectWorldUpdateBegin(boost::bind(&CarPlugin::Update, this, _1));
}

void CarPlugin::Update(const common::UpdateInfo &info)
{
  ControlResult control = UpdateControl();

  left_wheel_vel_pid_->s

  ignition::math::Pose3d relativePose = body_->WorldCoGPose();

  msgs::Vector3d *pos = new msgs::Vector3d();
  pos->set_x(relativePose.Pos()[0]);
  pos->set_y(relativePose.Pos()[1]);
  pos->set_z(relativePose.Pos()[2]);

  msgs::Quaternion *rot = new msgs::Quaternion();
  rot->set_x(relativePose.Rot().X());
  rot->set_y(relativePose.Rot().Y());
  rot->set_z(relativePose.Rot().Z());
  rot->set_w(relativePose.Rot().W());

  msgs::Pose *pose = new msgs::Pose();
  pose->set_allocated_position(pos);
  pose->set_allocated_orientation(rot);

  float left_vel_rps = 0;
  float right_vel_rps = 0;
  float left_angle = 0;
  float right_angle = 0;
  left_vel_rps = left_wheel_->GetVelocity(0);
  right_vel_rps = right_wheel_->GetVelocity(0);
  left_angle = left_wheel_->Position();
  right_angle = right_wheel_->Position();
}

GZ_REGISTER_MODEL_PLUGIN(CarPlugin)
