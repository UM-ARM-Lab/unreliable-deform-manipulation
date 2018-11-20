#include "linkbot_model_plugin.h"

#include <functional>
#include <ignition/math/Vector3.hh>

namespace gazebo {

    void LinkBotModelPlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf) {
        // Make sure the ROS node for Gazebo has already been initalized
        // Initialize ros, if it has not already bee initialized.
        if (!ros::isInitialized()) {
            int argc = 0;
            char **argv = nullptr;
            ros::init(argc, argv, "linkbot_model_plugin", ros::init_options::NoSigintHandler);
        }

        ros_node_.reset(new ros::NodeHandle("linkbot_model_plugin"));

        ros::SubscribeOptions so = ros::SubscribeOptions::create<sensor_msgs::Joy>(
                "/joy",
                1, boost::bind(&LinkBotModelPlugin::OnCmdVel, this, _1),
                ros::VoidPtr(), &queue_);
        cmd_sub_ = ros_node_->subscribe(so);
        ros_queue_thread_ = std::thread(std::bind(&LinkBotModelPlugin::QueueThread, this));

        if (!sdf->HasElement("kP")) {
            printf("using default kP=%f\n", kP_);
        } else {
            kP_ = sdf->GetElement("kP")->Get<double>();
        }

        if (!sdf->HasElement("kI")) {
            printf("using default kI=%f\n", kI_);
        } else {
            kI_ = sdf->GetElement("kI")->Get<double>();
        }

        if (!sdf->HasElement("kD")) {
            printf("using default kD=%f\n", kD_);
        } else {
            kD_ = sdf->GetElement("kD")->Get<double>();
        }


        if (!sdf->HasElement("joy_scale")) {
            printf("using default joy_scale=%f\n", joy_scale_);
        } else {
            joy_scale_ = sdf->GetElement("joy_scale")->Get<double>();
        }

        if (!sdf->HasElement("link_name")) {
            printf("using default link_name=%s\n", link_name_.c_str());
        } else {
            link_name_ = sdf->GetElement("link_name")->Get<std::string>();
        }

        ROS_INFO("kP=%f, kI=%f, kD=%f", kP_, kI_, kD_);

        model_ = parent;
        link_ = parent->GetLink(link_name_);
        alive_ = true;
        updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&LinkBotModelPlugin::OnUpdate, this));
        x_pid_ = common::PID(kP_, kI_, kD_, 100, -100, 800, -800);
        y_pid_ = common::PID(kP_, kI_, kD_, 100, -100, 800, -800);
    }

    void LinkBotModelPlugin::OnUpdate() {
        auto const current_linear_vel = link_->GetWorldLinearVel();
        auto const error = current_linear_vel - target_linear_vel_;
        math::Vector3 force;
        force.x = x_pid_.Update(error.x, 0.001);
        force.y = y_pid_.Update(error.y, 0.001);
        ROS_WARN_THROTTLE(1, "fx: %f, fy: %f", force.x, force.y);
        link_->AddForce(force);
    }

    void LinkBotModelPlugin::OnCmdVel(sensor_msgs::JoyConstPtr const &msg) {
        target_linear_vel_.x = -msg->axes[1] * joy_scale_;
        target_linear_vel_.y = -msg->axes[0] * joy_scale_;
        ROS_WARN("TARGET: %f, %f", target_linear_vel_.x, target_linear_vel_.y);
    }

    void LinkBotModelPlugin::QueueThread() {
        double constexpr timeout = 0.01;
        while (ros_node_->ok()) {
            queue_.callAvailable(ros::WallDuration(timeout));
        }
    }

    LinkBotModelPlugin::~LinkBotModelPlugin() {
        alive_ = false;
        queue_.clear();
        queue_.disable();
        ros_node_->shutdown();
        ros_queue_thread_.join();
    }

    // Register this plugin with the simulator
    GZ_REGISTER_MODEL_PLUGIN(LinkBotModelPlugin)
}
