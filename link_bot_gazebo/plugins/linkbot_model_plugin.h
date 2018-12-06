#pragma once

#include <string>
#include <thread>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math.hh>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/Joy.h>
#include <link_bot_gazebo/LinkBotConfiguration.h>

namespace gazebo {
    class LinkBotModelPlugin : public ModelPlugin {
    public:
        void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;

        ~LinkBotModelPlugin() override;

        void OnUpdate();

        void OnCmdVel(sensor_msgs::JoyConstPtr _msg);

        void OnConfiguration(link_bot_gazebo::LinkBotConfigurationConstPtr _msg);

    protected:

    private:
        void QueueThread();

        physics::ModelPtr model_;
        event::ConnectionPtr updateConnection_;
        double kP_{500};
        double kI_{0};
        double kD_{0};
        common::PID x_pid_;
        common::PID y_pid_;
        std::string control_link_name_ = "head";
        ignition::math::Vector3d target_linear_vel_{0, 0, 0};
        physics::LinkPtr control_link_;
        std::unique_ptr<ros::NodeHandle> ros_node_;
        ros::Subscriber cmd_sub_;
        ros::Subscriber config_sub_;
        ros::CallbackQueue queue_;
        std::thread ros_queue_thread_;
        double joy_scale_{1.0};
    };
}
