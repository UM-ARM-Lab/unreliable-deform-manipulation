#pragma once

#include <string>
#include <thread>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <ros/callback_queue.h>
#include <geometry_msgs/Twist.h>

namespace gazebo {
    class LinkBotModelPlugin : public ModelPlugin {
    public:
        void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;

        virtual ~LinkBotModelPlugin();

        void OnUpdate();

        void OnCmdVel(geometry_msgs::TwistConstPtr const &_msg);

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
        std::string link_name_ = "head";
        math::Vector3 target_linear_vel_{0, 0, 0};
        physics::LinkPtr link_;
        std::unique_ptr<ros::NodeHandle> ros_node_;
        ros::Subscriber ros_sub_;
        ros::CallbackQueue queue_;
        std::thread ros_queue_thread_;
        bool alive_{false};
    };
}
