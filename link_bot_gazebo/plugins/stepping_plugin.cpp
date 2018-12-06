#include <ignition/math/Pose3.hh>
#include <link_bot_gazebo/WorldControl.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <ros/callback_queue.h>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <atomic>

namespace gazebo {
    class SteppingPlugin : public WorldPlugin {
    private:
        transport::PublisherPtr pub_;
        event::ConnectionPtr step_connection__;
        std::unique_ptr<ros::NodeHandle> ros_node_;
        ros::ServiceServer service_;
        ros::CallbackQueue queue_;
        std::thread ros_queue_thread_;
        std::atomic<int> step_count_{0};

        void QueueThread() {
            double constexpr timeout = 0.01;
            while (ros_node_->ok()) {
                queue_.callAvailable(ros::WallDuration(timeout));
            }
        }


    public:
        void Load(physics::WorldPtr _parent, sdf::ElementPtr /*_sdf*/) override {
            // set up ros topic
            if (!ros::isInitialized()) {
                auto argc = 0;
                char **argv = nullptr;
                ros::init(argc, argv, "linkbot_model_plugin", ros::init_options::NoSigintHandler);
            }

            ros_node_ = std::make_unique<ros::NodeHandle>("linkbot_model_plugin");
            auto cb = [&](link_bot_gazebo::WorldControlRequest &req, link_bot_gazebo::WorldControlResponse &res) {
                step_count_ = req.steps;
                msgs::WorldControl gz_msg;
                gz_msg.set_multi_step(req.steps);
                pub_->Publish(gz_msg);
                while (step_count_ != 0) ;
                return true;
            };

            auto so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::WorldControl>("/world_control", cb,
                                                                                          ros::VoidConstPtr(), &queue_);
            service_ = ros_node_->advertiseService(so);
            ros_queue_thread_ = std::thread(std::bind(&SteppingPlugin::QueueThread, this));

            // set up gazebo topic
            transport::NodePtr node(new transport::Node());
            node->Init(_parent->Name());

            pub_ = node->Advertise<msgs::WorldControl>("~/world_control");

            step_connection__ = event::Events::ConnectWorldUpdateEnd([&]() {
                if (step_count_ > 0) {
                    --step_count_;
                }
            });

            printf("Finished loading stepping plugin!\n");
        }

        ~SteppingPlugin() override {
            queue_.clear();
            queue_.disable();
            ros_node_->shutdown();
            ros_queue_thread_.join();
        }
    };

// Register this plugin with the simulator
    GZ_REGISTER_WORLD_PLUGIN(SteppingPlugin)
}