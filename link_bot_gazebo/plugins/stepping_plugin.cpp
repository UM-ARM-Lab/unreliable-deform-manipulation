#include <peter_msgs/WorldControl.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>

#include <atomic>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/Pose3.hh>
#include <memory>

namespace gazebo
{
class SteppingPlugin : public WorldPlugin
{
private:
  transport::PublisherPtr pub_;
  event::ConnectionPtr step_connection__;
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::ServiceServer service_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  std::atomic<int> step_count_{ 0 };
  double seconds_per_step_{ 0.0 };

  void QueueThread()
  {
    double constexpr timeout = 0.01;
    while (ros_node_->ok())
    {
      queue_.callAvailable(ros::WallDuration(timeout));
    }
  }

public:
  void Load(physics::WorldPtr parent, sdf::ElementPtr /*_sdf*/) override
  {
    if (!ros::isInitialized())
    {
      ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
        << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
      return;
    }

    seconds_per_step_ = parent->Physics()->GetMaxStepSize();

    ros_node_ = std::make_unique<ros::NodeHandle>("stepping_plugin");
    auto cb = [&](peter_msgs::WorldControlRequest &req, peter_msgs::WorldControlResponse &res) {
      (void)res;
      auto const steps = [this, req]() {
        if (req.seconds > 0)
        {
          return static_cast<unsigned int>(req.seconds / seconds_per_step_);
        }
        else
        {
          return req.steps;
        }
      }();
      step_count_ = steps;
      msgs::WorldControl gz_msg;
      gz_msg.set_multi_step(steps);
      pub_->Publish(gz_msg);
      while (step_count_ != 0)
        ;
      return true;
    };

    auto so = ros::AdvertiseServiceOptions::create<peter_msgs::WorldControl>("/world_control", cb, ros::VoidConstPtr(),
                                                                             &queue_);
    service_ = ros_node_->advertiseService(so);
    ros_queue_thread_ = std::thread(std::bind(&SteppingPlugin::QueueThread, this));

    // set up gazebo topic
    transport::NodePtr node(new transport::Node());
    node->Init(parent->Name());

    pub_ = node->Advertise<msgs::WorldControl>("~/world_control");

    step_connection__ = event::Events::ConnectWorldUpdateEnd([&]() {
      if (step_count_ > 0)
      {
        --step_count_;
      }
    });

    ROS_INFO("Finished loading stepping plugin!");
  }

  ~SteppingPlugin() override
  {
    queue_.clear();
    queue_.disable();
    ros_node_->shutdown();
    ros_queue_thread_.join();
  }
};

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(SteppingPlugin)
}  // namespace gazebo
