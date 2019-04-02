#include <iostream>
#include <math.h>
#include <boost/shared_ptr.hpp>
#include <sdf/sdf.hh>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <ros/callback_queue.h>

#include <gazebo/gazebo.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>

#include <link_bot_gazebo/WriteSDF.h>
#include <visualization_msgs/Marker.h>
#include <sdf_tools/collision_map.hpp>
#include <sdf_tools/SDF.h>

namespace gazebo {

    class CollisionMapPlugin : public WorldPlugin {
        std::unique_ptr<ros::NodeHandle> ros_node_;
        ros::Subscriber sub_;
        ros::Publisher gazebo_sdf_viz_pub_;
        ros::CallbackQueue queue_;
        std::thread ros_queue_thread_;
        visualization_msgs::Marker collision_map_marker_for_export_;
        gazebo::physics::RayShapePtr ray;

        static const sdf_tools::COLLISION_CELL oob_value;
        static const sdf_tools::COLLISION_CELL occupied_value;

    public:
        void Load(physics::WorldPtr world, sdf::ElementPtr _sdf);

    public:
        void OnWriteSDF(link_bot_gazebo::WriteSDFConstPtr msg);

        ~CollisionMapPlugin();

    private:
        void QueueThread();

    };
}
