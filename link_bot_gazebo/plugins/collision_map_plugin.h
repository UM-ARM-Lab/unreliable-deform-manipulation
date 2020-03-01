#include <math.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <sdf/sdf.hh>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <ignition/math/Vector3.hh>

#include <sdf_tools/SDF.h>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/voxel_grid.hpp>
#include <sdf_tools/collision_map.hpp>
#include <peter_msgs/ComputeOccupancy.h>

namespace gazebo {

class CollisionMapPlugin : public WorldPlugin {
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::Subscriber sub_;
  ros::Publisher gazebo_sdf_viz_pub_;
  ros::ServiceServer query_service_;
  ros::ServiceServer get_service_;
  ros::ServiceServer get_service2_;
  ros::ServiceServer get_occupancy_service_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  gazebo::physics::RayShapePtr ray;
  bool ready_{false};

  sdf_tools::CollisionMapGrid grid_;
  sdf_tools::SignedDistanceField sdf_;
  VoxelGrid::VoxelGrid<std::vector<double>> sdf_gradient_;

  static const sdf_tools::COLLISION_CELL oob_value;
  static const sdf_tools::COLLISION_CELL occupied_value;
  static const sdf_tools::COLLISION_CELL unoccupied_value;

 public:
  void Load(physics::WorldPtr world, sdf::ElementPtr _sdf) override;

 public:
  ~CollisionMapPlugin() override;

 private:
  void QueueThread();

  void compute_sdf(int64_t h_rows, int64_t w_cols, geometry_msgs::Point center, float resolution,
                   std::string const &robot_name, float min_z, float max_z, bool verbose = false);
};
}  // namespace gazebo
