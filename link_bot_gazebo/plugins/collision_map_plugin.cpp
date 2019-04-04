#include <link_bot_gazebo/WriteSDF.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sdf_tools/collision_map.hpp>
#include <std_msgs/ColorRGBA.h>
#include <sdf_tools/SDF.h>
#include <arc_utilities/arc_helpers.hpp>
#include <cnpy/cnpy.h>
#include <chrono>

#include "collision_map_plugin.h"

using namespace gazebo;

const sdf_tools::COLLISION_CELL CollisionMapPlugin::oob_value{-10000};
const sdf_tools::COLLISION_CELL CollisionMapPlugin::occupied_value{1};

void CollisionMapPlugin::Load(physics::WorldPtr world, sdf::ElementPtr _sdf)
{
  auto engine = world->Physics();
  engine->InitForThread();
  auto ray_shape = engine->CreateShape("ray", gazebo::physics::CollisionPtr());
  ray = boost::dynamic_pointer_cast<gazebo::physics::RayShape>(ray_shape);

  if (!ros::isInitialized())
  {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "collision_map_plugin", ros::init_options::NoSigintHandler);
  }

  ros_node_ = std::make_unique<ros::NodeHandle>("collision_map_plugin");

  auto bind = boost::bind(&CollisionMapPlugin::OnWriteSDF, this, _1);
  auto so = ros::SubscribeOptions::create<link_bot_gazebo::WriteSDF>("/write_sdf", 1, bind, ros::VoidPtr(),
                                                                     &queue_);

  gazebo_sdf_viz_pub_ = ros_node_->advertise<visualization_msgs::Marker>("gazebo_sdf_viz", 1);

  sub_ = ros_node_->subscribe(so);
  ros_queue_thread_ = std::thread(std::bind(&CollisionMapPlugin::QueueThread, this));
}

void CollisionMapPlugin::OnWriteSDF(link_bot_gazebo::WriteSDFConstPtr msg)
{
  Eigen::Isometry3d origin_transform = Eigen::Isometry3d::Identity();
  origin_transform.translation() = Eigen::Vector3d{msg->center.x - msg->x_width / 2,
                                                   msg->center.y - msg->y_height / 2, 0};
  // hard coded for 1-cell in Z
  sdf_tools::CollisionMapGrid grid{origin_transform, "/gazebo_world", msg->resolution,
                                   msg->x_width, msg->y_height, msg->resolution, oob_value};
  ignition::math::Vector3d start, end;
  start.Z(msg->max_z);
  end.Z(msg->min_z);

  // parameters needed for the GetIntersection check
  std::string entityName;
  double dist;

  auto const dont_draw_color = arc_helpers::GenerateUniqueColor<std_msgs::ColorRGBA>(0u);
  auto const collision_color = arc_helpers::GenerateUniqueColor<std_msgs::ColorRGBA>(1u);

  auto const t0 = std::chrono::steady_clock::now();

  for (auto x_idx{0l}; x_idx < grid.GetNumXCells(); ++x_idx)
  {
    for (auto y_idx{0l}; y_idx < grid.GetNumYCells(); ++y_idx)
    {
      auto const grid_location = grid.GridIndexToLocation(x_idx, y_idx, 0);
      start.X(grid_location(0));
      end.X(grid_location(0));
      start.Y(grid_location(1));
      end.Y(grid_location(1));
      ray->SetPoints(start, end);
      ray->GetIntersection(dist, entityName);
      if (not entityName.empty() and entityName.find(msg->robot_name) != 0)
      {
        grid.SetValue(x_idx, y_idx, 0, occupied_value);
      }
    }
  }

  auto const t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> const time_to_compute_occupancy_grid = t1 - t0;
  std::cout << "Time to compute occupancy grid: " << time_to_compute_occupancy_grid.count() << std::endl;


  auto const sdf = grid.ExtractSignedDistanceField(oob_value.occupancy, false, false).first;

  auto const t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> const time_to_compute_sdf = t2 - t1;
  std::cout << "Time to compute sdf: " << time_to_compute_sdf.count() << std::endl;

  auto const map_marker_msg = grid.ExportSurfacesForDisplay(collision_color, dont_draw_color, dont_draw_color);


  auto const t3 = std::chrono::steady_clock::now();
  auto const sdf_gradient = sdf.GetFullGradient(true);
  auto const sdf_gradient_flat = [&]()
  {
    auto const data = sdf_gradient.GetImmutableRawData();
    std::vector<double> flat;
    for (auto const &d : data)
    {
      for (auto const &component : d)
      {
        flat.emplace_back(component);
      }
    }
    return flat;
  }();
  auto const t4 = std::chrono::steady_clock::now();
  std::chrono::duration<double> const time_to_compute_sdf_gradient = t4 - t3;
  std::cout << "Time to compute sdf gradient: " << time_to_compute_sdf_gradient.count() << std::endl;

  // publish to rviz
  gazebo_sdf_viz_pub_.publish(map_marker_msg);


  // save to a file
  std::vector<size_t> shape{static_cast<unsigned long>(grid.GetNumXCells()),
                            static_cast<unsigned long>(grid.GetNumYCells()),
                            static_cast<unsigned long>(grid.GetNumZCells())};
  std::vector<size_t> gradient_shape{static_cast<unsigned long>(grid.GetNumXCells()),
                                     static_cast<unsigned long>(grid.GetNumYCells()),
                                     static_cast<unsigned long>(grid.GetNumZCells()),
                                     3};
  cnpy::npz_save(msg->filename, "sdf", &sdf.GetImmutableRawData()[0], shape, "w");
  cnpy::npz_save(msg->filename, "sdf_gradient", &sdf_gradient_flat[0], gradient_shape, "a");
}

CollisionMapPlugin::~CollisionMapPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_->shutdown();
  ros_queue_thread_.join();
}

void CollisionMapPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_->ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(CollisionMapPlugin)
