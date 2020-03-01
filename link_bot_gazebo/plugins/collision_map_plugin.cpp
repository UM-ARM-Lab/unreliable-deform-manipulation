#include <cnpy/cnpy.h>
#include <std_msgs/ColorRGBA.h>
#include <std_msgs/MultiArrayDimension.h>
#include <visualization_msgs/Marker.h>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/serialization.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <chrono>
#include <experimental/filesystem>
#include <functional>

#include "collision_map_plugin.h"

using namespace gazebo;

const sdf_tools::COLLISION_CELL CollisionMapPlugin::oob_value{-10000};
const sdf_tools::COLLISION_CELL CollisionMapPlugin::occupied_value{1};
const sdf_tools::COLLISION_CELL CollisionMapPlugin::unoccupied_value{0};

void CollisionMapPlugin::Load(physics::WorldPtr world, sdf::ElementPtr _sdf)
{
  auto engine = world->Physics();
  engine->InitForThread();
  auto ray_shape = engine->CreateShape("ray", gazebo::physics::CollisionPtr());
  ray = boost::dynamic_pointer_cast<gazebo::physics::RayShape>(ray_shape);

  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "collision_map_plugin", ros::init_options::NoSigintHandler);
  }

  auto get_occupancy = [&](peter_msgs::ComputeOccupancyRequest &req, peter_msgs::ComputeOccupancyResponse &res) {
    if (req.request_new) {
      compute_sdf(req.h_rows, req.w_cols, req.center, req.resolution, req.robot_name, req.min_z, req.max_z);
    }
    res.h_rows = req.h_rows;
    res.w_cols = req.w_cols;
    res.res = std::vector<float>(2, req.resolution);

    auto const grid_00_x = req.center.x - static_cast<float>(req.w_cols) * req.resolution / 2.0;
    auto const grid_00_y = req.center.y - static_cast<float>(req.h_rows) * req.resolution / 2.0;
    auto const origin_x_col = static_cast<int>(-grid_00_x / req.resolution);
    auto const origin_y_row = static_cast<int>(-grid_00_y / req.resolution);

    std::vector<int> origin_vec{origin_y_row, origin_x_col};
    res.origin = origin_vec;
    auto const grid_float = [&]() {
      auto const &data = grid_.GetImmutableRawData();
      std::vector<float> flat;
      for (auto const &d : data) {
        flat.emplace_back(d.occupancy);
      }
      return flat;
    }();
    res.grid = grid_float;
    std_msgs::MultiArrayDimension row_dim;
    row_dim.label = "row";
    row_dim.size = grid_.GetNumXCells();
    row_dim.stride = 1;
    std_msgs::MultiArrayDimension col_dim;
    col_dim.label = "col";
    col_dim.size = grid_.GetNumYCells();
    col_dim.stride = 1;
    return true;
  };

  ros_node_ = std::make_unique<ros::NodeHandle>("collision_map_plugin");

  gazebo_sdf_viz_pub_ = ros_node_->advertise<visualization_msgs::Marker>("gazebo_sdf_viz", 1);

  {
    auto so = ros::AdvertiseServiceOptions::create<peter_msgs::ComputeOccupancy>("/occupancy", get_occupancy, ros::VoidConstPtr(),
                                                                                 &queue_);
    get_occupancy_service_ = ros_node_->advertiseService(so);
  }

  ros_queue_thread_ = std::thread(std::bind(&CollisionMapPlugin::QueueThread, this));
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
  while (ros_node_->ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void CollisionMapPlugin::compute_sdf(int64_t h_rows, int64_t w_cols, geometry_msgs::Point center, float resolution,
                                     std::string const &robot_name, float min_z, float max_z, bool verbose)
{
  Eigen::Isometry3d origin_transform = Eigen::Isometry3d::Identity();
  auto const x_width = resolution * w_cols;
  auto const y_height = resolution * h_rows;
  origin_transform.translation() = Eigen::Vector3d{center.x - x_width / 2, center.y - y_height / 2, 0};
  // hard coded for 1-cell in Z
  grid_ = sdf_tools::CollisionMapGrid(origin_transform, "/gazebo_world", resolution, w_cols, h_rows, 1l,
                                      oob_value);
  ignition::math::Vector3d start, end;
  start.Z(max_z);
  end.Z(min_z);

  std::string entityName;
  double dist{0};

  auto const t0 = std::chrono::steady_clock::now();

  for (auto x_idx{0l}; x_idx < grid_.GetNumXCells(); ++x_idx) {
    for (auto y_idx{0l}; y_idx < grid_.GetNumYCells(); ++y_idx) {
      auto const grid_location = grid_.GridIndexToLocation(x_idx, y_idx, 0);
      start.X(grid_location(0));
      end.X(grid_location(0));
      start.Y(grid_location(1));
      end.Y(grid_location(1));
      ray->SetPoints(start, end);
      ray->GetIntersection(dist, entityName);
      if (not entityName.empty() and (robot_name.empty() or entityName.find(robot_name) != 0)) {
        grid_.SetValue(x_idx, y_idx, 0, occupied_value);
      }
      else {
        grid_.SetValue(x_idx, y_idx, 0, unoccupied_value);
      }
    }
  }

  auto const t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> const time_to_compute_occupancy_grid = t1 - t0;
  if (verbose) {
    std::cout << "Time to compute occupancy grid_: " << time_to_compute_occupancy_grid.count() << std::endl;
  }

  sdf_ = grid_.ExtractSignedDistanceField(oob_value.occupancy, false, false).first;

  auto const t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> const time_to_compute_sdf = t2 - t1;
  if (verbose) {
    std::cout << "Time to compute sdf: " << time_to_compute_sdf.count() << std::endl;
  }

  auto const t3 = std::chrono::steady_clock::now();
  sdf_tools::SignedDistanceField::GradientFunction gradient_function = [&](const int64_t x_index, const int64_t y_index,
                                                                           const int64_t z_index,
                                                                           const bool enable_edge_gradients = false) {
    return sdf_.GetGradient(x_index, y_index, z_index, enable_edge_gradients);
  };
  sdf_gradient_ = sdf_.GetFullGradient(gradient_function, true);
  auto const t4 = std::chrono::steady_clock::now();
  if (verbose) {
    std::chrono::duration<double> const time_to_compute_sdf_gradient = t4 - t3;
    std::cout << "Time to compute sdf gradient: " << time_to_compute_sdf_gradient.count() << std::endl;
  }
}

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(CollisionMapPlugin)
