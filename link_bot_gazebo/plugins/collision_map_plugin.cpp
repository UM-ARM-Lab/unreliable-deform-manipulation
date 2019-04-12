#include <cnpy/cnpy.h>
#include <link_bot_gazebo/WriteSDF.h>
#include <link_bot_gazebo/QuerySDF.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <arc_utilities/arc_helpers.hpp>
#include <chrono>
#include <experimental/filesystem>
#include <functional>

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

  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "collision_map_plugin", ros::init_options::NoSigintHandler);
  }

  auto cb = [&](link_bot_gazebo::QuerySDFRequest &req, link_bot_gazebo::QuerySDFResponse &res) {
    if (not ready_) {
      std::cout << "sdf not generated yet, you must publish that message first\n";
      res.success = false;
      return true;
    }
    std::tie(res.sdf_value, std::ignore) = sdf_.GetImmutable(req.point.x, req.point.y, req.point.z);
    auto const g = sdf_gradient_.GetImmutable(req.point.x, req.point.y, req.point.z);
    res.gradient.x = g.first[0];
    res.gradient.y = g.first[1];
    res.gradient.z = g.first[2];
    auto const i = sdf_.LocationToGridIndex(req.point.x, req.point.y, req.point.z);
    res.x_index = i.x;
    res.y_index = i.y;
    res.z_index = i.z;
    res.success = true;
    return true;
  };


  ros_node_ = std::make_unique<ros::NodeHandle>("collision_map_plugin");

  gazebo_sdf_viz_pub_ = ros_node_->advertise<visualization_msgs::Marker>("gazebo_sdf_viz", 1);

  {
    auto bind = boost::bind(&CollisionMapPlugin::OnWriteSDF, this, _1);
    auto so = ros::SubscribeOptions::create<link_bot_gazebo::WriteSDF>("/write_sdf", 1, bind, ros::VoidPtr(), &queue_);
    sub_ = ros_node_->subscribe(so);
  }

  {
    auto so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::QuerySDF>("/sdf", cb, ros::VoidConstPtr(), &queue_);
    service_ = ros_node_->advertiseService(so);
  }

  ros_queue_thread_ = std::thread(std::bind(&CollisionMapPlugin::QueueThread, this));
}

void CollisionMapPlugin::OnWriteSDF(link_bot_gazebo::WriteSDFConstPtr msg)
{
  std::experimental::filesystem::path const path(msg->filename);

  if (not std::experimental::filesystem::exists(path.parent_path())) {
    std::cout << "Output path parent [" << path.parent_path() << "] does not exist\n";
    return;
  }

  Eigen::Isometry3d origin_transform = Eigen::Isometry3d::Identity();
  origin_transform.translation() =
      Eigen::Vector3d{msg->center.x - msg->x_width / 2, msg->center.y - msg->y_height / 2, 0};
  // hard coded for 1-cell in Z
  sdf_tools::CollisionMapGrid grid{origin_transform, "/gazebo_world", msg->resolution, msg->x_width,
                                   msg->y_height,    msg->resolution, oob_value};
  ignition::math::Vector3d start, end;
  start.Z(msg->max_z);
  end.Z(msg->min_z);

  // parameters needed for the GetIntersection check
  std::string entityName;
  double dist;

  auto const dont_draw_color = arc_helpers::GenerateUniqueColor<std_msgs::ColorRGBA>(0u);
  auto const collision_color = arc_helpers::GenerateUniqueColor<std_msgs::ColorRGBA>(1u);

  auto const t0 = std::chrono::steady_clock::now();

  for (auto x_idx{0l}; x_idx < grid.GetNumXCells(); ++x_idx) {
    for (auto y_idx{0l}; y_idx < grid.GetNumYCells(); ++y_idx) {
      auto const grid_location = grid.GridIndexToLocation(x_idx, y_idx, 0);
      start.X(grid_location(0));
      end.X(grid_location(0));
      start.Y(grid_location(1));
      end.Y(grid_location(1));
      ray->SetPoints(start, end);
      ray->GetIntersection(dist, entityName);
      if (not entityName.empty() and (msg->robot_name.empty() or entityName.find(msg->robot_name) != 0)) {
        grid.SetValue(x_idx, y_idx, 0, occupied_value);
      }
    }
  }

  auto const t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> const time_to_compute_occupancy_grid = t1 - t0;
  std::cout << "Time to compute occupancy grid: " << time_to_compute_occupancy_grid.count() << std::endl;

  sdf_ = grid.ExtractSignedDistanceField(oob_value.occupancy, false, false).first;

  auto const t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> const time_to_compute_sdf = t2 - t1;
  std::cout << "Time to compute sdf: " << time_to_compute_sdf.count() << std::endl;

  auto const map_marker_msg = grid.ExportSurfacesForDisplay(collision_color, dont_draw_color, dont_draw_color);

  auto const t3 = std::chrono::steady_clock::now();
  sdf_tools::SignedDistanceField::GradientFunction gradient_function = [&](const int64_t x_index, const int64_t y_index,
                                                                           const int64_t z_index,
                                                                           const bool enable_edge_gradients = false) {
    return sdf_.GetGradient(x_index, y_index, z_index, enable_edge_gradients);
  };
  sdf_gradient_ = sdf_.GetFullGradient(gradient_function, true);

  auto const sdf_gradient_flat = [&]() {
    auto const &data = sdf_gradient_.GetImmutableRawData();
    std::vector<float> flat;
    for (auto const &d : data) {
      // only save the x/y currently
      flat.emplace_back(d[0]);
      flat.emplace_back(d[1]);
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
                            static_cast<unsigned long>(grid.GetNumYCells())};
  std::vector<size_t> gradient_shape{static_cast<unsigned long>(grid.GetNumXCells()),
                                     static_cast<unsigned long>(grid.GetNumYCells()), 2};

  std::vector<float> resolutions{msg->resolution, msg->resolution, msg->resolution};
  cnpy::npz_save(msg->filename, "sdf", &sdf_.GetImmutableRawData()[0], shape, "w");
  cnpy::npz_save(msg->filename, "sdf_gradient", &sdf_gradient_flat[0], gradient_shape, "a");
  cnpy::npz_save(msg->filename, "sdf_resolution", &resolutions[0], {2}, "a");

  ready_ = true;
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

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(CollisionMapPlugin)
