#include "collision_map_plugin.h"

#include <std_msgs/ColorRGBA.h>
#include <std_msgs/MultiArrayDimension.h>
#include <visualization_msgs/Marker.h>

#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>

using namespace gazebo;

const sdf_tools::COLLISION_CELL CollisionMapPlugin::oob_value{-10000};
const sdf_tools::COLLISION_CELL CollisionMapPlugin::occupied_value{1};
const sdf_tools::COLLISION_CELL CollisionMapPlugin::unoccupied_value{0};

void CollisionMapPlugin::Load(physics::WorldPtr world, sdf::ElementPtr _sdf)
{
  this->world_ = world;
  this->engine_ = world->Physics();

  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "collision_map_plugin", ros::init_options::NoSigintHandler);
  }

  auto get_occupancy = [&](peter_msgs::ComputeOccupancyRequest &req, peter_msgs::ComputeOccupancyResponse &res) {
    compute_occupancy_grid(req.h_rows, req.w_cols, req.c_channels, req.center, req.resolution, req.robot_name);

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
    std_msgs::MultiArrayDimension channel_dim;
    channel_dim.label = "channel";
    channel_dim.size = grid_.GetNumZCells();
    channel_dim.stride = 1;
    return true;
  };

  ros_node_ = std::make_unique<ros::NodeHandle>("collision_map_plugin");

  {
    auto so = ros::AdvertiseServiceOptions::create<peter_msgs::ComputeOccupancy>("/occupancy", get_occupancy,
                                                                                 ros::VoidConstPtr(), &queue_);
    get_occupancy_service_ = ros_node_->advertiseService(so);
  }

  gzlog << "Finished loading collision map plugin!\n";
  ros_queue_thread_ = std::thread([this] { QueueThread(); });
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

void nearCallback(void *data, dGeomID _o1, dGeomID _o2)
{
  dContactGeom contact;

  if (dGeomIsSpace(_o1) || dGeomIsSpace(_o2)) {
    dSpaceCollide2(_o1, _o2, nullptr, &nearCallback);
  }
  else {
    int n = dCollide(_o1, _o2, 1, &contact, sizeof(contact));
    if (n > 0) {
      std::cout << contact.g1 << " " << contact.g2 << "\n";
    }
  }
}

void CollisionMapPlugin::compute_occupancy_grid(int64_t h_rows, int64_t w_cols, int64_t c_channels,
                                                geometry_msgs::Point center, float resolution,
                                                std::string const &robot_name, bool verbose)
{
  auto const x_width = resolution * w_cols;
  auto const y_height = resolution * h_rows;
  auto const z_size = resolution * c_channels;
  Eigen::Isometry3d origin_transform = Eigen::Isometry3d::Identity();
  origin_transform.translation() =
      Eigen::Vector3d{center.x - x_width / 2, center.y - y_height / 2, center.z - z_size / 2};

  grid_ = sdf_tools::CollisionMapGrid(origin_transform, "/world", resolution, w_cols, h_rows, c_channels, oob_value);

  auto const t0 = std::chrono::steady_clock::now();
  std::string entityName;
  double dist{0};

  // lock physics engine will creating/testing collision
  {
    boost::recursive_mutex::scoped_lock lock(*engine_->GetPhysicsUpdateMutex());
    engine_->InitForThread();

    physics::BasePtr basePtr;
    basePtr.reset(new physics::Base(physics::BasePtr()));
    basePtr->SetName("world_root_element");
    basePtr->SetWorld(world_);

    physics::ModelPtr m(new physics::Model(basePtr));
    m->SetName("test_ray");
    auto l = engine_->CreateLink(m);
    auto ode_c = boost::make_shared<physics::ODECollision>(l);
    auto s = boost::make_shared<physics::ODERayShape>(ode_c);

    ignition::math::Vector3d start, end;
    start.Z(0.1);
    end.Z(0.01);
    start.X(0.1);
    start.Y(0);
    end.X(1);
    end.Y(0);
    s->SetPoints(start, end);

    ode_c->SetShape(s);
    s->SetWorld(world_);

    auto ode = boost::dynamic_pointer_cast<physics::ODEPhysics>(engine_);

    dSpaceCollide2(ode_c->GetCollisionId(), (dGeomID)(ode->GetSpaceId()), nullptr, &nearCallback);

    //    world_->RemoveModel(m);

    //    auto shape = engine->CreateShape("ray", physics::CollisionPtr());

    //    for (auto x_idx{0l}; x_idx < grid_.GetNumXCells(); ++x_idx) {
    //      for (auto y_idx{0l}; y_idx < grid_.GetNumYCells(); ++y_idx) {
    //        for (auto z_idx{0l}; z_idx < grid_.GetNumZCells(); ++z_idx) {
    //          auto const grid_location = grid_.GridIndexToLocation(x_idx, y_idx, z_idx);
    //          ignition::math::Vector3d start, end;
    //          start.X(grid_location(0));
    //          start.Y(grid_location(1));
    //          start.Z(5.0);
    //          end.X(grid_location(0));
    //          end.Y(grid_location(1));
    //          end.Z(grid_location(2));
    //
    //          ray->SetPoints(start, end);
    //
    //          ray->GetIntersection(dist, entityName);
    //          if (not entityName.empty() and (robot_name.empty() or entityName.find(robot_name) != 0)) {
    //            grid_.SetValue(x_idx, y_idx, z_idx, occupied_value);
    //          }
    //          else {
    //            grid_.SetValue(x_idx, y_idx, z_idx, unoccupied_value);
    //          }
    //        }
    //      }
    //    }
  }

  auto const t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> const time_to_compute_occupancy_grid = t1 - t0;
  if (verbose) {
    gzlog << "Time to compute occupancy grid_: " << time_to_compute_occupancy_grid.count() << std::endl;
  }
}

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(CollisionMapPlugin)
