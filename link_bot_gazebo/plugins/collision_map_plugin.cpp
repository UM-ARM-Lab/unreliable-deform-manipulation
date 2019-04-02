#include <link_bot_gazebo/WriteSDF.h>
#include <visualization_msgs/Marker.h>
#include <sdf_tools/collision_map.hpp>
#include <std_msgs/ColorRGBA.h>
#include <sdf_tools/SDF.h>
#include <arc_utilities/arc_helpers.hpp>

#include "collision_map_plugin.h"

using namespace gazebo;

const sdf_tools::COLLISION_CELL CollisionMapPlugin::oob_value{-10000};
const sdf_tools::COLLISION_CELL CollisionMapPlugin::occupied_value{1};

void CollisionMapPlugin::Load(physics::WorldPtr world, sdf::ElementPtr _sdf) {
    auto engine = world->Physics();
    engine->InitForThread();
    auto ray_shape = engine->CreateShape("ray", gazebo::physics::CollisionPtr());
    ray = boost::dynamic_pointer_cast<gazebo::physics::RayShape>(ray_shape);

    if (!ros::isInitialized()) {
        auto argc = 0;
        char **argv = nullptr;
        ros::init(argc, argv, "collision_map_plugin", ros::init_options::NoSigintHandler);
    }

    ros_node_ = std::make_unique<ros::NodeHandle>("collision_map_plugin");

    auto bind = boost::bind(&CollisionMapPlugin::OnWriteSDF, this, _1);
    auto so = ros::SubscribeOptions::create<link_bot_gazebo::WriteSDF>("/write_sdf", 1, bind, ros::VoidPtr(),
                                                                       &queue_);

    gazebo_sdf_viz_pub_ = ros_node_->advertise<visualization_msgs::MarkerArray>("gazebo_sdf_viz", 1);

    sub_ = ros_node_->subscribe(so);
    ros_queue_thread_ = std::thread(std::bind(&CollisionMapPlugin::QueueThread, this));
}

void CollisionMapPlugin::OnWriteSDF(link_bot_gazebo::WriteSDFConstPtr msg) {
    Eigen::Isometry3d origin_transform{};
    origin_transform.translation() = Eigen::Vector3d{msg->center.x - msg->x_width / 2,
                                                     msg->center.y - msg->y_height / 2, 0};
    sdf_tools::CollisionMapGrid grid{origin_transform, "/gazebo_world", msg->resolution,
                                     msg->x_width, msg->y_height, msg->max_z, oob_value};
    ignition::math::Vector3d start, end;
    start.Z(msg->max_z);
    end.Z(0.001);

    // parameters needed for the GetIntersection check
    std::string entityName;
    double dist;

    for (auto x_idx{0l}; x_idx < grid.GetNumXCells(); ++x_idx) {
        for (auto y_idx{0l}; y_idx < grid.GetNumYCells(); ++y_idx) {
            auto const grid_location = grid.GridIndexToLocation(x_idx, y_idx, 0);
            start.X(grid_location(0));
            end.X(grid_location(0));
            start.Y(grid_location(1));
            end.Y(grid_location(1));
            ray->SetPoints(start, end);
            ray->GetIntersection(dist, entityName);
            if (!entityName.empty()) {
                grid.SetValue(x_idx, y_idx, 0, occupied_value);
            }
        }
    }

    std::cout << "Computing SDF..." << std::endl;
    auto const dont_draw_color = arc_helpers::GenerateUniqueColor<std_msgs::ColorRGBA>(0u);
    auto const collision_color = arc_helpers::GenerateUniqueColor<std_msgs::ColorRGBA>(1u);
    auto const sdf = grid.ExtractSignedDistanceField(oob_value.occupancy, false, false).first;
    auto const map_marker_msg = grid.ExportSurfacesForDisplay(collision_color, dont_draw_color, dont_draw_color);

    gazebo_sdf_viz_pub_.publish(map_marker_msg);
}

CollisionMapPlugin::~CollisionMapPlugin() {
    queue_.clear();
    queue_.disable();
    ros_node_->shutdown();
    ros_queue_thread_.join();
}

void CollisionMapPlugin::QueueThread() {
    double constexpr timeout = 0.01;
    while (ros_node_->ok()) {
        queue_.callAvailable(ros::WallDuration(timeout));
    }
}

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(CollisionMapPlugin)
