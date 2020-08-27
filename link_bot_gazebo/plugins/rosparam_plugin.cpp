#include <fstream>

#include <experimental/filesystem>
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

#include <gazebo/common/Plugin.hh>

namespace gazebo
{
  class ROSParamPlugin : public WorldPlugin
  {
  public:
    void Load(physics::WorldPtr /*world*/, sdf::ElementPtr sdf) override
    {
      // setup ROS stuff
      if (!ros::isInitialized())
      {
        int argc = 0;
        ros::init(argc, nullptr, "rosparam_plugin", ros::init_options::NoSigintHandler);
      }

      if (!sdf->HasElement("yaml_file"))
      {
        ROS_WARN("No element yaml_file, this plugin will do nothing.");
      } else
      {
        auto const yaml_filename = sdf->GetElement("yaml_file")->Get<std::string>();
        std::experimental::filesystem::path const yaml_path(yaml_filename);
        if (!std::experimental::filesystem::exists(yaml_path))
        {
          ROS_ERROR_STREAM("YAML File " << yaml_filename << " not found.");
          return;
        }

//        std::experimental::filesystem::path const sdf_file_path(sdf->FilePath());

        try
        {
          YAML::Node params = YAML::LoadFile(yaml_filename);
          for (auto it = params.begin(); it != params.end(); ++it)
          {
            std::cout << it->Type() << "\n";
          }
        } catch (const YAML::ParserException &ex)
        {
          std::cout << ex.what() << std::endl;
        }


//        while (parser.Get(doc))
//        {
//          ROS_WARN_STREAM("type " << doc.Type() << " value " << doc);
//          if (doc.IsMap())
//          {
//            for (auto it = doc.begin(); it != doc.end(); ++it)
//            {
//              std::string key, value;
//              it.first() >> key;
//              it.second() >> value;
//              std::cout << "Key: " << key << ", value: " << value << std::endl;
//            }
//          } else
//          {
//            ROS_WARN_STREAM("Skipping yaml node " << doc << " because it's not a map type.")
//          }
//        }
      }

      ph_ = std::make_unique<ros::NodeHandle>("~");
      nh_ = std::make_unique<ros::NodeHandle>();

      ROS_INFO("Finished loading ROSParam plugin!\n");

    }

  private:
    std::unique_ptr <ros::NodeHandle> ph_;
    std::unique_ptr <ros::NodeHandle> nh_;
  };

  GZ_REGISTER_WORLD_PLUGIN(ROSParamPlugin)

}  // namespace gazebo