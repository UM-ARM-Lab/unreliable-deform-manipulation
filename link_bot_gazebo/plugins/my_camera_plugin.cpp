#include "gazebo/gazebo.hh"
#include "gazebo/plugins/CameraPlugin.hh"
#include "gazebo/rendering/Camera.hh"

namespace gazebo {
class MyCameraPlugin : public CameraPlugin {
 public:
  MyCameraPlugin() : CameraPlugin() {}

  void Load(sensors::SensorPtr parent, sdf::ElementPtr sdf) override { CameraPlugin::Load(parent, sdf); }

    void OnNewFrame(const unsigned char *image, unsigned int width, unsigned int height, unsigned int depth,
                    const std::string &format) override
    {
      auto const size = rendering::Camera::ImageByteSize(width, height, format);
//      memcpy(latest_image_, image, size);
    }

  unsigned char const *GetLatestImage() const { return this->camera->ImageData(); }

 private:
  unsigned char *latest_image_;
};

// Register this plugin with the simulator
GZ_REGISTER_SENSOR_PLUGIN(MyCameraPlugin)
}