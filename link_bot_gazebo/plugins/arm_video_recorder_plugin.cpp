#include <geometry_msgs/Point32.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <cstring>

#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>

#include <ignition/math.hh>

#include <arm_video_recorder/TriggerVideoRecording.h>
#include "gazebo/gazebo.hh"
#include "gazebo/plugins/CameraPlugin.hh"

#define create_service_options(type, name, bind)                                                                       \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

namespace ph = std::placeholders;

namespace gazebo
{
class MyCameraPlugin : public CameraPlugin
{
private:
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::CallbackQueue queue_;
  ros::ServiceServer recording_srv_;
  std::thread ros_queue_thread_;

  gazebo::event::ConnectionPtr new_frame_connection_;
  gazebo::rendering::CameraPtr rendering_camera_;

  // Variables for saving video
  AVCodecContext *c = nullptr;
  AVFrame *frame;
  AVFrame *frame2;
  AVPacket pkt;
  FILE *file;
  struct SwsContext *sws_context = NULL;
  int64_t frame_number_ = 0;
  bool recording_ = false;

public:
  void Load(sensors::SensorPtr parent, sdf::ElementPtr sdf) override
  {
    CameraPlugin::Load(parent, sdf);

    if (!ros::isInitialized())
    {
      auto argc = 0;
      char **argv = nullptr;
      ros::init(argc, argv, "my_camera", ros::init_options::NoSigintHandler);
    }

    ros_node_ = std::make_unique<ros::NodeHandle>("gazebo_arm_video_recorder");

    auto record_bind = [this](arm_video_recorder::TriggerVideoRecordingRequest &req,
                              arm_video_recorder::TriggerVideoRecordingResponse &res) { return OnRecord(req, res); };
    auto record_so = create_service_options(arm_video_recorder::TriggerVideoRecording, "video_recorder", record_bind);
    recording_srv_ = ros_node_->advertiseService(record_so);

    ros_queue_thread_ = std::thread(std::bind(&MyCameraPlugin::QueueThread, this));

    rendering_camera_ = std::dynamic_pointer_cast<sensors::CameraSensor>(parent)->Camera();
    new_frame_connection_ = rendering_camera_->ConnectNewImageFrame(
        std::bind(&MyCameraPlugin::OnNewFrame, this, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5));

    ffmpeg_encoder_start(filename, AV_CODEC_ID_H264, 25, width, height);
  }

  void OnNewFrame(const unsigned char *image, unsigned int width, unsigned int height, unsigned int depth,
                  const std::string & /*format*/)
  {
    if (!recording_)
    {
      return;
    }
    frame->pts = frame_number_;
    auto const size = height * width * depth;
    gzlog << depth << " " << size << '\n';
    uint8_t nonconst_copy[size];
    memcpy(&nonconst_copy, image, size);
    // ffmpeg_encoder_encode_frame(&nonconst_copy[0]);
    frame_number_ += 1;
  }

  bool OnRecord(arm_video_recorder::TriggerVideoRecordingRequest &req,
                arm_video_recorder::TriggerVideoRecordingResponse &res)
  {
    if (req.record)
    {
      ROS_INFO_STREAM("Starting recording " << req.filename);
      recording_ = true;
    }
    else
    {
      recording_ = false;
      ROS_INFO_STREAM("Stopping recording.");
      ffmpeg_encoder_finish();
    }
    return true;
  }

  void QueueThread()
  {
    double constexpr timeout = 0.01;
    while (ros_node_->ok())
    {
      queue_.callAvailable(ros::WallDuration(timeout));
    }
  }

  ~MyCameraPlugin() override
  {
    queue_.clear();
    queue_.disable();
    ros_node_->shutdown();
    ros_queue_thread_.join();
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  void ffmpeg_encoder_init_frame(AVFrame **framep, int width, int height)
  {
    AVFrame *frame;
    frame = av_frame_alloc();
    if (!frame)
    {
      fprintf(stderr, "Could not allocate video frame\n");
      exit(1);
    }
    frame->format = c->pix_fmt;
    frame->width = width;
    frame->height = height;
    auto const ret = av_image_alloc(frame->data, frame->linesize, frame->width, frame->height,
                                    static_cast<AVPixelFormat>(frame->format), 32);
    if (ret < 0)
    {
      fprintf(stderr, "Could not allocate raw picture buffer\n");
      exit(1);
    }
    *framep = frame;
  }

  void ffmpeg_encoder_scale(uint8_t *rgb)
  {
    sws_context = sws_getCachedContext(sws_context, frame->width, frame->height, AV_PIX_FMT_YUV420P, frame2->width,
                                       frame2->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL);
    sws_scale(sws_context, (const uint8_t *const *)frame->data, frame->linesize, 0, frame->height, frame2->data,
              frame2->linesize);
  }

  void ffmpeg_encoder_set_frame_yuv_from_rgb(uint8_t *rgb)
  {
    const int in_linesize[1] = { 3 * frame->width };
    sws_context = sws_getCachedContext(sws_context, frame->width, frame->height, AV_PIX_FMT_RGB24, frame->width,
                                       frame->height, AV_PIX_FMT_YUV420P, 0, NULL, NULL, NULL);
    sws_scale(sws_context, (const uint8_t *const *)&rgb, in_linesize, 0, frame->height, frame->data, frame->linesize);
  }

  void ffmpeg_encoder_start(std::string const filename, AVCodecID codec_id, int fps, int width, int height)
  {
    AVCodec *codec;
    int ret;
    int width2 = width;
    int height2 = height;
    codec = avcodec_find_encoder(codec_id);
    if (!codec)
    {
      fprintf(stderr, "Codec not found\n");
      exit(1);
    }
    c = avcodec_alloc_context3(codec);
    if (!c)
    {
      fprintf(stderr, "Could not allocate video codec context\n");
      exit(1);
    }
    c->bit_rate = 400000;
    c->width = width2;
    c->height = height2;
    c->time_base.num = 1;
    c->time_base.den = fps;
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;
    if (codec_id == AV_CODEC_ID_H264)
      av_opt_set(c->priv_data, "preset", "slow", 0);
    if (avcodec_open2(c, codec, NULL) < 0)
    {
      fprintf(stderr, "Could not open codec\n");
      exit(1);
    }
    file = fopen(filename.c_str(), "wb");
    if (!file)
    {
      gzerr << "Could not open " << filename << " for recording\n";
      exit(1);
    }
    ffmpeg_encoder_init_frame(&frame, width, height);
    ffmpeg_encoder_init_frame(&frame2, width2, height2);
  }

  void ffmpeg_encoder_finish()
  {
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };
    int got_output, ret;
    do
    {
      fflush(stdout);
      ret = avcodec_encode_video2(c, &pkt, NULL, &got_output);
      if (ret < 0)
      {
        fprintf(stderr, "Error encoding frame\n");
        exit(1);
      }
      if (got_output)
      {
        fwrite(pkt.data, 1, pkt.size, file);
        av_packet_unref(&pkt);
      }
    } while (got_output);
    fwrite(endcode, 1, sizeof(endcode), file);
    fclose(file);
    avcodec_close(c);
    av_free(c);
    av_freep(&frame->data[0]);
    av_frame_free(&frame);
    av_freep(&frame2->data[0]);
    av_frame_free(&frame2);
  }

  void ffmpeg_encoder_encode_frame(uint8_t *rgb)
  {
    int ret, got_output;
    ffmpeg_encoder_set_frame_yuv_from_rgb(rgb);
    ffmpeg_encoder_scale(rgb);
    frame2->pts = frame->pts;
    av_init_packet(&pkt);
    pkt.data = nullptr;
    pkt.size = 0;
    ret = avcodec_encode_video2(c, &pkt, frame2, &got_output);
    if (ret < 0)
    {
      fprintf(stderr, "Error encoding frame\n");
      exit(1);
    }
    if (got_output)
    {
      fwrite(pkt.data, 1, pkt.size, file);
      av_packet_unref(&pkt);
    }
  }

};  // namespace gazebo

// Register this plugin with the simulator
GZ_REGISTER_SENSOR_PLUGIN(MyCameraPlugin)
}  // namespace gazebo