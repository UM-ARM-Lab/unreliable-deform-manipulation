#pragma once

#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <rviz/panel.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int64.h>

#include <QObject>
#include <QWidget>
#include <thread>

#include "ui_mainwidget.h"

namespace merrrt_visualization
{
class RVizAnimationController : public rviz::Panel
{
  Q_OBJECT

public:
  explicit RVizAnimationController(QWidget *parent = nullptr);

  virtual ~RVizAnimationController();

  void TimeCallback(const std_msgs::Int64::ConstPtr &msg);

  void MaxTimeCallback(const std_msgs::Int64::ConstPtr &msg);

public slots:

  void ForwardClicked();

  void BackwardClicked();

  void PlayPauseClicked();

  void DoneClicked();

private:
  void QueueThread();

  Ui_MainWidget ui;
  ros::NodeHandle ros_node_;
  ros::Publisher fwd_pub_;
  ros::Publisher bwd_pub_;
  ros::Publisher play_pause_pub_;
  ros::Publisher done_pub_;
  ros::ServiceServer period_srv_;
  ros::ServiceServer auto_play_srv_;
  ros::Subscriber time_sub_;
  ros::Subscriber max_time_sub_;

  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
};

}  // namespace merrrt_visualization