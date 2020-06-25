#pragma once

#include <ros/ros.h>
#include <rviz/panel.h>
#include <rviz/rviz_export.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>

#include <QObject>
#include <QWidget>

#include "ui_merrrt_widget.h"

namespace merrrt_visualization
{
class MerrrtWidget : public rviz::Panel
{
  Q_OBJECT

public:
  explicit MerrrtWidget(QWidget *parent = nullptr);

  void BoolCallback(const std_msgs::Bool::ConstPtr &msg);
  void StdevCallback(const std_msgs::Float32::ConstPtr &msg);
  void OnAcceptProbability(const std_msgs::Float32::ConstPtr &msg);

  void load(const rviz::Config &config) override;
  void save(rviz::Config config) const override;

private:
  Ui_MerrrtWidget ui;
  ros::NodeHandle ros_node_;
  ros::Subscriber bool_sub_;
  ros::Subscriber stdev_sub_;
  ros::Subscriber accept_probability_sub_;
};

}  // namespace merrrt_visualization
