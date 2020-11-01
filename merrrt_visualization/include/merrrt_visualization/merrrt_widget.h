#pragma once

#include <peter_msgs/LabelStatus.h>
#include <peter_msgs/WorldControl.h>
#include <ros/ros.h>
#include <rviz/panel.h>
#include <rviz/rviz_export.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_srvs/SetBool.h>

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

  void LabelCallback(const peter_msgs::LabelStatus::ConstPtr &msg);
  void ErrorCallback(const std_msgs::Float32::ConstPtr &msg);
  void StdevCallback(const std_msgs::Float32::ConstPtr &msg);
  void OnAcceptProbability(const std_msgs::Float32::ConstPtr &msg);
  void OnRecoveryProbability(const std_msgs::Float32::ConstPtr &msg);
  void OnTrajIdx(const std_msgs::Float32::ConstPtr &msg);

  void load(const rviz::Config &config) override;
  void save(rviz::Config config) const override;
  void Settle();

private:
  Ui_MerrrtWidget ui;
  ros::NodeHandle ros_node_;
  ros::Subscriber label_sub_;
  ros::Subscriber error_sub_;
  ros::Subscriber stdev_sub_;
  ros::Subscriber traj_idx_sub_;
  ros::Subscriber recov_prob_sub_;
  ros::Subscriber accept_probability_sub_;
  ros::ServiceClient world_control_srv_;
};

}  // namespace merrrt_visualization
