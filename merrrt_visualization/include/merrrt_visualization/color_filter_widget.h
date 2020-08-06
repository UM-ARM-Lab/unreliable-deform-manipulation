#pragma once

#include <ros/ros.h>
#include <rviz/panel.h>
#include <rviz/rviz_export.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>

#include <QObject>
#include <QWidget>

#include "ui_color_filter_widget.h"

namespace merrrt_visualization
{
class ColorFilterWidget : public rviz::Panel
{
  Q_OBJECT

public:
  explicit ColorFilterWidget(QWidget *parent = nullptr);

  void load(const rviz::Config &config) override;
  void save(rviz::Config config) const override;

public slots:
  void OnMask1LowerHMoved(int position);
  void OnMask1LowerSMoved(int position);
  void OnMask1LowerVMoved(int position);

  void OnMask1UpperHMoved(int position);
  void OnMask1UpperSMoved(int position);
  void OnMask1UpperVMoved(int position);

  void OnMask2LowerHMoved(int position);
  void OnMask2LowerSMoved(int position);
  void OnMask2LowerVMoved(int position);

  void OnMask2UpperHMoved(int position);
  void OnMask2UpperSMoved(int position);
  void OnMask2UpperVMoved(int position);

private:
  Ui_ColorFilterWidget ui;
};

}  // namespace merrrt_visualization
