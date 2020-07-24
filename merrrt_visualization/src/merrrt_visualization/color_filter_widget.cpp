#include <merrrt_visualization/color_filter_widget.h>

#include <iostream>

namespace merrrt_visualization
{
ColorFilterWidget::ColorFilterWidget(QWidget *parent)
  : rviz::Panel(parent)
{
  ui.setupUi(this);
  connect(ui.mask1_lower_h, &QSlider::sliderMoved, this, &ColorFilterWidget::OnMask1LowerHMoved);
}

void ColorFilterWidget::OnMask1LowerHMoved(int const position)
{
  ros::param::set("mask1_lower_h", static_cast<float>(position));
}

void ColorFilterWidget::OnMask1LowerSMoved(int const position)
{
  ros::param::set("mask1_lower_s", static_cast<float>(position) / 10.f);
}

void ColorFilterWidget::OnMask1LowerVMoved(int const position)
{
  ros::param::set("mask1_lower_v", static_cast<float>(position) / 10.f);
}

void ColorFilterWidget::OnMask1UpperHMoved(int const position)
{
  ros::param::set("mask1_upper_h", static_cast<float>(position));
}

void ColorFilterWidget::OnMask1UpperSMoved(int const position)
{
  ros::param::set("mask1_upper_s", static_cast<float>(position) / 10.f);
}

void ColorFilterWidget::OnMask1UpperVMoved(int const position)
{
  ros::param::set("mask1_upper_v", static_cast<float>(position) / 10.f);
}

void ColorFilterWidget::OnMask2LowerHMoved(int const position)
{
  ros::param::set("mask2_lower_h", static_cast<float>(position));
}

void ColorFilterWidget::OnMask2LowerSMoved(int const position)
{
  ros::param::set("mask2_lower_s", static_cast<float>(position) / 10.f);
}

void ColorFilterWidget::OnMask2LowerVMoved(int const position)
{
  ros::param::set("mask2_lower_v", static_cast<float>(position) / 10.f);
}

void ColorFilterWidget::OnMask2UpperHMoved(int const position)
{
  ros::param::set("mask2_upper_h", static_cast<float>(position));
}

void ColorFilterWidget::OnMask2UpperSMoved(int const position)
{
  ros::param::set("mask2_upper_s", static_cast<float>(position) / 10.f);
}

void ColorFilterWidget::OnMask2UpperVMoved(int const position)
{
  ros::param::set("mask2_upper_v", static_cast<float>(position) / 10.f);
}

void ColorFilterWidget::load(const rviz::Config &config)
{
  rviz::Panel::load(config);
}
void ColorFilterWidget::save(rviz::Config config) const
{
  rviz::Panel::save(config);
}

}  // namespace merrrt_visualization

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(merrrt_visualization::ColorFilterWidget, rviz::Panel)
