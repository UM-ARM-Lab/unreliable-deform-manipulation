#include <merrrt_visualization/merrrt_widget.h>

#include <iostream>

namespace merrrt_visualization
{
MerrrtWidget::MerrrtWidget(QWidget *parent) : rviz::Panel(parent)
{
  ui.setupUi(this);
  bool_sub_ = ros_node_.subscribe<std_msgs::Bool>("mybool", 10, &MerrrtWidget::BoolCallback, this);
  stdev_sub_ = ros_node_.subscribe<std_msgs::Float32>("stdev", 10, &MerrrtWidget::StdevCallback, this);
  accept_probability_sub_ =
      ros_node_.subscribe<std_msgs::Float32>("accept_probability_viz", 10, &MerrrtWidget::OnAcceptProbability, this);
  traj_idx_sub_ = ros_node_.subscribe<std_msgs::Float32>("traj_idx_viz", 10, &MerrrtWidget::OnTrajIdx, this);
}

void MerrrtWidget::OnTrajIdx(const std_msgs::Float32::ConstPtr &msg)
{
  ui.traj_idx->setText(QString::number(msg->data));
}
void MerrrtWidget::StdevCallback(const std_msgs::Float32::ConstPtr &msg)
{
  ui.stdev_label->setText(QString::number(msg->data));
}
void MerrrtWidget::BoolCallback(const std_msgs::Bool::ConstPtr &msg)
{
  if (msg->data)
  {
    ui.bool_indicator->setStyleSheet("background-color: rgb(0, 250, 0);");
  }
  else
  {
    ui.bool_indicator->setStyleSheet("background-color: rgb(250, 0, 0);");
  }
}

void MerrrtWidget::OnAcceptProbability(const std_msgs::Float32::ConstPtr &msg)
{
  auto const blue = 50;
  auto red = 0;
  auto green = 0;
  if (msg->data >= 0 and msg->data <= 1)
  {
    // *0.8 to cool the colors
    auto const cool_factor = 0.7;
    red = static_cast<int>(255 * (1 - msg->data) * cool_factor);
    green = static_cast<int>(255 * msg->data * cool_factor);
  }
  else
  {
    red = 0;
    green = 0;
  }
  ui.accept_probability->setStyleSheet(QString("color: rgb(%1, %2, %3);").arg(red).arg(green).arg(blue));
  ui.accept_probability->setText(QString::number(msg->data));
}

void MerrrtWidget::load(const rviz::Config &config)
{
  rviz::Panel::load(config);
}
void MerrrtWidget::save(rviz::Config config) const
{
  rviz::Panel::save(config);
}

}  // namespace merrrt_visualization

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(merrrt_visualization::MerrrtWidget, rviz::Panel)
