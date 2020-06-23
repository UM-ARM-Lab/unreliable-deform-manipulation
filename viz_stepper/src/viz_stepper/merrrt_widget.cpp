#include <viz_stepper/merrrt_widget.h>

#include <iostream>

namespace merrrt_widget
{

MerrrtWidget::MerrrtWidget(QWidget *parent) : rviz::Panel(parent)
{
  ui.setupUi(this);
  bool_sub_ = ros_node_.subscribe<std_msgs::Bool>("mybool", 10, &MerrrtWidget::BoolCallback, this);
  accept_probability_sub_ = ros_node_.subscribe<std_msgs::Float32>("accept_probability_viz", 10, &MerrrtWidget::OnAcceptProbability, this);
}

void MerrrtWidget::BoolCallback(const std_msgs::Bool::ConstPtr &msg)
{
  if (msg->data) {
    ui.bool_indicator->setStyleSheet("background-color: rgb(0, 250, 0);");
  }
  else {
    ui.bool_indicator->setStyleSheet("background-color: rgb(250, 0, 0);");
  }
}
void MerrrtWidget::OnAcceptProbability(const std_msgs::Float32::ConstPtr &msg)
{
    ui.accept_probability->setText(QString::number(msg->data));
}

void MerrrtWidget::load(const rviz::Config &config) {
  rviz::Panel::load(config);
}
void MerrrtWidget::save(rviz::Config config) const {
  rviz::Panel::save(config);
}

} // namespace merrrt_widget

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(merrrt_widget::MerrrtWidget, rviz::Panel)
