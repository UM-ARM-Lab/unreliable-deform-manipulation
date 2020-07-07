#include <merrrt_visualization/rviz_animation_controller.h>
#include <peter_msgs/GetBool.h>
#include <peter_msgs/GetFloat32.h>
#include <std_msgs/Empty.h>

#include <QApplication>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QMainWindow>
#include <QPushButton>
#include <iostream>

#define create_service_options(type, name, bind)                                                                       \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

namespace merrrt_visualization
{
RVizAnimationController::RVizAnimationController(QWidget *parent) : rviz::Panel(parent)
{
  ui.setupUi(this);
  connect(ui.forward_button, &QPushButton::clicked, this, &RVizAnimationController::ForwardClicked);
  connect(ui.backward_button, &QPushButton::clicked, this, &RVizAnimationController::BackwardClicked);
  connect(ui.play_pause_button, &QPushButton::clicked, this, &RVizAnimationController::PlayPauseClicked);
  connect(ui.done_button, &QPushButton::clicked, this, &RVizAnimationController::DoneClicked);

  // highlight_animation_ = new QPropertyAnimation(ui, "background-color");
  // highlight_animation_->setDuration(1000);
  // highlight_animation_->setStartValue("#ffff00");
  // highlight_animation_->setEndValue("#ffffff");

  fwd_pub_ = ros_node_.advertise<std_msgs::Empty>("rviz_anim/forward", 10);
  bwd_pub_ = ros_node_.advertise<std_msgs::Empty>("rviz_anim/backward", 10);
  play_pause_pub_ = ros_node_.advertise<std_msgs::Empty>("rviz_anim/play_pause", 10);
  done_pub_ = ros_node_.advertise<std_msgs::Empty>("rviz_anim/done", 10);

  auto period_bind = [this](peter_msgs::GetFloat32Request &req, peter_msgs::GetFloat32Response &res) {
    res.data = static_cast<float>(ui.period_spinbox->value());
    return true;
  };
  auto period_so = create_service_options(peter_msgs::GetFloat32, "rviz_anim/period", period_bind);
  period_srv_ = ros_node_.advertiseService(period_so);

  auto auto_play_bind = [this](peter_msgs::GetBoolRequest &req, peter_msgs::GetBoolResponse &res) {
    res.data = ui.auto_play_checkbox->isChecked();
    return true;
  };
  auto auto_play_so = create_service_options(peter_msgs::GetBool, "rviz_anim/auto_play", auto_play_bind);
  auto_play_srv_ = ros_node_.advertiseService(auto_play_so);

  // this is stupid why must I list this type here but not when I do this for services!?
  boost::function<void(const std_msgs::Int64::ConstPtr &)> time_cb = [this](const std_msgs::Int64::ConstPtr &msg) {
    TimeCallback(msg);
  };
  auto time_sub_so = ros::SubscribeOptions::create("rviz_anim/time", 10, time_cb, ros::VoidPtr(), &queue_);
  time_sub_ = ros_node_.subscribe(time_sub_so);

  // this is stupid why must I list this type here but not when I do this for services!?
  boost::function<void(const std_msgs::Int64::ConstPtr &)> max_time_cb = [this](const std_msgs::Int64::ConstPtr &msg) {
    MaxTimeCallback(msg);
  };
  auto max_time_sub_so = ros::SubscribeOptions::create("rviz_anim/max_time", 10, max_time_cb, ros::VoidPtr(), &queue_);
  max_time_sub_ = ros_node_.subscribe(max_time_sub_so);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
}

RVizAnimationController::~RVizAnimationController()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  ros_queue_thread_.join();
}

void RVizAnimationController::TimeCallback(const std_msgs::Int64::ConstPtr &msg)
{
  QString text;
  text.sprintf("%3ld", msg->data);
  ui.step_number_label->setText(text);
  emit update();
}

void RVizAnimationController::MaxTimeCallback(const std_msgs::Int64::ConstPtr &msg)
{
  auto const text = QString::number(msg->data);
  ui.max_step_number_label->setText(text);
  emit update();
}

void RVizAnimationController::DoneClicked()
{
  done_pub_.publish(std_msgs::Empty());
}

void RVizAnimationController::ForwardClicked()
{
  fwd_pub_.publish(std_msgs::Empty());
}

void RVizAnimationController::BackwardClicked()
{
  bwd_pub_.publish(std_msgs::Empty());
}

void RVizAnimationController::PlayPauseClicked()
{
  play_pause_pub_.publish(std_msgs::Empty());
}

void RVizAnimationController::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

}  // namespace merrrt_visualization

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(merrrt_visualization::RVizAnimationController, rviz::Panel)