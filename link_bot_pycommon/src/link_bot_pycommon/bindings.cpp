#include <pybind11/pybind11.h>
#include <ros/console.h>
#include <iostream>
#include <std_msgs/Int64.h>
#include <ros/ros.h>

namespace py = pybind11;

void cb(std_msgs::Int64ConstPtr const &msg)
{
    ROS_INFO_STREAM("received " << msg->data);
}

struct PyCppRosSub
{
    PyCppRosSub() {
        ros::NodeHandle nh;
        auto const topic_name = "test_topic";
        ROS_INFO_STREAM("topic name " << topic_name);
        auto pub = nh.advertise<std_msgs::Int64>(topic_name, 10, false);
        ROS_INFO_STREAM("resolved topic name " << nh.resolveName(topic_name));
        std_msgs::Int64 msg;
        for (auto i = 0; i < 10; ++i)
        {
            msg.data = i;
            pub.publish(msg);
            sleep(1);
        }
    }
};

PYBIND11_MODULE(pycpp_ros_sub, m)
{
  py::class_<PyCppRosSub>(m, "PyCppRosSub")
      .def(py::init<>());
}
