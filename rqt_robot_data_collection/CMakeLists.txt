cmake_minimum_required(VERSION 3.0.2)
project(rqt_robot_data_collection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  rqt_gui
  rqt_gui_py
)

catkin_python_setup()

catkin_package(
 CATKIN_DEPENDS rospy rqt_gui rqt_gui_py
)
