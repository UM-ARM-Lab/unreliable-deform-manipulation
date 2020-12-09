/*
 * Copyright 2014 Open Source Robotics Foundation
 * Copyright 2015 Clearpath Robotics
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

#pragma once

#include <string>
#include <vector>
#include <thread>

#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>

#include <robotiq_3f_gripper_articulated_msgs/Robotiq3FGripperRobotInput.h>
#include <robotiq_3f_gripper_articulated_msgs/Robotiq3FGripperRobotOutput.h>

constexpr auto const PLUGIN_LOG_NAME = "robotiq_hand_plugin";

/// \brief A plugin that implements the Robotiq 3-Finger Adaptative Gripper.
/// The plugin exposes the next parameters via SDF tags:
///   * <side> Determines if we are controlling the left or right hand. This is
///            a required parameter and the allowed values are 'left' or 'right'
///   * <kp_position> P gain for the PID that controls the position
///                   of the joints. This parameter is optional.
///   * <ki_position> I gain for the PID that controls the position
///                   of the joints. This parameter is optional.
///   * <kd_position> D gain for the PID that controls the position
///                   of the joints. This parameter is optional.
///   * <position_effort_min> Minimum output of the PID that controls the
///                           position of the joints. This parameter is optional
///   * <position_effort_max> Maximum output of the PID that controls the
///                           position of the joints. This parameter is optional
///   * <topic_command> ROS topic name used to send new commands to the hand.
///                     This parameter is optional.
///   * <topic_state> ROS topic name used to receive state from the hand.
///                   This parameter is optional.

class RobotiqControl
{

  /// \brief Hand states.
  enum State
  {
    Disabled = 0,
    Emergency,
    ICS,
    ICF,
    ChangeModeInProgress,
    Simplified
  };

  /// \brief Different grasping modes.
  enum GraspingMode
  {
    Basic = 0,
    Pinch,
    Wide,
    Scissor
  };

 public:
  explicit RobotiqControl(gazebo::physics::ModelPtr model_,
                          double kp,
                          double ki,
                          double kd,
                          std::string side);

  /// \brief ROS topic callback to update Robotiq Hand Control Commands.
  /// \param[in] _msg Incoming ROS message with the next hand command.
  void SetHandleCommand(robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput const &_msg);

  /// \brief Update PID Joint controllers.
  /// \param[in] _dt time step size since last update.
  void UpdatePIDControl(double _dt);

  /// \brief Publish Robotiq Hand state.
  robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotInput GetHandleState();

  /// \brief Update the controller.
  void UpdateStates();

  /// \brief Grab pointers to all the joints.
  /// \return true on success, false otherwise.
  bool FindJoints();

  /// \brief Fully open the hand at half of the maximum speed.
  void ReleaseHand();

  /// \brief Stop the fingers.
  void StopHand();

  /// \brief Checks if the hand is fully open.
  /// return True when all the fingers are fully open or false otherwise.
  bool IsHandFullyOpen();

  /// \brief Internal helper to get the object detection value.
  /// \param[in] _joint Finger joint.
  /// \param[in] _index Index of the position PID for this joint.
  /// \param[in] _rPR Current position request.
  /// \param[in] _prevrPR Previous position request.
  /// \return The information on possible object contact:
  /// 0 Finger is in motion (only meaningful if gGTO = 1).
  /// 1 Finger has stopped due to a contact while opening.
  /// 2 Finger has stopped due to a contact while closing.
  /// 3 Finger is at the requested position.
  uint8_t GetObjectDetection(const gazebo::physics::JointPtr &_joint,
                             int _index, uint8_t _rPR, uint8_t _prevrPR);

  /// \brief Internal helper to get the actual position of the finger.
  /// \param[in] _joint Finger joint.
  /// \return The actual position of the finger. 0 is the minimum position
  /// (fully open) and 255 is the maximum position (fully closed).
  uint8_t GetCurrentPosition(const gazebo::physics::JointPtr &_joint) const;

  /// \brief Internal helper to reduce code duplication. If the joint name is
  /// found, a pointer to the joint is added to a vector of joint pointers.
  /// \param[in] _jointName Joint name.
  /// \param[out] _joints Vector of joint pointers.
  /// \return True when the joint was found or false otherwise.
  bool GetAndPushBackJoint(const std::string &_jointName,
                           gazebo::physics::Joint_V &_joints) const;

  /// \brief Verify that one command field is within the correct range.
  /// \param[in] _label Label of the field. E.g.: rACT, rMOD.
  /// \param[in] _min Minimum value.
  /// \param[in] _max Maximum value.
  /// \param[in] _v Value to be verified.
  /// \return True when the value is within the limits or false otherwise.
  static bool VerifyField(const std::string &_label, int _min, int _max, int _v);

  /// \brief Verify that all the command fields are within the correct range.
  /// \param[in] _command Robot output message.
  /// \return True if all the fields are withing the correct range or false
  /// otherwise.
  static bool VerifyCommand(robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput const &command);

  /// \brief Number of joints in the hand.
  /// The three fingers can do abduction/adduction.
  /// Fingers 1 and 2 can do circumduction in one axis.
  static const int NumJoints = 5;

  /// \brief Velocity tolerance. Below this value we assume that the joint is
  /// stopped (rad/s).
  static constexpr double VelTolerance = 0.002;

  /// \brief Position tolerance. If the difference between target position and
  /// current position is within this value we'll conclude that the joint
  /// reached its target (rad).
  static constexpr double PoseTolerance = 0.002;

  /// \brief Min. joint speed (rad/s). Finger is 125mm and tip speed is 22mm/s.
  static constexpr double MinVelocity = 0.176;

  /// \brief Max. joint speed (rad/s). Finger is 125mm and tip speed is 110mm/s.
  static constexpr double MaxVelocity = 0.88;

  /// \brief Default topic name for sending control updates to the left hand.
  static const std::string DefaultLeftTopicCommand;

  /// \brief Default topic name for receiving state updates from the left hand.
  static const std::string DefaultLeftTopicState;

  /// \brief Default topic name for sending control updates to the right hand.
  static const std::string DefaultRightTopicCommand;

  /// \brief Default topic name for receiving state updates from the right hand.
  static const std::string DefaultRightTopicState;

  /// \brief HandleControl message. Originally published by user but some of the
  /// fields might be internally modified. E.g.: When releasing the hand for
  // changing the grasping mode.
  robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput handleCommand;

  /// \brief HandleControl message. Last command received before changing the
  /// grasping mode.
  robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput lastHandleCommand;

  /// \brief Previous command received. We know if the hand is opening or
  /// closing by comparing the current command and the previous one.
  robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput prevCommand;

  /// \brief Original HandleControl message (published by user and unmodified).
  robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput userHandleCommand;

  /// \brief gazebo world update connection.
  gazebo::event::ConnectionPtr updateConnection;

  /// \brief keep track of controller update sim-time.
  gazebo::common::Time lastControllerUpdateTime;

  /// \brief Robotiq Hand State.
  robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotInput handleState;

  /// \brief Controller update mutex.
  boost::mutex controlMutex;

  /// \brief Grasping mode.
  GraspingMode graspingMode;

  /// \brief Hand state.
  State handState;

  /// \brief World pointer.
  gazebo::physics::WorldPtr world_;

  /// \brief Parent model of the hand.
  gazebo::physics::ModelPtr model_;

  /// \brief Used to select between 'left' or 'right' hand.
  std::string side_;

  /// \brief Vector containing all the actuated finger joints.
  gazebo::physics::Joint_V fingerJoints;

  /// \brief Vector containing all the joints.
  gazebo::physics::Joint_V joints;

  /// \brief PIDs used to control the finger positions.
  gazebo::common::PID posePID[NumJoints];
};
