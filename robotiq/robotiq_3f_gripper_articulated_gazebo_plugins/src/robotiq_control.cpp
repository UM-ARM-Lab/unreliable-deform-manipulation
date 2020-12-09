/*
 * Copyright 2014 Open Source Robotics Foundation
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
/*
    This file has been modified from the original, by Devon Ash, then further by Peter Mitrano
*/

#include <ros/console.h>

#include <robotiq_3f_gripper_articulated_gazebo_plugins/robotiq_control.h>

////////////////////////////////////////////////////////////////////////////////
RobotiqControl::RobotiqControl(gazebo::physics::ModelPtr model,
                               double const kp,
                               double const ki,
                               double const kd,
                               std::string side) : world_(model->GetWorld()), model_(model), side_(side)
{
  // PID default parameters.
  for (auto &i : posePID)
  {
    i.Init(kp, ki, kd, 0.0, 0.0, 60.0, -60.0);
    i.SetCmd(0.0);
  }

  // Default grasping mode: Basic mode.
  graspingMode = Basic;

  // Load the vector of all joints.
  std::string prefix;
  if (side == "left")
  {
    prefix = "l_";
  } else
  {
    prefix = "r_";
  }

  // Load the vector of all joints.
  if (!FindJoints())
  {
    return;
  }

  // Controller time control.
  lastControllerUpdateTime = world_->SimTime();

  // Log information.
  for (int i = 0; i < NumJoints; ++i)
  {
    ROS_DEBUG_STREAM("Position PID parameters for joint ["
                         << fingerJoints[i]->GetName() << "]:" << std::endl
                         << "\tKP: " << posePID[i].GetPGain() << std::endl
                         << "\tKI: " << posePID[i].GetIGain() << std::endl
                         << "\tKD: " << posePID[i].GetDGain() << std::endl
                         << "\tIMin: " << posePID[i].GetIMin() << std::endl
                         << "\tIMax: " << posePID[i].GetIMax() << std::endl
                         << "\tCmdMin: " << posePID[i].GetCmdMin() << std::endl
                         << "\tCmdMax: " << posePID[i].GetCmdMax() << std::endl
    );
  }
}


////////////////////////////////////////////////////////////////////////////////
bool RobotiqControl::VerifyField(const std::string &_label, int _min,
                                 int _max, int _v)
{
  if (_v < _min || _v > _max)
  {
    std::cerr << "Illegal " << _label << " value: [" << _v << "]. The correct "
              << "range is [" << _min << "," << _max << "]" << std::endl;
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
bool RobotiqControl::VerifyCommand(robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput const &command)
{
  return VerifyField("rACT", 0, 1, command.rACT) &&
         VerifyField("rMOD", 0, 3, command.rACT) &&
         VerifyField("rGTO", 0, 1, command.rACT) &&
         VerifyField("rATR", 0, 1, command.rACT) &&
         VerifyField("rICF", 0, 1, command.rACT) &&
         VerifyField("rICS", 0, 1, command.rACT) &&
         VerifyField("rPRA", 0, 255, command.rACT) &&
         VerifyField("rSPA", 0, 255, command.rACT) &&
         VerifyField("rFRA", 0, 255, command.rACT) &&
         VerifyField("rPRB", 0, 255, command.rACT) &&
         VerifyField("rSPB", 0, 255, command.rACT) &&
         VerifyField("rFRB", 0, 255, command.rACT) &&
         VerifyField("rPRC", 0, 255, command.rACT) &&
         VerifyField("rSPC", 0, 255, command.rACT) &&
         VerifyField("rFRC", 0, 255, command.rACT) &&
         VerifyField("rPRS", 0, 255, command.rACT) &&
         VerifyField("rSPS", 0, 255, command.rACT) &&
         VerifyField("rFRS", 0, 255, command.rACT);
}

////////////////////////////////////////////////////////////////////////////////
void RobotiqControl::SetHandleCommand(robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput const &msg)
{
  boost::mutex::scoped_lock lock(controlMutex);

  // Sanity check.
  if (!VerifyCommand(msg))
  {
    std::cerr << "Ignoring command" << std::endl;
    return;
  }

  prevCommand = handleCommand;

  // Update handleCommand.
  handleCommand = msg;
}


////////////////////////////////////////////////////////////////////////////////
void RobotiqControl::ReleaseHand()
{
  // Open the fingers.
  handleCommand.rPRA = 0;
  handleCommand.rPRB = 0;
  handleCommand.rPRC = 0;

  // Half speed.
  handleCommand.rSPA = 127;
  handleCommand.rSPB = 127;
  handleCommand.rSPC = 127;
}

////////////////////////////////////////////////////////////////////////////////
void RobotiqControl::StopHand()
{
  // Set the target positions to the current ones.
  handleCommand.rPRA = handleState.gPRA;
  handleCommand.rPRB = handleState.gPRB;
  handleCommand.rPRC = handleState.gPRC;
}

////////////////////////////////////////////////////////////////////////////////
bool RobotiqControl::IsHandFullyOpen()
{
  bool fingersOpen = true;

  // The hand will be fully open when all the fingers are within 'tolerance'
  // from their lower limits.
  ignition::math::Angle tolerance;
  tolerance.Degree(1.0);

  for (int i = 2; i < NumJoints; ++i)
  {
    fingersOpen = fingersOpen &&
                  (joints[i]->Position(0) < (joints[i]->LowerLimit(0) + tolerance()));
  }

  return fingersOpen;
}

////////////////////////////////////////////////////////////////////////////////
void RobotiqControl::UpdateStates()
{
  boost::mutex::scoped_lock lock(controlMutex);

  auto const curTime = world_->SimTime();

  // Step 1: State transitions.
  if (curTime > lastControllerUpdateTime)
  {
    userHandleCommand = handleCommand;

    // Deactivate gripper.
    if (handleCommand.rACT == 0)
    {
      handState = Disabled;
    }
      // Emergency auto-release.
    else if (handleCommand.rATR == 1)
    {
      handState = Emergency;
    }
      // Individual Control of Scissor.
    else if (handleCommand.rICS == 1)
    {
      handState = ICS;
    }
      // Individual Control of Fingers.
    else if (handleCommand.rICF == 1)
    {
      handState = ICF;
    } else
    {
      // Change the grasping mode.
      if (static_cast<int>(handleCommand.rMOD) != graspingMode)
      {
        handState = ChangeModeInProgress;
        lastHandleCommand = handleCommand;

        // Update the grasping mode.
        graspingMode = static_cast<GraspingMode>(handleCommand.rMOD);
      } else if (handState != ChangeModeInProgress)
      {
        handState = Simplified;
      }

      // Grasping mode initialized, let's change the state to Simplified Mode.
      if (handState == ChangeModeInProgress && IsHandFullyOpen())
      {
        prevCommand = handleCommand;

        // Restore the original command.
        handleCommand = lastHandleCommand;
        handState = Simplified;
      }
    }

    // Step 2: Actions in each state.
    switch (handState)
    {
      case Disabled:
        break;

      case Emergency:
        // Open the hand.
        if (IsHandFullyOpen())
          StopHand();
        else
          ReleaseHand();
        break;

      case ICS:
        if (handleCommand.rGTO == 0)
        {
          // "Stop" action.
          StopHand();
        }
        break;

      case ICF:
        if (handleCommand.rGTO == 0)
        {
          // "Stop" action.
          StopHand();
        }
        break;

      case ChangeModeInProgress:
        // Open the hand.
        ReleaseHand();
        break;

      case Simplified:
        // We are in Simplified mode, so all the fingers should follow finger A.
        // Position.
        handleCommand.rPRB = handleCommand.rPRA;
        handleCommand.rPRC = handleCommand.rPRA;
        // Velocity.
        handleCommand.rSPB = handleCommand.rSPA;
        handleCommand.rSPC = handleCommand.rSPA;
        // Force.
        handleCommand.rFRB = handleCommand.rFRA;
        handleCommand.rFRC = handleCommand.rFRA;

        if (handleCommand.rGTO == 0)
        {
          // "Stop" action.
          StopHand();
        }
        break;

      default:
        std::cerr << "Unrecognized state [" << handState << "]" << std::endl;
    }

    // Update the hand controller.
    UpdatePIDControl((curTime - lastControllerUpdateTime).Double());

    // Gather robot state data and publish them.
    auto const handle_state = GetHandleState();

    lastControllerUpdateTime = curTime;
  }
}

////////////////////////////////////////////////////////////////////////////////
uint8_t RobotiqControl::GetObjectDetection(const gazebo::physics::JointPtr &_joint, int _index, uint8_t _rPR,
                                           uint8_t _prevrPR)
{
  // Check finger's speed.
  bool isMoving = _joint->GetVelocity(0) > VelTolerance;

  // Check if the finger reached its target positions. We look at the error in
  // the position PID to decide if reached the target.
  double pe, ie, de;
  posePID[_index].GetErrors(pe, ie, de);
  bool reachPosition = pe < PoseTolerance;

  if (isMoving)
  {
    // Finger is in motion.
    return 0;
  } else
  {
    if (reachPosition)
    {
      // Finger is at the requestedPosition.
      return 3;
    } else if (_rPR - _prevrPR > 0)
    {
      // Finger has stopped due to a contact while closing.
      return 2;
    } else
    {
      // Finger has stopped due to a contact while opening.
      return 1;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
uint8_t RobotiqControl::GetCurrentPosition(const gazebo::physics::JointPtr &_joint) const
{
  // Full range of motion.
  ignition::math::Angle range = _joint->UpperLimit(0) - _joint->LowerLimit(0);

  // The maximum value in pinch mode is 177.
  if (graspingMode == Pinch)
  {
    range *= 177.0 / 255.0;
  }

  // Angle relative to the lower limit.
  ignition::math::Angle relAngle = _joint->Position(0) - _joint->LowerLimit(0);

  return static_cast<uint8_t>(round(255.0 * relAngle() / range()));
}

////////////////////////////////////////////////////////////////////////////////
robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotInput RobotiqControl::GetHandleState()
{
  // gACT. Initialization status.
  handleState.gACT = userHandleCommand.rACT;

  // gMOD. Operation mode status.
  handleState.gMOD = userHandleCommand.rMOD;

  // gGTO. Action status.
  handleState.gGTO = userHandleCommand.rGTO;

  // gIMC. Gripper status.
  if (handState == Emergency)
  {
    handleState.gIMC = 0;
  } else if (handState == ChangeModeInProgress)
  {
    handleState.gIMC = 2;
  } else
  {
    handleState.gIMC = 3;
  }

  // Check fingers' speed.
  bool isMovingA = joints[2]->GetVelocity(0) > VelTolerance;
  bool isMovingB = joints[3]->GetVelocity(0) > VelTolerance;
  bool isMovingC = joints[4]->GetVelocity(0) > VelTolerance;

  // Check if the fingers reached their target positions.
  double pe, ie, de;
  posePID[2].GetErrors(pe, ie, de);
  bool reachPositionA = pe < PoseTolerance;
  posePID[3].GetErrors(pe, ie, de);
  bool reachPositionB = pe < PoseTolerance;
  posePID[4].GetErrors(pe, ie, de);
  bool reachPositionC = pe < PoseTolerance;

  // gSTA. Motion status.
  if (isMovingA || isMovingB || isMovingC)
  {
    // Gripper is in motion.
    handleState.gSTA = 0;
  } else
  {
    if (reachPositionA && reachPositionB && reachPositionC)
    {
      // Gripper is stopped: All fingers reached requested position.
      handleState.gSTA = 3;
    } else if (!reachPositionA && !reachPositionB && !reachPositionC)
    {
      // Gripper is stopped: All fingers stopped before requested position.
      handleState.gSTA = 2;
    } else
    {
      // Gripper stopped. One or two fingers stopped before requested position.
      handleState.gSTA = 1;
    }
  }

  // gDTA. Finger A object detection.
  handleState.gDTA = GetObjectDetection(joints[2], 2, handleCommand.rPRA, prevCommand.rPRA);

  // gDTB. Finger B object detection.
  handleState.gDTB = GetObjectDetection(joints[3], 3, handleCommand.rPRB, prevCommand.rPRB);

  // gDTC. Finger C object detection
  handleState.gDTC = GetObjectDetection(joints[4], 4, handleCommand.rPRC, prevCommand.rPRC);

  // gDTS. Scissor object detection. We use finger A as a reference.
  handleState.gDTS = GetObjectDetection(joints[0], 0, handleCommand.rPRS, prevCommand.rPRS);

  // gFLT. Fault status.
  if (handState == ChangeModeInProgress)
  {
    handleState.gFLT = 6;
  } else if (handState == Disabled)
  {
    handleState.gFLT = 7;
  } else if (handState == Emergency)
  {
    handleState.gFLT = 11;
  } else
  {
    handleState.gFLT = 0;
  }

  // gPRA. Echo of requested position for finger A.
  handleState.gPRA = userHandleCommand.rPRA;
  // gPOA. Finger A position [0-255].
  handleState.gPOA = GetCurrentPosition(joints[2]);
  // gCUA. Not implemented.
  handleState.gCUA = 0;

  // gPRB. Echo of requested position for finger B.
  handleState.gPRB = userHandleCommand.rPRB;
  // gPOB. Finger B position [0-255].
  handleState.gPOB = GetCurrentPosition(joints[3]);
  // gCUB. Not implemented.
  handleState.gCUB = 0;

  // gPRC. Echo of requested position for finger C.
  handleState.gPRC = userHandleCommand.rPRC;
  // gPOC. Finger C position [0-255].
  handleState.gPOC = GetCurrentPosition(joints[4]);
  // gCUS. Not implemented.
  handleState.gCUC = 0;

  // gPRS. Echo of requested position of the scissor action
  handleState.gPRS = userHandleCommand.rPRS;
  // gPOS. Scissor current position [0-255]. We use finger B as reference.
  handleState.gPOS = GetCurrentPosition(joints[1]);
  // gCUS. Not implemented.
  handleState.gCUS = 0;

  // Publish robot states.
  return handleState;
}

////////////////////////////////////////////////////////////////////////////////
void RobotiqControl::UpdatePIDControl(double _dt)
{
  // TODO: this shouldn't be a for-loop, each joint should be named
  for (int i = 0; i < NumJoints; ++i)
  {
    double targetPose = 0.0;

    if (i == 0)
    {
      switch (graspingMode)
      {
        case Wide:
          targetPose = joints[i]->UpperLimit(0);
          break;

        case Pinch:
          // --11 degrees.
          targetPose = -0.1919;
          break;

        case Scissor:
          // Max position is reached at value 215.
          targetPose = joints[i]->UpperLimit(0) -
                       (joints[i]->UpperLimit(0) -
                        joints[i]->LowerLimit(0)) * (215.0 / 255.0)
                       * handleCommand.rPRS / 255.0;
          break;
        case Basic:
          break;
      }
    } else if (i == 1)
    {
      switch (graspingMode)
      {
        case Wide:
          targetPose = joints[i]->LowerLimit(0);
          break;

        case Pinch:
          // 11 degrees.
          targetPose = 0.1919;
          break;

        case Scissor:
          // Max position is reached at value 215.
          targetPose = joints[i]->LowerLimit(0) +
                       (joints[i]->UpperLimit(0) -
                        joints[i]->LowerLimit(0)) * (215.0 / 255.0)
                       * handleCommand.rPRS / 255.0;
          break;
        case Basic:
          break;
      }
    } else if (i == 2)
    {
      if (graspingMode == Pinch)
      {
        // Max position is reached at value 177.
        targetPose = joints[i]->LowerLimit(0) +
                     (joints[i]->UpperLimit(0) -
                      joints[i]->LowerLimit(0)) * (177.0 / 255.0)
                     * handleCommand.rPRA / 255.0;
      } else
      {
        targetPose = joints[i]->LowerLimit(0) +
                     (joints[i]->UpperLimit(0) -
                      joints[i]->LowerLimit(0))
                     * handleCommand.rPRA / 255.0;
      }
    } else if (i == 3)
    {
      if (graspingMode == Pinch)
      {
        // Max position is reached at value 177.
        targetPose = joints[i]->LowerLimit(0) +
                     (joints[i]->UpperLimit(0) -
                      joints[i]->LowerLimit(0)) * (177.0 / 255.0)
                     * handleCommand.rPRB / 255.0;
      } else
      {
        targetPose = joints[i]->LowerLimit(0) +
                     (joints[i]->UpperLimit(0) -
                      joints[i]->LowerLimit(0))
                     * handleCommand.rPRB / 255.0;
      }
    } else if (i == 4)
    {
      if (graspingMode == Pinch)
      {
        // Max position is reached at value 177.
        targetPose = joints[i]->LowerLimit(0) +
                     (joints[i]->UpperLimit(0) -
                      joints[i]->LowerLimit(0)) * (177.0 / 255.0)
                     * handleCommand.rPRC / 255.0;
      } else
      {
        targetPose = joints[i]->LowerLimit(0) +
                     (joints[i]->UpperLimit(0) -
                      joints[i]->LowerLimit(0))
                     * handleCommand.rPRC / 255.0;
      }
    }

//    ROS_DEBUG_STREAM_NAMED(PLUGIN_LOG_NAME, "Target Positions for joint " << i << ": " << targetPose);

    // Get the current pose.
    double currentPose = joints[i]->Position(0);

    // Position error.
    double poseError = currentPose - targetPose;

    // Update the PID.
    double torque = posePID[i].Update(poseError, _dt);

    // Apply the PID command.
    fingerJoints[i]->SetForce(0, torque);
  }
}

/// Init helper functions
bool RobotiqControl::GetAndPushBackJoint(const std::string &_jointName, gazebo::physics::Joint_V &_joints) const
{
  gazebo::physics::JointPtr joint = model_->GetJoint(_jointName);

  if (!joint)
  {
    ROS_WARN_STREAM("Failed to find joint [" << _jointName << "] aborting plugin load.");
    return false;
  }
  _joints.push_back(joint);
  return true;
}

bool RobotiqControl::FindJoints()
{
  // Load up the joints we expect to use, finger by finger.
  gazebo::physics::JointPtr joint;
  std::string prefix;
  std::string suffix;
  if (side_ == "left")
  {
    prefix = "l_";
  } else
  {
    prefix = "r_";
  }

  // palm_finger_1_joint (actuated).
  suffix = "palm_finger_1_joint";
  if (!GetAndPushBackJoint(prefix + suffix, joints))
    return false;
  if (!GetAndPushBackJoint(prefix + suffix, fingerJoints))
    return false;

  // palm_finger_2_joint (actuated).
  suffix = "palm_finger_2_joint";
  if (!GetAndPushBackJoint(prefix + suffix, joints))
    return false;
  if (!GetAndPushBackJoint(prefix + suffix, fingerJoints))
    return false;

  // We read the joint state from finger_1_joint_1
  // but we actuate finger_1_joint_proximal_actuating_hinge (actuated).
  suffix = "finger_1_joint_proximal_actuating_hinge";
  if (!GetAndPushBackJoint(prefix + suffix, fingerJoints))
    return false;
  suffix = "finger_1_joint_1";
  if (!GetAndPushBackJoint(prefix + suffix, joints))
    return false;

  // We read the joint state from finger_2_joint_1
  // but we actuate finger_2_proximal_actuating_hinge (actuated).
  suffix = "finger_2_joint_proximal_actuating_hinge";
  if (!GetAndPushBackJoint(prefix + suffix, fingerJoints))
    return false;
  suffix = "finger_2_joint_1";
  if (!GetAndPushBackJoint(prefix + suffix, joints))
    return false;

  // We read the joint state from finger_middle_joint_1
  // but we actuate finger_middle_proximal_actuating_hinge (actuated).
  suffix = "finger_middle_joint_proximal_actuating_hinge";
  if (!GetAndPushBackJoint(prefix + suffix, fingerJoints))
    return false;
  suffix = "finger_middle_joint_1";
  if (!GetAndPushBackJoint(prefix + suffix, joints))
    return false;

  // palm_finger_1_joint (actuated).
  // palm_finger_2_joint (actuated).
  // but we actuate finger_1_joint_proximal_actuating_hinge (actuated).
  // but we actuate finger_2_proximal_actuating_hinge (actuated).
  // but we actuate finger_middle_proximal_actuating_hinge (actuated).

  // finger_1_joint_2 (underactuated).
  suffix = "finger_1_joint_2";
  if (!GetAndPushBackJoint(prefix + suffix, joints))
    return false;

  // finger_1_joint_3 (underactuated).
  suffix = "finger_1_joint_3";
  if (!GetAndPushBackJoint(prefix + suffix, joints))
    return false;

  // finger_2_joint_2 (underactuated).
  suffix = "finger_2_joint_2";
  if (!GetAndPushBackJoint(prefix + suffix, joints))
    return false;

  // finger_2_joint_3 (underactuated).
  suffix = "finger_2_joint_3";
  if (!GetAndPushBackJoint(prefix + suffix, joints))
    return false;

  // finger_middle_joint_2 (underactuated).
  suffix = "finger_middle_joint_2";
  if (!GetAndPushBackJoint(prefix + suffix, joints))
    return false;

  // finger_middle_joint_3 (underactuated).
  suffix = "finger_middle_joint_3";
  if (!GetAndPushBackJoint(prefix + suffix, joints))
    return false;

  return true;
}
