from time import sleep

import numpy as np

import rospy
from peter_msgs.msg import AnimationControl
from peter_msgs.srv import GetFloat32, GetFloat32Request, GetBool, GetBoolRequest
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import Int64


class RvizAnimationController:

    def __init__(self, time_steps=None, n_time_steps: int = None):
        if time_steps is not None:
            self.time_steps = np.array(time_steps, dtype=np.int64)
        if n_time_steps is not None:
            self.time_steps = np.arange(n_time_steps, dtype=np.int64)
        self.command_sub = rospy.Subscriber("/rviz_anim/control", AnimationControl, self.on_control)
        self.period_srv = rospy.ServiceProxy("/rviz_anim/period", GetFloat32)
        self.time_pub = rospy.Publisher("/rviz_anim/time", Int64, queue_size=10)
        self.max_time_pub = rospy.Publisher("/rviz_anim/max_time", Int64, queue_size=10)
        self.auto_play_srv = rospy.ServiceProxy("/rviz_anim/auto_play", GetBool)

        rospy.wait_for_service("/rviz_anim/period")

        self.idx = 0
        self.max_idx = self.time_steps.shape[0]
        self.max_t = self.time_steps[-1]
        self.period = self.period_srv(GetFloat32Request()).data
        self.auto_play = self.auto_play_srv(GetBoolRequest()).data
        self.playing = self.auto_play
        self.should_step = False
        self.fwd = True
        self.done = False

    def on_control(self, msg: AnimationControl):
        if msg.command == AnimationControl.STEP_BACKWARD:
            self.on_bwd()
        elif msg.command == AnimationControl.STEP_FORWARD:
            self.on_fwd()
        elif msg.command == AnimationControl.PLAY_BACKWARD:
            self.on_play_backward()
        elif msg.command == AnimationControl.PLAY_FORWARD:
            self.on_play_forward()
        elif msg.command == AnimationControl.PAUSE:
            self.on_pause()
        elif msg.command == AnimationControl.DONE:
            self.on_done()
        else:
            raise NotImplementedError(f"Unsupported animation control {msg.command}")

    def on_fwd(self):
        self.should_step = True
        self.playing = False
        self.fwd = True

    def on_bwd(self):
        self.should_step = True
        self.playing = False
        self.fwd = False

    def on_play_forward(self):
        self.playing = True
        self.fwd = True

    def on_play_backward(self):
        self.playing = True
        self.fwd = False

    def on_pause(self):
        self.playing = False

    def on_done(self):
        self.done = True

    def step(self):
        if self.playing:
            # don't use ros time because we don't want to rely on simulation time
            sleep(self.period)
        else:
            while not self.should_step and not self.playing and not self.done:
                sleep(0.01)

        if self.fwd:
            if self.idx < self.max_idx - 1:
                self.idx += 1
            else:
                if self.auto_play:
                    self.done = True
                else:
                    self.playing = False
        else:
            if self.idx > 0:
                self.idx -= 1

        t_msg = Int64()
        t_msg.data = self.time_steps[self.idx]
        self.time_pub.publish(t_msg)

        self.should_step = False

        max_t_msg = Int64()
        max_t_msg.data = self.time_steps[-1]
        self.max_time_pub.publish(max_t_msg)

        return self.done

    def t(self):
        return self.time_steps[self.idx]


class RvizSimpleStepper:

    def __init__(self):
        self.command_sub = rospy.Subscriber("/rviz_anim/control", AnimationControl, self.on_control)
        self.should_step = False

    def on_control(self, msg: AnimationControl):
        if msg.command == AnimationControl.STEP_FORWARD:
            self.should_step = True

    def step(self):
        while not self.should_step:
            sleep(0.05)
        self.should_step = False
