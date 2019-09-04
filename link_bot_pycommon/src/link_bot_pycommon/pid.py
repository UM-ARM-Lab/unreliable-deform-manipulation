import numpy as np


class PID:

    def __init__(self, kP, kI=0, kD=0, max_integral=0, max_output=0, max_acc=0):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.max_integral = max_integral
        self.max_output = max_output
        self.max_acc = max_acc
        self.setpoint = 0
        self.integral = 0
        self.last_error = 0
        self.current_setpoint = 0

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def output(self, measurement):
        delta = self.setpoint - self.current_setpoint
        clamped_acc = max(min(self.max_acc, delta), -self.max_acc)
        self.current_setpoint = self.current_setpoint + clamped_acc

        error = self.setpoint - measurement
        derivative = error - self.last_error
        self.integral += error
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)

        output = self.kP * error + self.kI * self.integral + self.kD * derivative
        output = np.clip(output, -self.max_output, self.max_output)
        return output
