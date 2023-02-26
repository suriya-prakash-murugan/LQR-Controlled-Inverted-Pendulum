#!/usr/bin/env python3
from control_msgs.msg import JointControllerState
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import rospy
import control
import math
import numpy as np

# constants from Gazebo
pole_mass = 2
cart_Mass = 20
gravity = 9.8
length_of_pole = 0.5

# A and B matrix from class lecture slides
A = np.array([[0, 1, 0, 0],
               [0, 0, (-12 * pole_mass * gravity) / ((12 * cart_Mass) + pole_mass), 0],
               [0, 0, 0, 1],
               [0, 0, (12 * gravity * (cart_Mass + pole_mass)) / (length_of_pole * ((13 * cart_Mass) + pole_mass)), 0]
               ])

B = np.array([[0],
               [13 / ((13 * cart_Mass) + pole_mass)],
               [0],
               [-12 / (length_of_pole * ((13 * cart_Mass) + pole_mass))]
               ])

Q = np.diag([1, 1, 1, 1.]) * 75
R = np.diag([1.])

# Using control module to get optimal gains
K, S, E = control.lqr(A, B, Q, R)


class LQR_Controller:
    def __init__(self):
        rospy.init_node('LQR_Controller')
        self.command_publisher = rospy.Publisher("/invpend/joint1_velocity_controller/command",
                                           Float64, queue_size=10)
        self.theta_subscriber = rospy.Subscriber("/invpend/joint2_position_controller/state",
                                          JointControllerState, self.angle_callback)
        self.pos_subscriber = rospy.Subscriber("/invpend/joint_states",
                                        JointState, self.pos_callback)
        self.current_state = np.array([0., 0., 0., 0.])
        self.desired_state = np.array([0., 0., 0., 0.])
        self.command_msg = Float64()

    def angle_callback(self, theta_msg):
        self.current_state[2] = theta_msg.process_value
        self.current_state[3] = theta_msg.process_value_dot
        rospy.loginfo_throttle(2, f'Current Angle: {math.degrees(theta_msg.process_value)}')

    def pos_callback(self, pos_msg):
        self.current_state[0] = pos_msg.position[1]
        self.current_state[1] = pos_msg.velocity[1]

    def balance(self):
        state_difference = self.desired_state - self.current_state
        self.command_msg.data = np.matmul(K, state_difference)
        self.command_publisher.publish(self.command_msg)
        rospy.loginfo_throttle(2, f'Commanding: {self.command_msg.data}')


def main():
    b = LQR_Controller()
    while not rospy.is_shutdown():
        b.balance()


if __name__ == '__main__':
    main()