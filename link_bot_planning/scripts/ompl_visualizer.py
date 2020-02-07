import matplotlib.pyplot as plt
import numpy as np
import rospy
import argparse
import ompl.base as ob

from link_bot_pycommon.args import my_formatter


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    # parser.add_argument()

    args = parser.parse_args()

    # ROS
    rospy.Subscriber('/ompl_viz/planner_data')

    plt.figure()
    main_ax = plt.gca()



if __name__ == '__main__':
    main()
