#!/usr/bin/env python
# author: Matias Mattamala

import rospy
from mmstereo_ros.ros_interface import RosInterface

if __name__ == "__main__":
    rospy.init_node("mmstereo_ros")
    app = RosInterface()

    rospy.loginfo("[mmstereo_ros] Ready")
    rospy.spin()
