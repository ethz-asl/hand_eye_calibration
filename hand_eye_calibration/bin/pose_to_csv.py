#!/usr/bin/env python
import argparse
import math
import sys
import time

import numpy as np

import rosbag
import rospy
import tf
from tf2_msgs.msg import TFMessage
import warnings


def write_transformation_to_csv_file(bag_file, pose_topic, csv_file_name):
    bag = rosbag.Bag(bag_file)

    print("Exporting transformations from ", pose_topic, ".")
    # Reopen bag
    bag = rosbag.Bag(bag_file)
    pose_counter = 0
    csv_file = open(csv_file_name, 'w')
    print("Looking up transforms and writing to csv file...")
    for topic, msg, t in bag.read_messages():
        if topic == pose_topic:
            csv_file.write(
                str(msg.header.stamp.to_sec()) + ', ' +
                str(msg.pose.position.x) + ', ' + str(msg.pose.position.y) +
                ', ' + str(msg.pose.position.z) + ', ' +
                str(msg.pose.orientation.x) + ', ' +
                str(msg.pose.orientation.y) + ', ' +
                str(msg.pose.orientation.z) + ', ' +
                str(msg.pose.orientation.w) + '\n')
            pose_counter += 1

    print("Exported ", pose_counter, " poses.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True, help='Rosbag to parse.')
    parser.add_argument(
        '--pose_topic',
        required=True,
        help='Pose topic name that should get exrtracted.')
    parser.add_argument(
        '--csv_output_file', required=True, help='Path to output csv file')

    args = parser.parse_args()

    print("pose_to_csv.py: export pose to csv from bag: ", args.bag, "...")

    write_transformation_to_csv_file(args.bag, args.pose_topic,
                                     args.csv_output_file)
