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


def write_transformation_to_csv_file(bag_file, target_frame, source_frame,
                                     csv_file_name):
  print("Loading tfs into Transformer...")
  tf_tree = tf.Transformer(True, rospy.Duration(3600.0))
  bag = rosbag.Bag(bag_file)

  for topic, msg, t in bag.read_messages(topics=['/tf']):
    for msg_tf in msg.transforms:
      tf_tree.setTransform(msg_tf)
  bag.close()

  print("Listening to tf transformation from ", source_frame, " to ",
        target_frame)
  # Reopen bag
  bag = rosbag.Bag(bag_file)
  init = True
  tf_counter = 0
  tf_frequency_estimated = 0.
  tf_interval_estimated = 0.
  tf_previous_timestamp = 0.
  start_time_tf_message = rospy.Time()
  end_time_tf_message = rospy.Time()
  csv_file = open(csv_file_name, 'w')
  print("Looking up transforms and writing to csv file...")
  for topic, msg, t in bag.read_messages():
    if topic == "/tf" and msg.transforms:
      for single_tf in msg.transforms:
        if single_tf.child_frame_id == target_frame:
          try:
            (translation,
             hamilton_quaternion) = tf_tree.lookupTransform(target_frame, source_frame,
                                                            single_tf.header.stamp)
          except (tf.LookupException, tf.ConnectivityException,
                  tf.ExtrapolationException):
            # Only start counting if we already did at least one successful tf
            # lookup
            if tf_counter > 0:
              tf_counter += 1
            continue

          # Initialize start time to first successful tf message lookup.
          if init:
            start_time_tf_message = single_tf.header.stamp
            init = False

          # Write to csv file.
          quaternion = np.array(hamilton_quaternion)
          csv_file.write(
              str(single_tf.header.stamp.to_sec()) + ', ' +
              str(translation[0]) + ', ' + str(translation[1]) + ', ' +
              str(translation[2]) + ', ' + str(quaternion[0]) + ', ' +
              str(quaternion[1]) + ', ' + str(quaternion[2]) + ', ' +
              str(quaternion[3]) + '\n')

          # Update end time.
          end_time_tf_message = single_tf.header.stamp
          tf_counter += 1

          # Check if there was a drop in the tf frequency.
          if tf_counter > 3:
            tf_frequency_estimated = tf_counter / (
                end_time_tf_message - start_time_tf_message).to_sec()

            # Check if there has been a drop.
            tf_interval_estimated = 1. / tf_frequency_estimated
            tf_interval_measures = single_tf.header.stamp.to_sec() - tf_previous_timestamp.to_sec()

            # If the interval deviates from the frequency by more than x percent, print a warning.
            tf_interval_deviation_percent = abs(
                tf_interval_estimated -
                tf_interval_measures) / tf_interval_estimated * 100.
            if (tf_interval_deviation_percent > 50.0):
              seconds_from_start_time = (
                  single_tf.header.stamp.to_sec() -
                  start_time_tf_message.to_sec())
              print("There might have been a drop in the tf after ", "{0:.3f}".format(
                  seconds_from_start_time), "s.")
              print("\tThe interval deviated by ", "{0:.2f}".format(
                  tf_interval_deviation_percent), "%.")
              print("\tCurrent frequency estimate: ", "{0:.2f}".format(
                  tf_frequency_estimated), "Hz")

          tf_previous_timestamp = single_tf.header.stamp

  # Output final estimated frequency.
  if tf_counter > 3:
    tf_frequency_estimated = tf_counter / (
        end_time_tf_message - start_time_tf_message).to_sec()
    print("Final estimate of tf topic frequency: ", "{0:.2f}".format(
        tf_frequency_estimated), "Hz")

  print("Exported ", tf_counter, " tf poses.")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--bag', required=True, help='Rosbag to parse.')
  parser.add_argument(
      '--tf_source_frame', required=True, help='Name of tf source frame.')
  parser.add_argument(
      '--tf_target_frame', required=True, help='Name of tf target frame.')
  parser.add_argument(
      '--csv_output_file', required=True, help='Path to output csv file')

  args = parser.parse_args()

  print("tf_to_csv.py: export tf to csv from bag: ", args.bag, "...")

  write_transformation_to_csv_file(args.bag, args.tf_target_frame,
                                   args.tf_source_frame,
                                   args.csv_output_file)
