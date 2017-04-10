#!/usr/bin/env python
from subprocess import call

import argparse
import copy
import rospy
import yaml


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--yaml_config_file', required=True,
      help='A yaml file specifying topics and frames.')

  args = parser.parse_args()

  tf_arguments = ["rosrun", "hand_eye_calibration", "tf_to_csv.py"]
  te_arguments = ["rosrun", "hand_eye_calibration",
                  "target_extractor_interface.py"]

  with open(args.yaml_config_file, 'r') as stream:
    try:
      yaml_content = yaml.load(stream)
    except yaml.YAMLError as exc:
      print(exc)
  bags = yaml_content['bags']

  for bag in bags:
    bag_name = bag['name']
    bag_name_without_suffix = bag_name.split('.')[0]
    bag_path = bag['bag_path']
    intrinsics_path = bag['intrinsics_path']
    target_config_path = bag['target_config_path']
    print("\n\nExporting poses from {}...\n".format(bag_name))

    tf_frames = bag['tf_frames']
    if tf_frames is None:
      num_tf_source_frames = 0
      num_tf_target_frames = 0
    else:
      num_tf_source_frames = len(tf_frames['source'])
      num_tf_target_frames = len(tf_frames['target'])

    assert num_tf_source_frames == num_tf_target_frames, "Source and target frames should have equal length."

    for i in range(num_tf_source_frames):
      tf_source_frame = tf_frames['source'][i]
      tf_target_frame = tf_frames['target'][i]
      print("Exporting poses of {} in {} frame".format(
          tf_source_frame, tf_target_frame))
      # Setup the args for tf_to_csv.py.
      tf_call = copy.deepcopy(tf_arguments)
      tf_call.append('--bag')
      tf_call.append(bag_path + bag_name)
      tf_call.append('--tf_source_frame')
      tf_call.append(tf_source_frame)
      tf_call.append('--tf_target_frame')
      tf_call.append(tf_target_frame)
      tf_call.append('--csv_output_file')
      tf_call.append(bag_name_without_suffix + '_' +
                     tf_target_frame + '_' + tf_source_frame + '.csv')
      call(tf_call)
    cameras = bag['cameras']
    equal_length_warning = ("Camera topics and intrinsic calibrations should "
                            "have equal length.")
    if cameras is None:
      num_camera_topics = 0
      num_camera_intrinsics = 0
    else:
      num_camera_topics = len(cameras['cam_topics'])
      num_camera_intrinsics = len(
          cameras['cam_intrinsics'])
    assert num_camera_topics == num_camera_intrinsics, equal_length_warning
    target_config = bag['target']

    for i in range(num_camera_topics):
      camera_topic = cameras['cam_topics'][i]
      camera_intrinsics = cameras['cam_intrinsics'][i]
      print("Exporting {} poses in world (target) frame.".format(camera_topic))
      # Setup the args for target_extractor_interface.py.
      te_call = copy.deepcopy(te_arguments)
      te_call.append('--bag')
      te_call.append(bag_path + bag_name)
      te_call.append('--image_topic')
      te_call.append(camera_topic)
      te_call.append('--calib_file_camera')
      te_call.append(intrinsics_path + camera_intrinsics)
      te_call.append('--calib_file_target')
      te_call.append(target_config_path + target_config)
      te_call.append('--output_file')
      te_call.append(bag_name_without_suffix + '_' +
                     'W_' + camera_topic.replace("/", "_")[1:] + '.csv')
      call(te_call)
