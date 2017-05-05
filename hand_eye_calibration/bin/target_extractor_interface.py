#!/usr/bin/env python
from subprocess import call
import argparse
import yaml


class TargetExtractorConfig(object):
  base_directory = ""
  bag_file_name = ""
  calib_file_camera = "../calib/sr300_sim.yaml"
  topic = "/camera/rgb/image_raw"
  draw_extraction = True
  camera_pose_T_C_G_output_file = "sequenced_camera_poses.csv"
  camera_pose_T_C_G_output_file_timestamped = "timestamped_camera_poses.csv"
  alsologtostderr = True
  write_csv_header = False
  inlier_ratio_for_good_camera_pose = 0.
  april_tag_number_vertical = 6
  april_tag_number_horizontal = 6
  april_tag_number_pixel_boarder = 2
  april_tag_size_m = 0.055
  april_tag_gap_size_m = 0.0165

  def __init__(self):
    pass


def call_target_extractor(target_extractor_config):
  print("Extracting images from bag: {}.".format(te_config.bag_file_name))
  te_arguments = ["rosrun", "hand_eye_calibration_target_extractor", "target_extractor"]
  te_arguments.append("--bag")
  te_arguments.append(bag_path)
  te_arguments.append("--topic")
  te_arguments.append(target_extractor_config.topic)
  te_arguments.append("--eval_camera_yaml")
  te_arguments.append(target_extractor_config.calib_file_camera)
  te_arguments.append(
      "--draw_extraction={}".format(target_extractor_config.draw_extraction))
  te_arguments.append("--camera_pose_T_C_G_output_file")
  te_arguments.append(target_extractor_config.camera_pose_T_C_G_output_file)
  te_arguments.append("--camera_pose_T_C_G_output_file_timestamped")
  te_arguments.append(
      target_extractor_config.camera_pose_T_C_G_output_file_timestamped)
  if target_extractor_config.alsologtostderr:
    te_arguments.append("--alsologtostderr")
  te_arguments.append(
      "--write_csv_header={}".format(target_extractor_config.write_csv_header))
  te_arguments.append("--april_tag_number_vertical")
  te_arguments.append(str(target_extractor_config.april_tag_number_vertical))
  te_arguments.append("--april_tag_number_horizontal")
  te_arguments.append(str(target_extractor_config.april_tag_number_horizontal))
  te_arguments.append("--april_tag_size_m")
  te_arguments.append(str(target_extractor_config.april_tag_size_m))
  te_arguments.append("--april_tag_pixel_boarder")
  te_arguments.append(
      str(target_extractor_config.april_tag_number_pixel_boarder))
  te_arguments.append("--april_tag_gap_size_m")
  te_arguments.append(str(target_extractor_config.april_tag_gap_size_m))
  call(te_arguments)


if __name__ == "__main__":
  te_config = TargetExtractorConfig()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--bag', required=True, help='Rosbag to parse.')
  parser.add_argument(
      '--calib_file_camera', required=True,
      help='The yaml file containing the intrinsic camera calibration.')
  parser.add_argument(
      '--calib_file_target', required=True,
      help='The yaml file containing the april tag target information.')
  parser.add_argument(
      '--image_topic', required=True,
      help='Image topic to compute camera transformation frequency.')
  parser.add_argument(
      '--output_file', required=True,
      help='The output csv file name containing the stamped camera poses.')
  args = parser.parse_args()

  bag_path = args.bag
  te_config.bag_file_name = bag_path

  te_config.topic = args.image_topic
  te_config.calib_file_camera = args.calib_file_camera
  with open(args.calib_file_target, 'r') as stream:
    try:
      yaml_content = yaml.load(stream)
    except yaml.YAMLError as exc:
      print(exc)
  te_config.april_tag_number_vertical = yaml_content[
      'april_tag_number_vertical']
  te_config.april_tag_number_horizontal = yaml_content[
      'april_tag_number_horizontal']
  te_config.april_tag_number_pixel_boarder = yaml_content[
      'april_tag_number_pixel_boarder']
  te_config.april_tag_size_m = yaml_content['april_tag_size_m']
  te_config.april_tag_gap_size_m = yaml_content['april_tag_gap_size_m']
  te_config.camera_pose_T_C_G_output_file = "sequenced_" + args.output_file
  te_config.camera_pose_T_C_G_output_file_timestamped = args.output_file

  call_target_extractor(te_config)
