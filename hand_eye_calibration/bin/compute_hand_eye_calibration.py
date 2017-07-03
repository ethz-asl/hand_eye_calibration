#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv

import numpy as np

from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.dual_quaternion_hand_eye_calibration import (
    compute_hand_eye_calibration, compute_hand_eye_calibration_RANSAC,
    compute_hand_eye_calibration_BASELINE,
    align_paths_at_index, evaluate_alignment, HandEyeConfig)
from hand_eye_calibration.hand_eye_calibration_plotting_tools import plot_poses
from hand_eye_calibration.algorithm_config import (
    get_basic_config, get_RANSAC_classic_config,
    get_RANSAC_scalar_part_inliers_config,
    get_exhaustive_search_pose_inliers_config,
    get_exhaustive_search_scalar_part_inliers_config,
    get_baseline_config)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Align pairs of poses.')
  parser.add_argument(
      '--aligned_poses_B_H_csv_file', type=str,
      help='The path to the file containing the first poses. (e.g. Hand poses in Body frame)')
  parser.add_argument(
      '--aligned_poses_H_B_csv_file', type=str,
      help='Alternative input file for the first poses. (e.g. Body poses in Eye frame)')
  parser.add_argument(
      '--aligned_poses_W_E_csv_file', type=str,
      help='The path to the file containing the second poses. (e.g. Eye poses in World frame)')
  parser.add_argument(
      '--aligned_poses_E_W_csv_file', type=str,
      help='Alternative input file for the second poses. (e.g. World poses in Eye frame)')

  parser.add_argument(
      '--extrinsics_output_csv_file', type=str,
      help='Write estimated extrinsics to this file in spatial-extrinsics csv format')

  parser.add_argument('--visualize', type=bool, default=False, help='Visualize the poses.')
  parser.add_argument('--plot_every_nth_pose', type=int,
                      help='Plot only every n-th pose.', default=10)
  args = parser.parse_args()

  use_poses_B_H = (args.aligned_poses_B_H_csv_file is not None)
  use_poses_H_B = (args.aligned_poses_H_B_csv_file is not None)
  use_poses_W_E = (args.aligned_poses_W_E_csv_file is not None)
  use_poses_E_W = (args.aligned_poses_E_W_csv_file is not None)

  assert use_poses_B_H != use_poses_H_B, \
      "Provide either poses_B_H or poses_H_B!"
  assert use_poses_W_E != use_poses_E_W, \
      "Provide either poses_W_E or poses_E_W!"

  if use_poses_B_H:
    with open(args.aligned_poses_B_H_csv_file, 'r') as csvfile:
      poses1_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      poses1 = np.array(list(poses1_reader))
      poses1 = poses1.astype(float)
      dual_quat_B_H_vec = [DualQuaternion.from_pose_vector(
          pose[1:]) for pose in poses1]
  else:
    with open(args.aligned_poses_H_B_csv_file, 'r') as csvfile:
      poses1_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      poses1 = np.array(list(poses1_reader))
      poses1 = poses1.astype(float)
      dual_quat_B_H_vec = [DualQuaternion.from_pose_vector(
          pose[1:]).inverse() for pose in poses1]

  if use_poses_W_E:
    with open(args.aligned_poses_W_E_csv_file, 'r') as csvfile:
      poses2_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      poses2 = np.array(list(poses2_reader))
      poses2 = poses2.astype(float)
      dual_quat_W_E_vec = [DualQuaternion.from_pose_vector(
          pose[1:]) for pose in poses2]
  else:
    with open(args.aligned_poses_E_W_csv_file, 'r') as csvfile:
      poses2_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      poses2 = np.array(list(poses2_reader))
      poses2 = poses2.astype(float)
      dual_quat_W_E_vec = [DualQuaternion.from_pose_vector(
          pose[1:]).inverse() for pose in poses2]

  assert len(dual_quat_B_H_vec) == len(dual_quat_W_E_vec), "len(dual_quat_B_H_vec): {} vs len(dual_quat_W_E_vec): {}".format(
      len(dual_quat_B_H_vec), len(dual_quat_W_E_vec))

  dual_quat_B_H_vec = align_paths_at_index(dual_quat_B_H_vec)
  dual_quat_W_E_vec = align_paths_at_index(dual_quat_W_E_vec)

  # Draw both paths in their Global / World frame.
  if args.visualize:
    poses1 = np.array([dual_quat_B_H_vec[0].to_pose().T])
    poses2 = np.array([dual_quat_W_E_vec[0].to_pose().T])
    for i in range(1, len(dual_quat_B_H_vec)):
      poses1 = np.append(poses1, np.array(
          [dual_quat_B_H_vec[i].to_pose().T]), axis=0)
      poses2 = np.append(poses2, np.array(
          [dual_quat_W_E_vec[i].to_pose().T]), axis=0)
    every_nth_element = args.plot_every_nth_pose
    plot_poses([poses1[::every_nth_element], poses2[::every_nth_element]],
               True, title="3D Poses Before Alignment")

  # TODO(mfehr): Add param to switch between algorithms.
  # (_, hand_eye_config) = get_RANSAC_scalar_part_inliers_config(True)
  # (_, hand_eye_config) = get_RANSAC_classic_config(False)
  (_, hand_eye_config) =  get_exhaustive_search_scalar_part_inliers_config()
  # (_, hand_eye_config) = get_baseline_config(True)

  hand_eye_config.visualize = args.visualize
  hand_eye_config.visualize_plot_every_nth_pose = args.plot_every_nth_pose

  if hand_eye_config.use_baseline_approach:
    (success, dq_H_E, rmse,
     num_inliers, num_poses_kept,
     runtime, singular_values, bad_singular_values) = compute_hand_eye_calibration_BASELINE(
        dual_quat_B_H_vec, dual_quat_W_E_vec, hand_eye_config)
  else:
    (success, dq_H_E, rmse,
     num_inliers, num_poses_kept,
     runtime, singular_values, bad_singular_values) = compute_hand_eye_calibration_RANSAC(
        dual_quat_B_H_vec, dual_quat_W_E_vec, hand_eye_config)

  if args.extrinsics_output_csv_file is not None:
      print("Writing extrinsics to %s." % args.extrinsics_output_csv_file)
      from hand_eye_calibration.csv_io import write_double_numpy_array_to_csv_file
      write_double_numpy_array_to_csv_file(dq_H_E.to_pose(), args.extrinsics_output_csv_file);
