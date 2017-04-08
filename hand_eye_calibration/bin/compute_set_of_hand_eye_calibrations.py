#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import itertools

import numpy as np

from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.quaternion import Quaternion
from hand_eye_calibration.dual_quaternion_hand_eye_calibration import (
    compute_hand_eye_calibration, compute_hand_eye_calibration_RANSAC,
    align_paths_at_index, evaluate_alignment, HandEyeConfig)
from hand_eye_calibration.hand_eye_calibration_plotting_tools import plot_poses
from hand_eye_calibration.csv_io import (
    write_time_stamped_poses_to_csv_file, read_time_stamped_poses_from_csv_file)
from hand_eye_calibration.time_alignment import (
    calculate_time_offset, compute_aligned_poses, FilteringConfig)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Align pairs of poses.')
  parser.add_argument(
      '--aligned_poses_B_H_csv_files', type=str, required=True,
      help='The path to the file containing the first poses. (e.g. Hand poses in Body frame)')
  parser.add_argument('--result_file', type=str, required=True,
                      help='The path to the result file.')

  parser.add_argument('--visualize', type=bool, default=False, help='Visualize the poses.')
  parser.add_argument('--plot_every_nth_pose', type=int,
                      help='Plot only every n-th pose.', default=10)
  args = parser.parse_args()

  assert args.aligned_poses_B_H_csv_files is not None

  poses_B_H_csv_files = args.aligned_poses_B_H_csv_files.split(':')
  num_poses_sets = len(poses_B_H_csv_files)
  assert num_poses_sets >= 2, "Needs at least two sets of poses."

  set_of_pose_pairs = list(itertools.combinations(poses_B_H_csv_files, 2))
  num_pose_pairs = len(set_of_pose_pairs)

  # Config:
  hand_eye_config = HandEyeConfig()
  hand_eye_config.visualize = args.visualize
  hand_eye_config.visualize_plot_every_nth_pose = args.plot_every_nth_pose
  hand_eye_config.ransac_inlier_classification = "scalar_part_equality"
  # hand_eye_config.ransac_inlier_classification = "rmse_threshold"

  # Results:
  results_dataset_names = []
  results_success = []
  results_dq_H_E = []
  results_poses_H_E = []
  results_rmse = []
  result_num_inliers = []
  result_num_initial_poses = []
  result_num_poses_kept = []
  result_runtimes = []

  for (pose_file_B_H, pose_file_W_E) in set_of_pose_pairs:
    print("\n\nHand-calibration between \n\t{}\n and \n\t{}\n\n".format(
        pose_file_B_H, pose_file_W_E))
    (time_stamped_poses_B_H,
     times_B_H,
     quaternions_B_H
     ) = read_time_stamped_poses_from_csv_file(pose_file_B_H)
    print("Found ", time_stamped_poses_B_H.shape[0],
          " poses in file: ", pose_file_B_H)

    (time_stamped_poses_W_E,
     times_W_E,
     quaternions_W_E) = read_time_stamped_poses_from_csv_file(pose_file_W_E)
    print("Found ", time_stamped_poses_W_E.shape[0],
          " poses in file: ", pose_file_W_E)

    print("Computing time offset...")
    filtering_config = FilteringConfig()
    # TODO(mfehr): get filtering config from args!
    time_offset = calculate_time_offset(times_B_H, quaternions_B_H, times_W_E,
                                        quaternions_W_E, filtering_config, args.visualize)

    print("Final time offset: {}s".format(time_offset))

    print("Computing aligned poses...")
    (aligned_poses_B_H, aligned_poses_W_E) = compute_aligned_poses(
        time_stamped_poses_B_H, time_stamped_poses_W_E, time_offset, args.visualize)

    # Convert poses to dual quaterions.
    dual_quat_B_H_vec = [DualQuaternion.from_pose_vector(
        aligned_pose_B_H) for aligned_pose_B_H in aligned_poses_B_H[:, 1:]]
    dual_quat_W_E_vec = [DualQuaternion.from_pose_vector(
        aligned_pose_W_E) for aligned_pose_W_E in aligned_poses_W_E[:, 1:]]

    assert len(dual_quat_B_H_vec) == len(dual_quat_W_E_vec), "len(dual_quat_B_H_vec): {} vs len(dual_quat_W_E_vec): {}".format(
        len(dual_quat_B_H_vec), len(dual_quat_W_E_vec))

    dual_quat_B_H_vec = align_paths_at_index(dual_quat_B_H_vec)
    dual_quat_W_E_vec = align_paths_at_index(dual_quat_W_E_vec)

    # Draw both paths in their Global / World frame.
    if args.visualize:
      poses_B_H = np.array([dual_quat_B_H_vec[0].to_pose().T])
      poses_W_E = np.array([dual_quat_W_E_vec[0].to_pose().T])
      for i in range(1, len(dual_quat_B_H_vec)):
        poses_B_H = np.append(poses_B_H, np.array(
            [dual_quat_B_H_vec[i].to_pose().T]), axis=0)
        poses_W_E = np.append(poses_W_E, np.array(
            [dual_quat_W_E_vec[i].to_pose().T]), axis=0)
      every_nth_element = args.plot_every_nth_pose
      plot_poses(poses_B_H[::every_nth_element], poses_W_E[::every_nth_element],
                 True, title="3D Poses Before Alignment")

    print("Computing hand-eye calibration...")
    (success, dq_H_E, rmse, num_inliers, num_poses_kept, runtime) = compute_hand_eye_calibration_RANSAC(dual_quat_B_H_vec, dual_quat_W_E_vec,
                                                                                                        hand_eye_config)
    results_dataset_names.append((pose_file_B_H, pose_file_W_E))
    results_success.append(success)
    results_dq_H_E.append(dq_H_E)
    results_poses_H_E.append(dq_H_E.to_pose())
    results_rmse.append(rmse)
    result_num_inliers.append(num_inliers)
    result_num_initial_poses.append(len(dual_quat_B_H_vec))
    result_num_poses_kept.append(num_poses_kept)
    result_runtimes.append(runtime)

  assert len(results_dataset_names) == num_pose_pairs
  assert len(results_success) == num_pose_pairs
  assert len(results_dq_H_E) == num_pose_pairs
  assert len(results_poses_H_E) == num_pose_pairs
  assert len(results_rmse) == num_pose_pairs
  assert len(result_num_inliers) == num_pose_pairs
  assert len(result_num_initial_poses) == num_pose_pairs
  assert len(result_num_poses_kept) == num_pose_pairs
  assert len(result_runtimes) == num_pose_pairs

  output_file = open(args.result_file, 'w')
  for i in range(0, num_pose_pairs):
    output_file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
        results_dataset_names[i][0], results_dataset_names[i][1],
        results_success[i], results_rmse[i][0], results_rmse[i][1],
        result_num_inliers[i], result_num_initial_poses[i],
        result_num_poses_kept[i], result_runtimes[i]))

  dq_CALIGULA_MARS = results_dq_H_E[0]
  dq_CALIGULA_NERO = results_dq_H_E[1]
  dq_MARS_NERO = results_dq_H_E[2]

  dq_CALIGULA = DualQuaternion(Quaternion(0, 0, 0, 1), Quaternion(0, 0, 0, 0))
  dq_CALIGULA_NERO_2 = dq_CALIGULA_MARS * dq_MARS_NERO
  dq_CALIGULA_2 = dq_CALIGULA_NERO * dq_MARS_NERO.inverse() * dq_CALIGULA_MARS.inverse()

  poses = np.array([dq_CALIGULA.to_pose(), dq_CALIGULA_NERO.to_pose(), dq_CALIGULA_MARS.to_pose(),
                    dq_CALIGULA_NERO_2.to_pose(),
                    dq_CALIGULA_2.to_pose()])
  plot_poses(poses)
