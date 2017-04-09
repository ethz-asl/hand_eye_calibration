#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import itertools
import os
import sys

import numpy as np

from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.quaternion import Quaternion
from hand_eye_calibration.dual_quaternion_hand_eye_calibration import (
    compute_hand_eye_calibration, compute_hand_eye_calibration_RANSAC,
    compute_hand_eye_calibration_BASELINE,
    align_paths_at_index, evaluate_alignment, HandEyeConfig, compute_pose_error)
from hand_eye_calibration.hand_eye_calibration_plotting_tools import plot_poses
from hand_eye_calibration.csv_io import (
    write_time_stamped_poses_to_csv_file, read_time_stamped_poses_from_csv_file)
from hand_eye_calibration.time_alignment import (
    calculate_time_offset, compute_aligned_poses, FilteringConfig)
from hand_eye_calibration.algorithm_config import (
    get_basic_config, get_RANSAC_classic_config,
    get_RANSAC_scalar_part_inliers_config,
    get_exhaustive_search_pose_inliers_config,
    get_exhaustive_search_scalar_part_inliers_config,
    get_baseline_config)

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
  parser.add_argument('--num_iterations', type=int,
                      help='Number of iterations per algorithm.'
                      'Only applicable to non - deterministic algorithms!', default=2)
  args = parser.parse_args()

  assert args.aligned_poses_B_H_csv_files is not None

  poses_B_H_csv_files = args.aligned_poses_B_H_csv_files.split(':')
  num_poses_sets = len(poses_B_H_csv_files)
  assert num_poses_sets >= 2, "Needs at least two sets of poses."

  set_of_pose_pairs = list(itertools.combinations(poses_B_H_csv_files, 2))
  num_pose_pairs = len(set_of_pose_pairs)

  # Prepare result file.
  output_file = open(args.result_file, 'w')
  output_file.write("algorithm_name,prefiltering,poses_B_H_csv_file,poses_W_E_csv_file,"
                    "success,position_rmse,orientation_rmse,"
                    "num_inliers,num_input_poses,num_poses"
                    "after_filtering,runtime_s,"
                    "loop_error_position_m,"
                    "loop_error_orientation_deg,"
                    "singular_values,"
                    "bad_singular_values\n")

  # Prepare folders.
  if not os.path.exists('dq_H_E'):
    os.makedirs('dq_H_E')
  if not os.path.exists('poses_H_E'):
    os.makedirs('poses_H_E')

  all_algorithm_configurations = [get_RANSAC_scalar_part_inliers_config(True),
                                  get_RANSAC_scalar_part_inliers_config(False),
                                  get_RANSAC_classic_config(True),
                                  get_RANSAC_classic_config(False),
                                  get_exhaustive_search_pose_inliers_config(),
                                  get_exhaustive_search_scalar_part_inliers_config(),
                                  get_baseline_config(True),
                                  get_baseline_config(False)]

  for (filtering_config, hand_eye_config) in all_algorithm_configurations:

    print("\n\n\nRun algorithm {}\n\n\n".format(hand_eye_config.algorithm_name))
    hand_eye_config.visualize = args.visualize
    hand_eye_config.plot_every_nth_pose = args.plot_every_nth_pose

    if hand_eye_config.enable_exhaustive_search or hand_eye_config.use_baseline_approach:
      num_iterations = 1
    else:
      num_iterations = args.num_iterations
    print("\n\n\nRun {} iterations...\n\n\n".format(num_iterations))
    for iteration in range(0, num_iterations):

      # Results:
      results_dataset_names = []
      results_success = []
      results_dq_H_E = []
      results_poses_H_E = []
      results_rmse = []
      results_num_inliers = []
      results_num_initial_poses = []
      results_num_poses_kept = []
      results_runtimes = []
      results_singular_values = []
      results_bad_singular_value = []

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
          plot_poses([poses_B_H[::every_nth_element], poses_W_E[::every_nth_element]],
                     True, title="3D Poses Before Alignment")

        print("Computing hand-eye calibration...")

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

        results_dataset_names.append((pose_file_B_H, pose_file_W_E))
        results_success.append(success)
        results_dq_H_E.append(dq_H_E)
        if dq_H_E is not None:
          results_poses_H_E.append(dq_H_E.to_pose())
        else:
          results_poses_H_E.append(None)
        results_rmse.append(rmse)
        results_num_inliers.append(num_inliers)
        results_num_initial_poses.append(len(dual_quat_B_H_vec))
        results_num_poses_kept.append(num_poses_kept)
        results_runtimes.append(runtime)
        results_singular_values.append(singular_values)
        results_bad_singular_value.append(bad_singular_values)

      assert len(results_dataset_names) == num_pose_pairs
      assert len(results_success) == num_pose_pairs
      assert len(results_dq_H_E) == num_pose_pairs
      assert len(results_poses_H_E) == num_pose_pairs
      assert len(results_rmse) == num_pose_pairs
      assert len(results_num_inliers) == num_pose_pairs
      assert len(results_num_initial_poses) == num_pose_pairs
      assert len(results_num_poses_kept) == num_pose_pairs
      assert len(results_runtimes) == num_pose_pairs

      assert len(results_singular_values) == num_pose_pairs
      assert len(results_bad_singular_value) == num_pose_pairs

      calibration_transformation_chain = []

      # Add point at origin to represent the first coordinate
      # frame in the chain of transformations.
      calibration_transformation_chain.append(
          DualQuaternion(Quaternion(0, 0, 0, 1), Quaternion(0, 0, 0, 0)))

      # Add first transformation
      calibration_transformation_chain.append(results_dq_H_E[0])

      # Create chain of transformations from the first frame to the last.
      i = 0
      idx = 0
      while i < (num_poses_sets - 2):
        idx += (num_poses_sets - i - 1)
        print("i: {}, idx: {}".format(i, idx))
        calibration_transformation_chain.append(results_dq_H_E[idx])
        i += 1

      # Add inverse of first to last frame to close the loop.
      calibration_transformation_chain.append(
          results_dq_H_E[num_poses_sets - 2].inverse())

      # Check loop.
      assert len(calibration_transformation_chain) == (num_poses_sets + 1), (
          len(calibration_transformation_chain), (num_poses_sets + 1))

      # Chain the transformations together to get points we can plot.
      poses_to_plot = []
      dq_tmp = DualQuaternion(Quaternion(0, 0, 0, 1), Quaternion(0, 0, 0, 0))
      for i in range(0, len(calibration_transformation_chain)):
        dq_tmp *= calibration_transformation_chain[i]
        poses_to_plot.append(dq_tmp.to_pose())

      # Compute error of loop.
      (loop_error_position, loop_error_orientation) = compute_pose_error(poses_to_plot[0],
                                                                         poses_to_plot[-1])
      print("Error when closing the loop of hand eye calibrations - position: {}"
            " m orientation: {} deg".format(loop_error_position, loop_error_orientation))

      if args.visualize:
        assert len(poses_to_plot) == len(calibration_transformation_chain)
        plot_poses([np.array(poses_to_plot)], plot_arrows=True,
                   title="Hand-Eye Calibration Results - Closing The Loop")

      output_file = open(args.result_file, 'a')
      for i in range(0, num_pose_pairs):
        dq_H_E_file = "./dq_H_E/{}_pose_pair_{}".format(
            hand_eye_config.algorithm_name, i) + "_dq_H_E.txt"
        poses_H_E_file = "./poses_H_E/{}_pose_pair_{}".format(
            hand_eye_config.algorithm_name, i) + "_pose_H_E.txt"

        if os.path.exists(dq_H_E_file):
          append_write = 'a'  # append if already exists
          assert os.path.exists(poses_H_E_file)
        else:
          append_write = 'w'  # make a new file if not
          assert not os.path.exists(poses_H_E_file)

        output_file_dq_H_E = open(
            dq_H_E_file, append_write)
        output_file_pose_H_E = open(
            poses_H_E_file, append_write)

        output_file_dq_H_E.write("{}\n".format(
            np.array_str(results_dq_H_E[i].dq, max_line_width=1000000)))
        output_file_pose_H_E.write("{}\n".format(
            np.array_str(results_poses_H_E[i], max_line_width=1000000)))
        output_file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            hand_eye_config.algorithm_name, i, iteration, hand_eye_config.prefilter_poses_enabled,
            results_dataset_names[i][0], results_dataset_names[i][1],
            results_success[i], results_rmse[i][0], results_rmse[i][1],
            results_num_inliers[i], results_num_initial_poses[i],
            results_num_poses_kept[i], results_runtimes[i], loop_error_position,
            loop_error_orientation, results_singular_values[i],
            results_bad_singular_value[i]))
