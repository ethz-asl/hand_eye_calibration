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
from hand_eye_calibration.csv_io import (write_time_stamped_poses_to_csv_file,
                                         read_time_stamped_poses_from_csv_file,
                                         write_double_numpy_array_to_csv_file)
from hand_eye_calibration.time_alignment import (calculate_time_offset,
                                                 compute_aligned_poses,
                                                 FilteringConfig)
from hand_eye_calibration.bash_utils import run
from hand_eye_calibration.calibration_verification import evaluate_calibration
from hand_eye_calibration_experiments.all_algorithm_configs import get_all_configs
from hand_eye_calibration_experiments.experiment_results import ResultEntry


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Align pairs of poses.')
  parser.add_argument(
      '--aligned_poses_B_H_csv_files', type=str, required=True,
      help='The path to the file containing the first poses. (e.g. Hand poses in Body frame)')
  parser.add_argument('--result_file', type=str, required=True,
                      help='The path to the result file.')

  parser.add_argument('--visualize', type=bool,
                      default=False, help='Visualize the poses.')
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
  if not os.path.exists(args.result_file):
    output_file = open(args.result_file, 'w')
    example_result_entry = ResultEntry()
    output_file.write(example_result_entry.get_header())

  # Prepare folders.
  if not os.path.exists('dq_H_E'):
    os.makedirs('dq_H_E')
  if not os.path.exists('poses_H_E'):
    os.makedirs('poses_H_E')

  all_algorithm_configurations = get_all_configs()

  for (algorithm_name, filtering_config, hand_eye_config, optimization_config) in all_algorithm_configurations:

    print("\n\n\nRun algorithm {}\n\n\n".format(hand_eye_config.algorithm_name))
    hand_eye_config.visualize = args.visualize
    hand_eye_config.plot_every_nth_pose = args.plot_every_nth_pose

    if hand_eye_config.enable_exhaustive_search or hand_eye_config.use_baseline_approach:
      num_iterations = 1
    else:
      num_iterations = args.num_iterations
    print("\n\n\nRun {} iterations...\n\n\n".format(num_iterations))
    for iteration in range(0, num_iterations):

      result_entry = ResultEntry()
      result_entry.init_from_configs(algorithm_name, filtering_config,
                                     hand_eye_config, optimization_config)

      results_dq_H_E = []
      results_poses_H_E = []

      pose_pair_num = 0
      for (pose_file_B_H, pose_file_W_E) in set_of_pose_pairs:
        print("\n\nHand-calibration between \n\t{}\n and \n\t{}\n\n".format(
            pose_file_B_H, pose_file_W_E))

        # Define output file paths.
        initial_guess_calibration_file = ("./optimization/{}_pose_pair_{}_it_{}_" +
                                          "init_guess.json").format(
            hand_eye_config.algorithm_name, pose_pair_num, iteration)

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

        if not optimization_config.optimization_only:

          print("Computing time offset...")
          time_offset = calculate_time_offset(times_B_H, quaternions_B_H, times_W_E,
                                              quaternions_W_E, filtering_config, args.visualize)

          print("Time offset: {}s".format(time_offset))

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

          # Write results to file for the optimization.
          calib = ExtrinsicCalibration(
              time_offset, dq_H_E)
          calib.writeJson(initial_guess_calibration_file)

        else:  # Run optimization only mode.
          # No time offset.
          time_offset_initial_guess = 0
          # Unit DualQuaternion.
          dq_H_E_initial_guess = DualQuaternion.from_vector(
              [0., 0., 0., 1.0, 0., 0., 0., 0.])

          # Write some initial guess into the optimization input file.
          initial_guess_calibration = ExtrinsicCalibration(
              time_offset_initial_guess, dq_H_E_initial_guess)
          initial_guess_calibration.writeJson(initial_guess_calibration_file)

        # Fill in result entries.
        result_entry.dataset_names.append((pose_file_B_H, pose_file_W_E))
        result_entry.success.append(success)
        result_entry.num_initial_poses.append(len(dual_quat_B_H_vec))
        result_entry.num_poses_kept.append(num_poses_kept)
        result_entry.runtimes.append(runtime)
        result_entry.singular_values.append(singular_values)
        result_entry.bad_singular_value.append(bad_singular_values)

        # Run optimization if enabled.
        if optimization_config.optimization_enabled:
          optimized_calibration_file = "./optimization/{}_pose_pair_{}_it_{}_optimized.json".format(
              hand_eye_config.algorithm_name, pose_pair_num, iteration)

          # Init optimization result
          time_offset_optimized = None
          dq_H_E_optimized = None
          optimization_success = True
          rmse_optimized = (-1, -1)
          num_inliers_optimized = 0

          try:
            run("rosrun hand_eye_calibration_batch_estimation batch_estimator -v 1 \
              --pose1_csv=%s --pose2_csv=%s \
              --initial_guess_calibration_file=%s \
              --output_file=%s" % (pose_file_B_H, pose_file_W_E,
                                   initial_guess_calibration_file,
                                   optimized_calibration_file))
          except Exception as ex:
            print("Optimization failed: {}".format(exception))
            optimization_success = False

          # If the optimization was successful, evaluate it.
          if optimization_success:
            optimized_calibration = ExtrinsicCalibration.fromJson(
                optimized_calibration_file)
            initial_guess_calibration = ExtrinsicCalibration.fromJson(
                initial_guess_calibration_file)
            print("Optimized calibration: \n", initial_guess_calibration,
                  "->\n", optimized_calibration)

            dq_H_E_optimized = optimized_calibration.pose_dq
            time_offset_optimized = optimized_calibration.time_offset

            (rmse_optimized,
             num_inliers_optimized) = evaluate_calibration(time_stamped_poses_B_H,
                                                           time_stamped_poses_W_E,
                                                           dq_H_E_optimized,
                                                           time_offset_optimized)

          # Store results.
          result_entry.optimization_success.append(optimization_success)
          result_entry.rmse.append(rmse_optimized)
          result_entry.num_inliers.append(num_inliers_optimized)

          results_poses_H_E.append(dq_H_E_optimized.to_pose())
          results_dq_H_E.append(dq_H_E_optimized)

        else:  # Use result of initial algorithms without optimization

          if dq_H_E is not None:
            results_poses_H_E.append(dq_H_E.to_pose())
          else:
            results_poses_H_E.append(None)
          results_dq_H_E.append(dq_H_E)

          result_entry.rmse.append(rmse)
          result_entry.num_inliers.append(num_inliers)

        pose_pair_num = pose_pair_num + 1

      # Verify results.
      assert len(results_dq_H_E) == num_pose_pairs
      assert len(results_poses_H_E) == num_pose_pairs

      assert len(result_entry.dataset_names) == num_pose_pairs
      assert len(result_entry.success) == num_pose_pairs
      assert len(result_entry.rmse) == num_pose_pairs
      assert len(result_entry.num_inliers) == num_pose_pairs
      assert len(result_entry.num_initial_poses) == num_pose_pairs
      assert len(result_entry.num_poses_kept) == num_pose_pairs
      assert len(result_entry.runtimes) == num_pose_pairs
      assert len(result_entry.singular_values) == num_pose_pairs
      assert len(result_entry.bad_singular_value) == num_pose_pairs

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
      (result_entry.loop_error_position,
       result_entry.loop_error_orientation) = compute_pose_error(poses_to_plot[0],
                                                                 poses_to_plot[-1])
      print("Error when closing the loop of hand eye calibrations - position: {}"
            " m orientation: {} deg".format(result_entry.loop_error_position,
                                            result_entry.loop_error_orientation))

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
        output_file.write(result_entry.write_pose_pair_to_csv_line(i))
