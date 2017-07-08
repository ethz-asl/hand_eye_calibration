#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import itertools
import os
import errno
import sys
import timeit


import numpy as np

from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.quaternion import (
    Quaternion, angle_between_quaternions)
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
from hand_eye_calibration.extrinsic_calibration import ExtrinsicCalibration
from hand_eye_calibration.bash_utils import (run, create_path)
from hand_eye_calibration.calibration_verification import (
    evaluate_calibration, compute_loop_error)
from hand_eye_calibration_experiments.all_algorithm_configs import get_all_configs
from hand_eye_calibration_experiments.experiment_results import ResultEntry

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Align pairs of poses.')
  parser.add_argument(
      '--aligned_poses_B_H_csv_files', nargs='+', type=str, required=True,
      help='The path to the file containing the first poses. (e.g. Hand poses in Body frame)')
  parser.add_argument(
      '--is_absolute_pose_sensor', nargs='+', type=int,
      help=("For each file passed to - -aligned_poses_B_H_csv_files specify if the pose sensor "
            "is absolute or not. E.g. --is_absolute_pose_sensor 0 1 1"))
  parser.add_argument('--result_directory', type=str, required=True,
                      help='The path to the result directory.')

  parser.add_argument('--visualize', type=bool,
                      default=False, help='Visualize the poses.')
  parser.add_argument('--plot_every_nth_pose', type=int,
                      help='Plot only every n-th pose.', default=10)
  parser.add_argument('--num_iterations', type=int,
                      help='Number of iterations per algorithm.'
                      'Only applicable to non - deterministic algorithms!', default=2)
  args = parser.parse_args()

  print(args.aligned_poses_B_H_csv_files)
  print(args.is_absolute_pose_sensor)

  assert args.aligned_poses_B_H_csv_files is not None

  poses_B_H_csv_files = args.aligned_poses_B_H_csv_files
  num_poses_sets = len(poses_B_H_csv_files)
  assert num_poses_sets >= 2, "Needs at least two sets of poses."

  # If nothing was specified, assume they are all not absolute.
  is_absolute_pose_sensor_flags = None
  if args.is_absolute_pose_sensor is None:
    is_absolute_pose_sensor_flags = [False] * num_poses_sets
  else:
    is_absolute_pose_sensor_flags = [bool(flag)
                                     for flag in args.is_absolute_pose_sensor]
  assert len(is_absolute_pose_sensor_flags) == num_poses_sets

  set_of_pose_pairs = list(itertools.combinations(poses_B_H_csv_files, 2))
  set_of_is_absolute_sensor_flags = list(
      itertools.combinations(is_absolute_pose_sensor_flags, 2))
  num_pose_pairs = len(set_of_pose_pairs)
  assert len(set_of_pose_pairs) == len(set_of_is_absolute_sensor_flags)

  # Prepare result file.
  result_file_path = args.result_directory + "/results.csv"
  create_path(result_file_path)

  if not os.path.exists(result_file_path):
    output_file = open(result_file_path, 'w')
    example_result_entry = ResultEntry()
    output_file.write(example_result_entry.get_header())

  # Prepare folders.
  if not os.path.exists('dq_H_E'):
    os.makedirs('dq_H_E')
  if not os.path.exists('poses_H_E'):
    os.makedirs('poses_H_E')

  all_algorithm_configurations = get_all_configs()

  for (algorithm_name, filtering_config, hand_eye_config, optimization_config) in all_algorithm_configurations:
    print("\n\n\nRun algorithm {}\n\n\n".format(algorithm_name))
    hand_eye_config.visualize = args.visualize
    hand_eye_config.plot_every_nth_pose = args.plot_every_nth_pose

    if hand_eye_config.enable_exhaustive_search or hand_eye_config.use_baseline_approach:
      num_iterations = 1
    else:
      num_iterations = args.num_iterations
    print("\n\n\nRun {} iterations...\n\n\n".format(num_iterations))
    for iteration in range(0, num_iterations):
      result_entry = ResultEntry()
      result_entry.init_from_configs(algorithm_name, iteration, filtering_config,
                                     hand_eye_config, optimization_config)

      results_dq_H_E = []
      results_poses_H_E = []

      pose_pair_idx = 0

      for ((pose_file_B_H, pose_file_W_E), (is_absolute_sensor_B_H, is_absolute_sensor_W_E)) in zip(set_of_pose_pairs, set_of_is_absolute_sensor_flags):
        print("\n\nHand-calibration between \n\t{} (absolute: {})\n and \n\t{} (absolute: {})\n\n".format(
            pose_file_B_H, is_absolute_sensor_B_H, pose_file_W_E, is_absolute_sensor_W_E))

        # Define output file paths.
        initial_guess_calibration_file = ("{}/optimization/{}_pose_pair_{}_it_{}_" +
                                          "init_guess.json").format(args.result_directory,
                                                                    algorithm_name,
                                                                    pose_pair_idx, iteration)
        create_path(initial_guess_calibration_file)

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

        # No time offset.
        time_offset_initial_guess = 0.
        # Unit DualQuaternion.
        dq_H_E_initial_guess = DualQuaternion.from_vector(
            [0., 0., 0., 1.0, 0., 0., 0., 0.])

        print("Computing time offset...")
        time_offset_initial_guess = calculate_time_offset(times_B_H, quaternions_B_H, times_W_E,
                                                          quaternions_W_E, filtering_config, args.visualize)

        print("Time offset: {}s".format(time_offset_initial_guess))

        print("Computing aligned poses...")
        (aligned_poses_B_H, aligned_poses_W_E) = compute_aligned_poses(
            time_stamped_poses_B_H, time_stamped_poses_W_E, time_offset_initial_guess, args.visualize)

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
          (success, dq_H_E_initial_guess, rmse,
           num_inliers, num_poses_kept,
           runtime, singular_values, bad_singular_values) = compute_hand_eye_calibration_BASELINE(
              dual_quat_B_H_vec, dual_quat_W_E_vec, hand_eye_config)
        else:
          (success, dq_H_E_initial_guess, rmse,
           num_inliers, num_poses_kept,
           runtime, singular_values, bad_singular_values) = compute_hand_eye_calibration_RANSAC(
              dual_quat_B_H_vec, dual_quat_W_E_vec, hand_eye_config)

        # Fill in result entries.
        result_entry.success.append(success)
        result_entry.num_initial_poses.append(len(dual_quat_B_H_vec))
        result_entry.num_poses_kept.append(num_poses_kept)
        result_entry.runtimes.append(runtime)
        result_entry.singular_values.append(singular_values)
        result_entry.bad_singular_value.append(bad_singular_values)
        result_entry.dataset_names.append((pose_file_B_H, pose_file_W_E))

        # Run optimization if enabled.
        if optimization_config.enable_optimization:
          # Write initial guess to file for the optimization.
          initial_guess = ExtrinsicCalibration(
              time_offset_initial_guess, dq_H_E_initial_guess)
          initial_guess.writeJson(initial_guess_calibration_file)

          optimized_calibration_file = "{}/optimization/{}_pose_pair_{}_it_{}_optimized.json".format(
              args.result_directory, algorithm_name, pose_pair_idx, iteration)
          create_path(optimized_calibration_file)

          # Init optimization result
          time_offset_optimized = None
          dq_H_E_optimized = None
          optimization_success = True
          rmse_optimized = (-1, -1)
          num_inliers_optimized = 0

          model_config_string = ("pose1/absoluteMeasurements={},"
                                 "pose2/absoluteMeasurements={}").format(
              is_absolute_sensor_B_H, is_absolute_sensor_W_E).lower()

          optimization_runtime = 0.
          try:
            optimization_start_time = timeit.default_timer()
            run("rosrun hand_eye_calibration_batch_estimation batch_estimator -v 1 \
              --pose1_csv=%s --pose2_csv=%s \
              --init_guess_file=%s \
              --output_file=%s \
              --model_config=%s" % (pose_file_B_H, pose_file_W_E,
                                    initial_guess_calibration_file,
                                    optimized_calibration_file,
                                    model_config_string))
            optimization_end_time = timeit.default_timer()
            optimization_runtime = optimization_end_time - optimization_start_time

          except Exception as ex:
            print("Optimization failed: {}".format(ex))
            optimization_success = False

          # If the optimization was successful, evaluate it.
          if optimization_success:
            optimized_calibration = ExtrinsicCalibration.fromJson(
                optimized_calibration_file)

            dq_H_E_optimized = optimized_calibration.pose_dq
            time_offset_optimized = optimized_calibration.time_offset

            print("Initial guess time offset: \t{}".format(
                time_offset_initial_guess))
            print("Optimized time offset: \t\t{}".format(time_offset_optimized))
            print("Initial guess dq_H_E: \t\t{}".format(dq_H_E_initial_guess))
            print("Optimized dq_H_E: \t\t{}".format(dq_H_E_optimized))

            (rmse_optimized,
             num_inliers_optimized) = evaluate_calibration(time_stamped_poses_B_H,
                                                           time_stamped_poses_W_E,
                                                           dq_H_E_optimized,
                                                           time_offset_optimized,
                                                           hand_eye_config)
            if num_inliers_optimized == 0:
              print("Could not evaluate calibration, no matching poses found!")
              optimization_success = False
            else:
              print("Solution found by optimization\n"
                    "\t\tNumber of inliers: {}\n"
                    "\t\tRMSE position:     {:10.4f}\n"
                    "\t\tRMSE orientation:  {:10.4f}".format(num_inliers_optimized,
                                                             rmse_optimized[0],
                                                             rmse_optimized[1]))

          # Store results.
          if dq_H_E_optimized is not None:
            results_poses_H_E.append(dq_H_E_optimized.to_pose())
          else:
            results_poses_H_E.append(None)
          results_dq_H_E.append(dq_H_E_optimized)

          # Not relevant for this experiment, so we add None.
          result_entry.spoiled_initial_guess_angle_offset.append(None)
          result_entry.spoiled_initial_guess_translation_offset.append(None)
          result_entry.spoiled_initial_guess_time_offset.append(None)

          result_entry.optimization_success.append(optimization_success)
          result_entry.rmse.append(rmse_optimized)
          result_entry.num_inliers.append(num_inliers_optimized)
          result_entry.optimization_runtime.append(optimization_runtime)

        else:  # Use result of initial algorithms without optimization
          if dq_H_E_initial_guess is not None:
            results_poses_H_E.append(dq_H_E_initial_guess.to_pose())
          else:
            results_poses_H_E.append(None)
          results_dq_H_E.append(dq_H_E_initial_guess)

          # Not relevant for this experiment, so we add None.
          result_entry.spoiled_initial_guess_angle_offset.append(None)
          result_entry.spoiled_initial_guess_translation_offset.append(None)
          result_entry.spoiled_initial_guess_time_offset.append(None)

          result_entry.rmse.append(rmse)
          result_entry.num_inliers.append(num_inliers)
          result_entry.optimization_success.append(True)
          result_entry.optimization_runtime.append(0.)

        pose_pair_idx = pose_pair_idx + 1

      # Verify results.
      assert len(results_dq_H_E) == num_pose_pairs
      assert len(results_poses_H_E) == num_pose_pairs
      result_entry.check_length(num_pose_pairs)

      # Computing the loop error only makes sense if all pose pairs are successfully calibrated.
      # If optimization is disabled, optimization_success should always be True.
      if sum(result_entry.optimization_success) == num_pose_pairs:
        (result_entry.loop_error_position,
         result_entry.loop_error_orientation) = compute_loop_error(results_dq_H_E,
                                                                   num_poses_sets,
                                                                   args.visualize)
      else:
        print("Error: No loop error computed because not all pose pairs were successfully calibrated!")

      # Write to result files.
      output_file = open(result_file_path, 'a')
      for i in range(0, num_pose_pairs):
        dq_H_E_file = "{}/dq_H_E/{}_pose_pair_{}_dq_H_E.txt".format(args.result_directory,
                                                                    algorithm_name, i)
        create_path(dq_H_E_file)
        poses_H_E_file = "{}/poses_H_E/{}_pose_pair_{}_pose_H_E.txt".format(args.result_directory,
                                                                            algorithm_name, i)
        create_path(poses_H_E_file)

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

        if results_dq_H_E[i] is not None:
          output_file_dq_H_E.write("{}\n".format(
              np.array_str(results_dq_H_E[i].dq, max_line_width=1000000)))
          output_file_pose_H_E.write("{}\n".format(
              np.array_str(results_poses_H_E[i], max_line_width=1000000)))

        output_file.write(result_entry.write_pose_pair_to_csv_line(i))
