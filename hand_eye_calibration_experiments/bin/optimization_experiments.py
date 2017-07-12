#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import itertools
import os
import errno
import sys
import timeit
import random

from functools import partial
import numpy as np
from multiprocessing import Pool

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
from hand_eye_calibration_experiments.all_algorithm_configs import get_optimization_with_spoiled_initial_calibration_config
from hand_eye_calibration_experiments.experiment_results import ResultEntry


def Forwarder(args, func):
  return func(*args)


def spoil_initial_guess(time_offset_initial_guess, dq_H_E_initial_guess, angular_offset_range,
                        translation_offset_range, time_offset_range):
  """ Apply a random perturbation to the calibration."""

  random_time_offset_offset = np.random.uniform(
      time_offset_range[0], time_offset_range[1]) * random.choice([-1., 1.])

  # Get a random unit vector
  random_translation_offset = np.random.uniform(-1.0, 1.0, 3)
  random_translation_offset /= np.linalg.norm(random_translation_offset)
  assert np.isclose(np.linalg.norm(random_translation_offset), 1., atol=1e-8)

  # Scale unit vector to a random length between 0 and max_translation_offset.
  random_translation_length = np.random.uniform(
      translation_offset_range[0], translation_offset_range[1])
  random_translation_offset *= random_translation_length

  # Get orientation offset of at most max_angular_offset.
  random_quaternion_offset = Quaternion.get_random(
      angular_offset_range[0], angular_offset_range[1])

  translation_initial_guess = dq_H_E_initial_guess.to_pose()[0: 3]
  orientation_initial_guess = dq_H_E_initial_guess.q_rot

  time_offset_spoiled = time_offset_initial_guess + random_time_offset_offset
  translation_spoiled = random_translation_offset + translation_initial_guess
  orientation_spoiled = random_quaternion_offset * orientation_initial_guess

  # Check if we spoiled correctly.
  random_angle_offset = angle_between_quaternions(
      orientation_initial_guess, orientation_spoiled)

  assert random_angle_offset <= angular_offset_range[1]
  assert random_angle_offset >= angular_offset_range[0]

  assert np.linalg.norm(translation_spoiled -
                        translation_initial_guess) <= translation_offset_range[1]
  assert np.linalg.norm(translation_spoiled -
                        translation_initial_guess) >= translation_offset_range[0]

  # Get random orientation that distorts the original orientation by at most
  # max_angular_offset rad.
  dq_H_E_spoiled = DualQuaternion.from_pose(
      translation_spoiled[0], translation_spoiled[1], translation_spoiled[2],
      orientation_spoiled.q[0], orientation_spoiled.q[1],
      orientation_spoiled.q[2], orientation_spoiled.q[3])

  print("dq_H_E_initial_guess: {}".format(dq_H_E_initial_guess.dq))
  print("dq_H_E_spoiled: {}".format(dq_H_E_spoiled.dq))

  print("time_offset_initial_guess: {}".format(time_offset_initial_guess))
  print("time_offset_spoiled: {}".format(time_offset_spoiled))

  print("random_quaternion_offset: {}".format(random_quaternion_offset.q))

  return (time_offset_spoiled, dq_H_E_spoiled,
          (random_translation_offset, random_angle_offset, random_time_offset_offset))


def compute_initial_guess_for_all_pairs(set_of_pose_pairs, algorithm_name, hand_eye_config,
                                        filtering_config, optimization_config, visualize=False):
  """
  Iterate over all pairs and compute an initial guess for the calibration (both time and
  transformation). Also store the experiment results from this computation for each pair.
  """

  set_of_dq_H_E_initial_guess = []
  set_of_time_offset_initial_guess = []

  # Initialize the result entry.
  result_entry = ResultEntry()
  result_entry.init_from_configs(algorithm_name, 0, filtering_config,
                                 hand_eye_config, optimization_config)

  for (pose_file_B_H, pose_file_W_E) in set_of_pose_pairs:
    print("\n\nCompute initial guess for calibration between \n\t{} \n and \n\t{} \n\n".format(
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

    # No time offset.
    time_offset_initial_guess = 0.
    # Unit DualQuaternion.
    dq_H_E_initial_guess = DualQuaternion.from_vector(
        [0., 0., 0., 1.0, 0., 0., 0., 0.])

    print("Computing time offset...")
    time_offset_initial_guess = calculate_time_offset(times_B_H, quaternions_B_H, times_W_E,
                                                      quaternions_W_E, filtering_config,
                                                      args.visualize)

    print("Time offset: {}s".format(time_offset_initial_guess))

    print("Computing aligned poses...")
    (aligned_poses_B_H, aligned_poses_W_E) = compute_aligned_poses(
        time_stamped_poses_B_H, time_stamped_poses_W_E, time_offset_initial_guess, visualize)

    # Convert poses to dual quaterions.
    dual_quat_B_H_vec = [DualQuaternion.from_pose_vector(
        aligned_pose_B_H) for aligned_pose_B_H in aligned_poses_B_H[:, 1:]]
    dual_quat_W_E_vec = [DualQuaternion.from_pose_vector(
        aligned_pose_W_E) for aligned_pose_W_E in aligned_poses_W_E[:, 1:]]

    assert len(dual_quat_B_H_vec) == len(dual_quat_W_E_vec), ("len(dual_quat_B_H_vec): {} "
                                                              "vs len(dual_quat_W_E_vec): {}"
                                                              ).format(len(dual_quat_B_H_vec),
                                                                       len(dual_quat_W_E_vec))

    dual_quat_B_H_vec = align_paths_at_index(dual_quat_B_H_vec)
    dual_quat_W_E_vec = align_paths_at_index(dual_quat_W_E_vec)

    # Draw both paths in their Global / World frame.
    if visualize:
      poses_B_H = np.array([dual_quat_B_H_vec[0].to_pose().T])
      poses_W_E = np.array([dual_quat_W_E_vec[0].to_pose().T])
      for i in range(1, len(dual_quat_B_H_vec)):
        poses_B_H = np.append(poses_B_H, np.array(
            [dual_quat_B_H_vec[i].to_pose().T]), axis=0)
        poses_W_E = np.append(poses_W_E, np.array(
            [dual_quat_W_E_vec[i].to_pose().T]), axis=0)
      every_nth_element = args.plot_every_nth_pose
      plot_poses([poses_B_H[:: every_nth_element], poses_W_E[:: every_nth_element]],
                 True, title="3D Poses Before Alignment")

    print("Computing hand-eye calibration to obtain an initial guess...")

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

    result_entry.success.append(success)
    result_entry.num_initial_poses.append(len(dual_quat_B_H_vec))
    result_entry.num_poses_kept.append(num_poses_kept)
    result_entry.runtimes.append(runtime)
    result_entry.singular_values.append(singular_values)
    result_entry.bad_singular_value.append(bad_singular_values)
    result_entry.dataset_names.append((pose_file_B_H, pose_file_W_E))

    set_of_dq_H_E_initial_guess.append(dq_H_E_initial_guess)
    set_of_time_offset_initial_guess.append(time_offset_initial_guess)

  return (set_of_dq_H_E_initial_guess,
          set_of_time_offset_initial_guess, result_entry)


def run_optimization_experiment(time_offset_range, angle_offset_range,
                                translation_offset_range, iteration_idx,
                                result_entry_template, experiment_progress):
  result_entry = copy.deepcopy(result_entry_template)
  result_entry.iteration_num = iteration_idx

  # Init result variables.
  results_dq_H_E = []
  results_poses_H_E = []

  pose_pair_idx = 0
  # Perform optimization on all pairs.
  for ((pose_file_B_H, pose_file_W_E),
       (is_absolute_sensor_B_H, is_absolute_sensor_W_E),
       dq_H_E_initial_guess,
       time_offset_initial_guess) in zip(set_of_pose_pairs,
                                         set_of_is_absolute_sensor_flags,
                                         set_of_dq_H_E_initial_guess,
                                         set_of_time_offset_initial_guess):

    print(("\n\nCompute optimized hand-eye calibration between \n\t{} (absolute: {})\n " +
           "and \n\t{} (absolute: {})\n\n").format(
        pose_file_B_H, is_absolute_sensor_B_H, pose_file_W_E, is_absolute_sensor_W_E))

    # Load data.
    (time_stamped_poses_B_H, _,
     _) = read_time_stamped_poses_from_csv_file(pose_file_B_H)
    print("Found ", time_stamped_poses_B_H.shape[0],
          " poses in file: ", pose_file_B_H)

    (time_stamped_poses_W_E, _,
     _) = read_time_stamped_poses_from_csv_file(pose_file_W_E)
    print("Found ", time_stamped_poses_W_E.shape[0],
          " poses in file: ", pose_file_W_E)

    # Spoil initial guess, for current pair.
    (time_offset_initial_guess_spoiled, dq_H_E_initial_guess_spoiled,
     (random_translation_offset, random_angle_offset,
      random_time_offset_offset)) = spoil_initial_guess(
        time_offset_initial_guess, dq_H_E_initial_guess,
        angle_offset_range,
        translation_offset_range,
        time_offset_range)

    print(("Spoiled initial guess: time offset: {} angle offset: {} " +
           "translation offset: {}").format(
        random_time_offset_offset, random_angle_offset, random_translation_offset))

    # Save offsets from initial guess.
    result_entry.spoiled_initial_guess_angle_offset.append(
        random_angle_offset)
    result_entry.spoiled_initial_guess_translation_offset.append(
        random_translation_offset)
    result_entry.spoiled_initial_guess_time_offset.append(
        random_time_offset_offset)

    # Write calibration to optimization input file format.
    initial_guess_calibration_file = ("{}/optimization/{}_run_{}_it_{}_pose_pair_{}_" +
                                      "dt_{}_ds_{}_da_{}_" +
                                      "init_guess.json"
                                      ).format(args.result_directory,
                                               algorithm_name,
                                               experiment_progress[0],
                                               iteration,
                                               pose_pair_idx,
                                               random_time_offset_offset,
                                               np.linalg.norm(
                                                   random_translation_offset),
                                               random_angle_offset)
    create_path(initial_guess_calibration_file)
    initial_guess = ExtrinsicCalibration(
        time_offset_initial_guess_spoiled, dq_H_E_initial_guess_spoiled)
    initial_guess.writeJson(initial_guess_calibration_file)

    # Init optimization result
    time_offset_optimized = None
    dq_H_E_optimized = None
    optimization_success = True
    rmse_optimized = (-1, -1)
    num_inliers_optimized = 0

    model_config_string = ("pose1/absoluteMeasurements={},"
                           "pose2/absoluteMeasurements={}").format(
        is_absolute_sensor_B_H, is_absolute_sensor_W_E).lower()

    # Prepare output file path and folder.
    optimized_calibration_file = ("{}/optimization/{}_run_{}_it_{}_pose_pair_{}_" +
                                  "dt_{}_ds_{}_da_{}_" +
                                  "optimized.json").format(args.result_directory,
                                                           algorithm_name,
                                                           experiment_progress[0],
                                                           iteration,
                                                           pose_pair_idx,
                                                           random_time_offset_offset,
                                                           np.linalg.norm(
                                                               random_translation_offset),
                                                           random_angle_offset)
    create_path(optimized_calibration_file)

    # Run the optimization.
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
      print("Optimized time offset: \t\t{}".format(
          time_offset_optimized))
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

    result_entry.optimization_success.append(optimization_success)
    result_entry.rmse.append(rmse_optimized)
    result_entry.num_inliers.append(num_inliers_optimized)
    result_entry.optimization_runtime.append(optimization_runtime)

    pose_pair_idx = pose_pair_idx + 1

  # Verify results.
  assert len(results_dq_H_E) == num_pose_pairs
  assert len(results_poses_H_E) == num_pose_pairs
  result_entry.check_length(num_pose_pairs)

  # Computing the loop error only makes sense if all pose pairs are successfully calibrated.
  # If optimization is disabled, optimization_success should always be
  # True.
  if sum(result_entry.optimization_success) == num_pose_pairs:
    (result_entry.loop_error_position,
     result_entry.loop_error_orientation) = compute_loop_error(results_dq_H_E,
                                                               num_poses_sets,
                                                               args.visualize)
  else:
    print("Error: No loop error computed because not " +
          "all pose pairs were successfully calibrated!")

  print("\n\nFINISHED EXPERIMENT {}/{}\n\n".format(experiment_progress[0],
                                                   experiment_progress[1]))

  return (result_entry, results_dq_H_E, results_poses_H_E)


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
  result_file_path = args.result_directory + "/results_optimization.csv"
  create_path(result_file_path)

  if not os.path.exists(result_file_path):
    output_file = open(result_file_path, 'w')
    example_result_entry = ResultEntry()
    output_file.write(example_result_entry.get_header())

  # Get basic configuration.
  (algorithm_name, filtering_config, hand_eye_config,
   optimization_config) = get_optimization_with_spoiled_initial_calibration_config()

  # Define parameter ranges for experiment.
  plot_type = "angle_translation_complete"

  if plot_type == "time_angle":
    time_offset_ranges = [[0., 0.03], [0.03, 0.06], [0.06, 0.09], [0.09, 0.12],
                          [0.12, 0.15], [0.15, 0.18], [0.18, 0.21], [0.21, 0.24]]

    angle_offset_ranges = [[0., 15.], [15., 30.], [30., 45.], [45., 60.],
                           [60., 75.], [75., 90.], [90., 105.], [105., 120.]]

    translation_offset_ranges = [[0., 0.1]]

  elif plot_type == "time_translation":
    time_offset_ranges = [[0., 0.03], [0.03, 0.06], [0.06, 0.09], [0.09, 0.12],
                          [0.12, 0.15], [0.15, 0.18], [0.18, 0.21], [0.21, 0.24]]

    angle_offset_ranges = [[0., 15.]]

    translation_offset_ranges = [[0., 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4],
                                 [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8]]

  elif plot_type == "angle_translation":
    time_offset_ranges = [[0., 0.03]]

    angle_offset_ranges = [[0., 15.], [15., 30.], [30., 45.], [45., 60.],
                           [60., 75.], [75., 90.], [90., 105.], [105., 120.]]

    translation_offset_ranges = [[0., 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4],
                                 [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8]]
  elif plot_type == "angle_translation_complete":
    time_offset_ranges = [[0., 0.]]

    angle_offset_ranges = [[0., 15.], [15., 30.], [30., 45.], [45., 60.],
                           [60., 75.], [75., 90.], [90., 105.], [105., 120.],
                           [120., 135.], [135., 150.], [150., 165.], [165., 180.]]

    translation_offset_ranges = [[0., 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4],
                                 [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8],
                                 [0.8, 0.9],  [0.9, 1.0]]
  else:
    assert False, "Unkown plot type: {}".format(plot_type)

  # Convert degrees to rad.
  for angle_offset_range in angle_offset_ranges:
    angle_offset_range[0] = angle_offset_range[0] / 180. * np.math.pi
    angle_offset_range[1] = angle_offset_range[1] / 180. * np.math.pi

  # Compute initial guess which will be used as a basis for the spoiled
  # initial guess.
  (set_of_dq_H_E_initial_guess,
   set_of_time_offset_initial_guess,
   result_entry_template) = compute_initial_guess_for_all_pairs(set_of_pose_pairs,
                                                                algorithm_name,
                                                                hand_eye_config,
                                                                filtering_config,
                                                                optimization_config,
                                                                args.visualize)

  # Loop over experiment values
  number_of_experiment_exec = (len(time_offset_ranges) * len(angle_offset_ranges) *
                               len(translation_offset_ranges) * args.num_iterations)

  print("\n\nEXPERIMENT SIZE {}\n\n".format(number_of_experiment_exec))

  all_result_entries = [None] * number_of_experiment_exec
  experiment_progress_idx = 0

  input_data = []

  for time_offset_range in time_offset_ranges:
    for angle_offset_range in angle_offset_ranges:
      for translation_offset_range in translation_offset_ranges:
        assert args.num_iterations > 0
        for iteration in range(0, args.num_iterations):
          assert experiment_progress_idx < number_of_experiment_exec

          input_data.append((time_offset_range, angle_offset_range,
                             translation_offset_range, iteration, result_entry_template, (experiment_progress_idx, number_of_experiment_exec)))

          experiment_progress_idx += 1
#       || end_loop: iteration
#     || end_loop: translation offset
#   || end_loop: angle offset
# || end_loop: time offset

  assert experiment_progress_idx == number_of_experiment_exec
  assert len(input_data) == number_of_experiment_exec

  thread_pool = Pool(min(8, number_of_experiment_exec))
  func_wrapped = partial(Forwarder, func=run_optimization_experiment)
  output_data = thread_pool.map(func_wrapped, input_data)
  assert len(output_data) == number_of_experiment_exec
  thread_pool.close()

  print("\n\nFINISHED ALL EXPERIMENTS!\n\n")

  output_file = open(result_file_path, 'a')
  for result_idx in range(0, number_of_experiment_exec):
    # Write the results for all pairs for the current iteration to the
    # result file.
    output_data_entry = output_data[result_idx]
    for pair_idx in range(0, num_pose_pairs):
      output_file.write(
          output_data_entry[0].write_pose_pair_to_csv_line(pair_idx))

  print("\n\nFINISHED WRITING RESULTS!\n\n")
