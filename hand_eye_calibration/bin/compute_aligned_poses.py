#!/usr/bin/env python
from hand_eye_calibration.time_alignment import (
    calculate_time_offset, compute_aligned_poses, FilteringConfig)
from hand_eye_calibration.quaternion import Quaternion
from hand_eye_calibration.csv_io import (
    write_time_stamped_poses_to_csv_file,
    read_time_stamped_poses_from_csv_file)

import argparse
import numpy as np

if __name__ == '__main__':
  """
  Perform time alignment between two timestamped sets of poses and
  compute time-aligned poses pairs.

  The CSV files should have the following line format:
    timestamp [s], x [m], y [m], z [m], qx, qy, qz, qw

  The resulting aligned poses follow the same format.
  """

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--poses_B_H_csv_file',
      required=True,
      help='The CSV file containing the first time stamped poses. (e.g. Hand poses in Body frame)')
  parser.add_argument(
      '--poses_W_E_csv_file',
      required=True,
      help='The CSV file containing the second time stamped poses. (e.g. Eye poses in World frame)')

  parser.add_argument(
      '--aligned_poses_B_H_csv_file',
      required=True,
      help='Path to the CSV file where the aligned poses will be stored. (e.g. Hand poses in Body frame)')
  parser.add_argument(
      '--aligned_poses_W_E_csv_file',
      required=True,
      help='Path to the CSV file where the aligned poses will be stored. (e.g. Eye poses in World frame)')

  parser.add_argument(
      '--time_offset_output_csv_file', type=str,
      help='Write estimated time offset to this file in spatial-extrinsics csv format')

  parser.add_argument(
      '--quaternion_format',
      default='Hamilton',
      help='\'Hamilton\' [Default] or \'JPL\'. The input (and data in the output files) ' +
      'will be converted to Hamiltonian quaternions.')

  parser.add_argument('--visualize', type=bool, default=False,
                      help='Visualize the poses.')

  args = parser.parse_args()

  use_JPL_quaternion = False
  if args.quaternion_format == 'JPL':
    print("Input quaternion format was set to JPL. The input (and output of "
          "this script) will be converted to Hamiltonian quaternions.")
    use_JPL_quaternion = True
  elif args.quaternion_format == 'Hamilton':
    print("Input quaternion format was set to Hamilton.")
    use_JPL_quaternion = False
  else:
    assert False, "Unknown quaternion format: \'{}\'".format(args.quaternion_format)

  print("Reading CSV files...")
  (time_stamped_poses_B_H, times_B_H,
   quaternions_B_H) = read_time_stamped_poses_from_csv_file(
       args.poses_B_H_csv_file, use_JPL_quaternion)
  print("Found ", time_stamped_poses_B_H.shape[
      0], " poses in file: ", args.poses_B_H_csv_file)

  (time_stamped_poses_W_E, times_W_E,
   quaternions_W_E) = read_time_stamped_poses_from_csv_file(
       args.poses_W_E_csv_file, use_JPL_quaternion)
  print("Found ", time_stamped_poses_W_E.shape[
      0], " poses in file: ", args.poses_W_E_csv_file)

  print("Computing time offset...")
  filtering_config = FilteringConfig()
  filtering_config.visualize = args.visualize
  # TODO(mfehr): get filtering config from args!
  time_offset = calculate_time_offset(times_B_H, quaternions_B_H, times_W_E,
                                      quaternions_W_E, filtering_config,
                                      filtering_config.visualize)

  print("Final time offset: ", time_offset, "s")

  print("Computing aligned poses...")
  (aligned_poses_B_H, aligned_poses_W_E) = compute_aligned_poses(
      time_stamped_poses_B_H, time_stamped_poses_W_E, time_offset,
      filtering_config.visualize)

  print("Writing aligned poses to CSV files...")
  write_time_stamped_poses_to_csv_file(aligned_poses_B_H,
                                       args.aligned_poses_B_H_csv_file)
  write_time_stamped_poses_to_csv_file(aligned_poses_W_E,
                                       args.aligned_poses_W_E_csv_file)

  if args.time_offset_output_csv_file is not None:
    print("Writing time_offset to %s." % args.time_offset_output_csv_file)
    from hand_eye_calibration.csv_io import write_double_numpy_array_to_csv_file
    write_double_numpy_array_to_csv_file(np.array((time_offset, )),
                                         args.time_offset_output_csv_file)
