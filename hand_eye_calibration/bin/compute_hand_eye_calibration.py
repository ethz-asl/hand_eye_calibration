#!/usr/bin/env python

import argparse
import csv

import numpy as np

from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.dual_quaternion_hand_eye_calibration import (
    align, draw_poses, align_paths_at_index)

# CONFIG
paths_start_at_origin = True
enforce_same_non_dual_scalar_sign = True
enforce_positive_non_dual_scalar_sign = True

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Align pairs of poses.')
  parser.add_argument(
      '--aligned_poses_B_H_csv_file', type=str,
      help='The path to the file containing the first poses. (e.g. Hand poses in Body frame)')
  parser.add_argument(
      '--aligned_poses_H_B_csv_file', type=str,
      help='Alternative input file for the first poses. (e.g. Body poses in Ehe frame)')
  parser.add_argument(
      '--aligned_poses_W_E_csv_file', type=str,
      help='The path to the file containing the second poses. (e.g. Eye poses in World frame)')
  parser.add_argument(
      '--aligned_poses_E_W_csv_file', type=str,
      help='Alternative input file for the second poses. (e.g. World poses in Eye frame)')

  parser.add_argument('--visualize', type=bool, help='Visualize the poses.')
  parser.add_argument('--plot_every_nth_pose', type=int,
                      help='Plot only every n-th pose.', default=1)
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

  # Draw both paths in their Global/World frame.
  poses1 = np.array([dual_quat_B_H_vec[0].to_pose().T])
  poses2 = np.array([dual_quat_W_E_vec[0].to_pose().T])
  for i in range(1, len(dual_quat_B_H_vec)):
    poses1 = np.append(poses1, np.array(
        [dual_quat_B_H_vec[i].to_pose().T]), axis=0)
    poses2 = np.append(poses2, np.array(
        [dual_quat_W_E_vec[i].to_pose().T]), axis=0)
  if args.visualize:
    every_nth_element = args.plot_every_nth_pose
    draw_poses(poses1[::every_nth_element], poses2[::every_nth_element], False)
  dq_H_E_estimated = align(
      dual_quat_B_H_vec, dual_quat_W_E_vec, enforce_same_non_dual_scalar_sign)
  dq_H_E_estimated.normalize()
  pose_H_E_estimated = dq_H_E_estimated.to_pose()

  print("The hand-eye calibration's output pose is: \n" "{}".format(
        pose_H_E_estimated))
  print("The norm of the translation is: {}".format(
      np.linalg.norm(pose_H_E_estimated[0:3])))
