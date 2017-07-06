# -*- coding: utf-8 -*-

import math

from hand_eye_calibration.dual_quaternion_hand_eye_calibration import (
    evaluate_alignment, align_paths_at_index, get_aligned_poses)
from hand_eye_calibration.time_alignment import compute_aligned_poses
from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.dual_quaternion_hand_eye_calibration import HandEyeConfig


def evaluate_calibration(time_stamped_poses_B_H, time_stamped_poses_W_E, dq_H_E, time_offset, config):
  assert len(time_stamped_poses_B_H) > 0
  assert len(time_stamped_poses_W_E) > 0
  assert isinstance(config, HandEyeConfig)

  (aligned_poses_B_H, aligned_poses_W_E) = compute_aligned_poses(
      time_stamped_poses_B_H, time_stamped_poses_W_E, time_offset)
  assert len(aligned_poses_B_H) == len(aligned_poses_W_E)

  # If we found not matching poses, the evaluation failed.
  if len(aligned_poses_B_H) == 0:
    return ((float('inf'), float('inf')), 0)

  # Convert poses to dual quaterions.
  dual_quat_B_H_vec = [DualQuaternion.from_pose_vector(
      aligned_pose_B_H) for aligned_pose_B_H in aligned_poses_B_H[:, 1:]]
  dual_quat_W_E_vec = [DualQuaternion.from_pose_vector(
      aligned_pose_W_E) for aligned_pose_W_E in aligned_poses_W_E[:, 1:]]

  assert len(dual_quat_B_H_vec) == len(dual_quat_W_E_vec), "len(dual_quat_B_H_vec): {} vs len(dual_quat_W_E_vec): {}".format(
      len(dual_quat_B_H_vec), len(dual_quat_W_E_vec))

  aligned_dq_B_H = align_paths_at_index(dual_quat_B_H_vec, 0)
  aligned_dq_W_E = align_paths_at_index(dual_quat_W_E_vec, 0)

  (poses_B_H, poses_W_H) = get_aligned_poses(aligned_dq_B_H,
                                             aligned_dq_W_E,
                                             dq_H_E)

  (rmse_position,
   rmse_orientation,
   inlier_flags) = evaluate_alignment(poses_B_H, poses_W_H, config)

  return ((rmse_position, rmse_orientation), sum(inlier_flags))
