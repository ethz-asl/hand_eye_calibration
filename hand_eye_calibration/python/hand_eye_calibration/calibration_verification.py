# -*- coding: utf-8 -*-

from hand_eye_calibration.dual_quaternion_hand_eye_calibration import (
    evaluate_alignment, align_paths_at_index, get_aligned_poses)

  from hand_eye_calibration.time_alignment import (compute_aligned_poses)


def evaluate_calibration(time_stamped_poses_B_H, time_stamped_poses_W_E, dq_H_E, time_alignment):

  (aligned_poses_B_H, aligned_poses_W_E) = compute_aligned_poses(
      time_stamped_poses_B_H, time_stamped_poses_W_E, time_offset, args.visualize)

  aligned_dq_B_H = align_paths_at_index(aligned_poses_B_H, 0)
  aligned_dq_W_E = align_paths_at_index(aligned_poses_W_E, 0)

  (poses_B_H, poses_W_H) = get_aligned_poses(aligned_dq_B_H,
                                             aligned_dq_W_E,
                                             dq_H_E)

  (rmse_position,
   rmse_orientation,
   inlier_flags) = evaluate_alignment(poses_B_H, poses_W_H, config)

  return ((rmse_position, rmse_orientation), sum(inlier_flags))
