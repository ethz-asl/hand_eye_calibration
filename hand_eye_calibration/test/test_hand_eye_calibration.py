import unittest

import numpy as np
import numpy.testing as npt

from hand_eye_calibration.dual_quaternion_hand_eye_calibration import (
    align, draw_poses, make_paths_start_at_origin,
    compute_dual_quaternions_with_offset)
from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.quaternion import Quaternion
import hand_eye_calibration.hand_eye_test_helpers as he_helpers


class HandEyeCalibration(unittest.TestCase):
  # CONFIG
  paths_start_at_origin = True
  enforce_same_non_dual_scalar_sign = True
  enforce_positive_non_dual_scalar_sign = True

  dq_H_E = he_helpers.random_transform_as_dual_quaternion(
      enforce_positive_non_dual_scalar_sign)
  assert dq_H_E.q_rot.w >= -1e-8

  pose_H_E = dq_H_E.to_pose()
  dq_H_E.normalize()

  dq_B_W = he_helpers.random_transform_as_dual_quaternion()
  assert dq_B_W.q_rot.w >= -1e-8

  dq_B_H_vec, dq_W_E_vec = he_helpers.generate_test_paths(
      20, dq_H_E, dq_B_W, paths_start_at_origin)

  def test_hand_eye_calibration(self):

    dq_H_E_estimated = align(self.dq_B_H_vec, self.dq_W_E_vec,
                             self.enforce_same_non_dual_scalar_sign)
    pose_H_E_estimated = dq_H_E_estimated.to_pose()
    dq_H_E_estimated.normalize()

    assert pose_H_E_estimated[6] > 0.0, (
        "The real part of the pose's quaternion should be positive. "
        "The pose is: \n{}\n where the dual quaternion was: "
        "\n{}".format(pose_H_E_estimated, dq_H_E_estimated))

    print("The static input pose was: \n{}".format(self.pose_H_E))
    print("The hand-eye calibration's output pose is: \n{}".format(
          pose_H_E_estimated))

    print("T_H_E ground truth: \n{}".format(self.dq_H_E.to_matrix()))
    print("T_H_E estimated: \n{}".format(dq_H_E_estimated.to_matrix()))

    assert np.allclose(
        self.dq_H_E.dq, dq_H_E_estimated.dq, rtol=1e-3), (
        "input dual quaternion: {}, estimated dual quaternion: {}".format(
            self.dq_H_E, dq_H_E_estimated))

  def test_draw_poses(self):
      # Draw both paths in their Global/World frame.
    poses_B_H = np.array([self.dq_B_H_vec[0].to_pose().T])
    poses_W_E = np.array([self.dq_W_E_vec[0].to_pose().T])
    for i in range(1, len(self.dq_B_H_vec)):
      poses_B_H = np.append(poses_B_H, np.array(
          [self.dq_B_H_vec[i].to_pose().T]), axis=0)
      poses_W_E = np.append(poses_W_E, np.array(
          [self.dq_W_E_vec[i].to_pose().T]), axis=0)
    draw_poses(poses_B_H, poses_W_E)


if __name__ == '__main__':
  unittest.main()
