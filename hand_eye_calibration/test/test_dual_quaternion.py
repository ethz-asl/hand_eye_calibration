#!/usr/bin/env python
import unittest

import numpy as np
import numpy.testing as npt

from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.quaternion import Quaternion, quaternion_slerp


class DualQuaternionOperations(unittest.TestCase):

  def test_addition(self):
    qr_1 = Quaternion(1., 2., 3., 4.)
    qt_1 = Quaternion(1., 3., 3., 0.)
    dq_1 = DualQuaternion(qr_1, qt_1)

    qr_2 = Quaternion(3., 5., 3., 2.)
    qt_2 = Quaternion(-4., 2., 3., 0.)
    dq_2 = DualQuaternion(qr_2, qt_2)

    dq_expected = np.array([4., 7., 6., 6., -3., 5., 6., 0.]).T

    npt.assert_allclose((dq_1 + dq_2).dq, dq_expected, rtol=1e-6)

  def test_multiplication(self):
    qr_1 = Quaternion(1., 2., 3., 4.)
    qt_1 = Quaternion(1., 3., 3., 0.)
    dq_1 = DualQuaternion(qr_1, qt_1)
    qr_2 = Quaternion(1., 4., 5., 1.)
    qt_2 = Quaternion(-4., 2., 3., 0.)
    dq_2 = DualQuaternion(qr_2, qt_2)
    # TODO(ff): Check this by hand. (And check if we shouldn't always get 0
    # scalars for the rotational quaternion of the dual quaternion).
    dq_expected = DualQuaternion(dq_1.q_rot * dq_2.q_rot,
                                 dq_1.q_rot * dq_2.q_dual + dq_1.q_dual * dq_2.q_rot)
    npt.assert_allclose((dq_1 * dq_2).dq, dq_expected.dq)

  def test_multiplication_with_scalar(self):
    qr = Quaternion(0.5, 0.5, -0.5, 0.5)
    qt = Quaternion(1, 3, 3, 0)
    dq = DualQuaternion(qr, qt)
    dq_expected = np.array([1.25, 1.25, -1.25, 1.25, 2.5, 7.5, 7.5, 0]).T
    # Note, the scaling only applies to the translational part.
    npt.assert_allclose((dq * 2.5).dq, dq_expected)

  def test_division(self):
    qr_1 = Quaternion(1, 2, 3, 4)
    qt_1 = Quaternion(1, 3, 3, 6)
    dq_1 = DualQuaternion(qr_1, qt_1)
    identity_dq = np.array([0., 0., 0., 1., 0., 0., 0., 0.]).T
    npt.assert_allclose((dq_1 / dq_1).dq, identity_dq, atol=1e-6)
    # TODO(ff): Fix this test.
    qr_2 = Quaternion(1, 4, 5, 1)
    qt_2 = Quaternion(-4, 2, 3, 4)
    dq_2 = DualQuaternion(qr_2, qt_2)
    # dq_2_copy = dq_2.copy()
    # dq_2_copy.normalize()
    # dq_expected = dq_1 / dq_2.norm()[0] * dq_2.conjugate()
    # dq_expected = 1.0 / dq_2.norm()[0] * DualQuaternion(
    #     qr_1 * qr_2.conjugate(),
    #     -1.0 / dq_2.norm()[0] * (qr_1 * qt_2.conjugate() +
    #                              qt_1 * qr_2.conjugate()))
    # dq_expected = dq_1 * DualQuaternion(qr_2.inverse(),
    #                                     -qt_2 * qr_2.inverse() * qr_2.inverse())
    # npt.assert_allclose((dq_1 / dq_2).dq, dq_expected.dq, atol=1e-6)

  def test_division_with_scalar(self):
    qr = Quaternion(0.5, 0.5, -0.5, 0.5)
    qt = Quaternion(1., 3., 3., 0.)
    dq = DualQuaternion(qr, qt)
    dq_expected = np.array([0.25, 0.25, -0.25, 0.25, 0.5, 1.5, 1.5, 0.]).T
    npt.assert_allclose((dq / 2.).dq, dq_expected)

  def test_conjugate(self):
    qr = Quaternion(1., 2., 3., 4.)
    qt = Quaternion(1., 3., 3., 5.)
    dq = DualQuaternion(qr, qt)
    dq_expected = np.array([-1., -2., -3., 4., -1., -3., -3., 5.]).T
    npt.assert_allclose((dq.conjugate()).dq, dq_expected, atol=1e-6)

  def test_conjugate_identity(self):
    qr = Quaternion(1, 2, 3, 4)
    qt = Quaternion(1, 3, 3, 0)
    dq = DualQuaternion(qr, qt)
    # TODO(ff): Here we should use dq*dq.conjugate() in one
    # place.
    q_rot_identity = dq.q_rot * dq.q_rot.conjugate()
    q_dual_identity = (dq.q_rot.conjugate() * dq.q_dual +
                       dq.q_dual.conjugate() * dq.q_rot)
    identity_dq_expected = DualQuaternion(q_rot_identity, q_dual_identity)
    npt.assert_allclose((dq * dq.conjugate()).dq,
                        identity_dq_expected.dq, atol=1e-6)

  def test_normalize(self):
    qr = Quaternion(1, 2, 3, 4)
    qt = Quaternion(1, 3, 3, 0)
    dq = DualQuaternion(qr, qt)
    dq.normalize()
    dq_normalized = np.array([
        0.18257419, 0.36514837, 0.54772256, 0.73029674, 0.18257419, 0.54772256, 0.54772256, 0.
    ]).T
    npt.assert_allclose(dq.dq, dq_normalized)
    dq_2 = DualQuaternion.from_pose(1, 2, 3, 1, 1, 1, 1)
    dq_2.normalize()
    dq_2_normalized = np.array([0.5, 0.5, 0.5, 0.5, 0., 1., 0.5, -1.5]).T
    npt.assert_allclose(dq_2.dq, dq_2_normalized)

  def test_scalar(self):
    qr = Quaternion(1, 2, 3, 4)
    qt = Quaternion(1, 3, 3, 1)
    dq = DualQuaternion(qr, qt)
    scalar = dq.scalar()
    scalar_expected = np.array([0., 0., 0., 4., 0., 0., 0., 1.]).T
    npt.assert_allclose(scalar.dq, scalar_expected, atol=1e-6)

  def test_inverse(self):
    qr = Quaternion(1, 2, 3, 4)
    qt = Quaternion(5, 6, 7, 8)
    dq = DualQuaternion(qr, qt)
    identity_dq = np.array([0., 0., 0., 1., 0., 0., 0., 0.]).T
    npt.assert_allclose((dq * dq.inverse()).dq, identity_dq, atol=1e-6)

  def test_equality(self):
    qr = Quaternion(1, 2, 3, 4)
    qt = Quaternion(1, 3, 3, 1)
    dq_1 = DualQuaternion(qr, qt)
    dq_2 = DualQuaternion(qr, qt)
    self.assertEqual(dq_1, dq_2)

  def test_conversions(self):
    pose = [1, 2, 3, 1., 0., 0., 0.]
    dq = DualQuaternion.from_pose_vector(pose)
    pose_out = dq.to_pose()
    matrix_out = dq.to_matrix()
    matrix_expected = np.array(
        [[1, 0, 0, 1], [0, -1, 0, 2], [0, 0, -1, 3], [0, 0, 0, 1]])
    npt.assert_allclose(pose, pose_out)
    npt.assert_allclose(matrix_out, matrix_expected)

  def test_consecutive_transformations(self):
    dq_1_2 = DualQuaternion.from_pose(0, 10, 1, 1, 0, 0, 0)
    dq_2_3 = DualQuaternion.from_pose(2, 1, 3, 0, 1, 0, 0)
    dq_1_3 = DualQuaternion.from_pose(2, 9, -2, 0, 0, 1, 0)
    # Move coordinate frame dq_1 to dq_3
    dq_1_3_computed = dq_1_2 * dq_2_3
    npt.assert_allclose(dq_1_3_computed.dq, dq_1_3.dq)

  def test_transforming_points(self):
    dq_1_2 = DualQuaternion.from_pose(0, 10, 1, 1, 0, 0, 0)
    dq_2_3 = DualQuaternion.from_pose(2, 1, 3, 0, 1, 0, 0)
    dq_1_3 = DualQuaternion.from_pose(2, 9, -2, 0, 0, 1, 0)
    dq_1_3_computed = dq_1_2 * dq_2_3
    # Express point p (expressed in frame 1) in coordinate frame 3.
    p_1 = np.array([1, 2, 3])
    p_3_direct = dq_1_3.inverse().passive_transform_point(p_1)
    p_3_consecutive = dq_1_3_computed.inverse().passive_transform_point(p_1)
    npt.assert_allclose(p_3_direct, p_3_consecutive)

  def test_dq_to_matrix(self):

    pose = [1, 2, 3, 4., 5., 6., 7.]
    dq = DualQuaternion.from_pose_vector(pose)
    dq.normalize()

    matrix_out = dq.to_matrix()

    dq_from_matrix = DualQuaternion.from_transformation_matrix(matrix_out)
    matrix_out_2 = dq_from_matrix.to_matrix()

    npt.assert_allclose(dq.dq, dq_from_matrix.dq)
    npt.assert_allclose(matrix_out, matrix_out_2)


# TODO(ntonci): Add test for screw_axis method

if __name__ == '__main__':
  unittest.main()
