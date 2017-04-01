#!/usr/bin/env python
import unittest

import numpy as np
import numpy.testing as npt

from hand_eye_calibration.quaternion import (Quaternion, quaternion_lerp,
                                             angle_between_quaternions,
                                             quaternion_nlerp, quaternion_slerp)


class QuaternionOperations(unittest.TestCase):

  def test_addition(self):
    q_1 = Quaternion(1, 2, 3, 4)
    q_2 = Quaternion(2, 3, 4, 5)
    q_expected = np.array([3, 5, 7, 9]).T
    npt.assert_allclose((q_1 + q_2).q, q_expected)

  def test_multiplication(self):
    q_1 = Quaternion(0.45349524, -0.5514413, -0.16603085, -0.68021197)
    q_2 = Quaternion(0.4810662, -0.42344938, 0.74744865, 0.17488983)
    q_expected = np.array([-0.730395, -0.22724237, -0.46421313, -0.44653133]).T
    npt.assert_allclose((q_1 * q_2).q, q_expected)

  def test_multiplication_with_scalar(self):
    q = Quaternion(1, 2, 3, 4)
    q_expected = np.array([2.5, 5, 7.5, 10]).T
    npt.assert_allclose((q * 2.5).q, q_expected)

  def test_division(self):
    q_1 = Quaternion(0.45349524, -0.5514413, -0.16603085, -0.68021197)
    q_2 = Quaternion(0.4810662, -0.42344938, 0.74744865, 0.17488983)
    q_expected = np.array([0.88901841, 0.03435942, 0.40613892, 0.20860702]).T
    npt.assert_allclose((q_1 / q_2).q, q_expected)

  def test_division_with_scalar(self):
    q = Quaternion(1, 2, 3, 4)
    q_expected = np.array([0.5, 1.0, 1.5, 2.0]).T
    npt.assert_allclose((q / 2).q, q_expected)

  def test_conjugate(self):
    q = Quaternion(1, 2, 3, 4)
    q_conjugate = np.array([-1, -2, -3, 4]).T
    npt.assert_allclose((q.conjugate()).q, q_conjugate)

  def test_norm(self):
    q = Quaternion(1, 2, 3, 4)
    q_norm = 5.477225575
    npt.assert_almost_equal(q.norm(), q_norm)

  def test_inverse(self):
    q = Quaternion(1, 2, 3, 4)
    q_identity = np.array([0, 0, 0, 1]).T
    npt.assert_allclose((q * q.inverse()).q, q_identity)

  def test_normalize(self):
    q = Quaternion(1, 2, 3, 4)
    q.normalize()
    q_normalized = np.array([0.18257419, 0.36514837, 0.54772256, 0.73029674]).T
    npt.assert_allclose(q.q, q_normalized)

  def test_slerp(self):
    q_1 = Quaternion(1, 2, 3, 4)
    q_2 = Quaternion(2, 3, 4, 5)
    q_1.normalize()
    q_2.normalize()
    q_expected = np.array(
        [0.22772264489951, 0.38729833462074169, 0.54687402434197, 0.70644971406320634]).T
    npt.assert_allclose(quaternion_slerp(q_1, q_2, 0.5).q, q_expected)

  def test_lerp(self):
    q_1 = Quaternion(1, 2, 3, 4)
    q_2 = Quaternion(2, 3, 4, 5)
    q_1.normalize()
    q_2.normalize()
    q_expected = np.array([0.22736986, 0.38669833, 0.54602681, 0.70535528]).T
    npt.assert_allclose(quaternion_lerp(q_1, q_2, 0.5).q, q_expected, atol=1e-6)

  def test_nlerp(self):
    q_1 = Quaternion(1, 2, 3, 4)
    q_2 = Quaternion(2, 3, 4, 5)
    q_1.normalize()
    q_2.normalize()
    q_expected = np.array([0.22772264, 0.38729833, 0.54687402, 0.70644971]).T
    npt.assert_allclose(quaternion_nlerp(
        q_1, q_2, 0.5).q, q_expected, atol=1e-6)

  def test_quaternion_from_rotation_matrix(self):
    rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    q = Quaternion.from_rotation_matrix(rotation_matrix)
    expected_quaternion = Quaternion(-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2)
    npt.assert_allclose(q.q, expected_quaternion.q)

  def test_quaternion_to_rotation_matrix(self):
    q = Quaternion(0.5, 0.5, 0.5, 0.5)
    rotation_matrix = q.to_rotation_matrix()
    expected_rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    npt.assert_allclose(rotation_matrix, expected_rotation_matrix)

  def test_quaternion_to_transformation_matrix(self):
    q = Quaternion(0.5, 0.5, 0.5, 0.5)
    transformation_matrix = q.to_transformation_matrix()
    expected_transformation_matrix = np.array(
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    npt.assert_allclose(transformation_matrix, expected_transformation_matrix)

  def test_angular_velocity_between_quaternions(self):
    # TODO(ff): Add test.
    pass

  def test_quaternions_interpolate(self):
    # TODO(ff): Add test.
    pass

  def test_angle_bewtween_quaternions(self):
    q_1 = Quaternion(np.sqrt(2.) / 2., 0, 0, np.sqrt(2.) / 2.)
    q_2 = Quaternion(0, 0, 0, 1)
    angle = angle_between_quaternions(q_1, q_2)
    npt.assert_almost_equal(angle, np.pi / 2)


if __name__ == '__main__':
  unittest.main()
