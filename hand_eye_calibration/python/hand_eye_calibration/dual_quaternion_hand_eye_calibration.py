# -*- coding: utf-8 -*-

from itertools import compress
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import copy
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import tf
import timeit

from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.quaternion import (
    Quaternion, angle_between_quaternions)
from hand_eye_calibration.hand_eye_calibration_plotting_tools import (
    plot_alignment_errors, plot_poses)

# This implements the following paper.
#
# @article{doi:10.1177/02783649922066213,
# author = {Konstantinos Daniilidis},
# title = {Hand-Eye Calibration Using Dual Quaternions},
# journal = {The International Journal of Robotics Research},
# volume = {18},
# number = {3},
# pages = {286-298},
# year = {1999},
# doi = {10.1177/02783649922066213},
# URL = {http://dx.doi.org/10.1177/02783649922066213},
# eprint = {http://dx.doi.org/10.1177/02783649922066213},
# }

# All Quaternions are Hamiltonian Quaternions.
# Denoted as: q = [x, y, z, w]

# Notations:
# Frames are:
# H: Hand frame
# B: World (Base) frame of hand
# E: Eye frame
# W: World frame of eye
#
# T_B_W: Denotes the transformation from a point in the World frame to the
# base frame.


class HandEyeConfig:

  def __init__(self):

    # General config.
    self.algorithm_name = ""
    self.use_baseline_approach = False
    self.min_num_inliers = 10
    self.enable_exhaustive_search = False

    # Select distinctive poses based on skrew axis
    self.prefilter_poses_enabled = True
    self.prefilter_dot_product_threshold = 0.975

    # RANSAC
    self.ransac_sample_size = 3
    self.ransac_sample_rejection_scalar_part_equality_tolerance = 1e-2
    self.ransac_max_number_iterations = 20
    self.ransac_enable_early_abort = True
    self.ransac_outlier_probability = 0.5
    self.ransac_success_probability_threshold = 0.99
    self.ransac_inlier_classification = "scalar_part_equality"
    self.ransac_position_error_threshold_m = 0.02
    self.ransac_orientation_error_threshold_deg = 1.0

    # Model refinement
    self.ransac_model_refinement = True
    self.ransac_evaluate_refined_model_on_inliers_only = False

    # Hand-calibration
    self.hand_eye_calibration_scalar_part_equality_tolerance = 4e-2

    # Visualization
    self.visualize = False
    self.visualize_plot_every_nth_pose = 10


def compute_dual_quaternions_with_offset(dq_B_H_vec, dq_H_E, dq_B_W):
  n_samples = len(dq_B_H_vec)
  dq_W_E_vec = []

  dq_W_B = dq_B_W.inverse()
  for i in range(0, n_samples):
    dq_B_H = dq_B_H_vec[i]

    dq_W_E = dq_W_B * dq_B_H * dq_H_E

    dq_W_E.normalize()
    assert np.isclose(dq_W_E.norm()[0], 1.0, atol=1.e-8), dq_W_E
    dq_W_E_vec.append(dq_W_E)
  return dq_W_E_vec


def align_paths_at_index(dq_vec, align_index=0, enforce_positive_q_rot_w=True):
  dq_align_inverse = dq_vec[align_index].inverse().copy()
  n_samples = len(dq_vec)
  dq_vec_starting_at_origin = [None] * n_samples
  for i in range(0, n_samples):
    dq_vec_starting_at_origin[i] = dq_align_inverse * dq_vec[i].copy()
    if (enforce_positive_q_rot_w):
      if dq_vec_starting_at_origin[i].q_rot.w < 0.:
        dq_vec_starting_at_origin[i].dq = -(
            dq_vec_starting_at_origin[i].dq.copy())

  # Rearange poses such that it starts at the origin.
  dq_vec_rearanged = dq_vec_starting_at_origin[align_index:] + \
      dq_vec_starting_at_origin[:align_index]

  assert np.allclose(dq_vec_rearanged[0].dq,
                     [0., 0., 0., 1.0, 0., 0., 0., 0.],
                     atol=1.e-8), dq_vec_rearanged[0]

  return dq_vec_rearanged


def skew_from_vector(vector):
  skew = np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]],
                   [-vector[1], vector[0], 0]])
  return skew.copy()


def setup_s_matrix(dq_1, dq_2):
  """This sets up the [6x8] S matrix, see Eq. (31) of the referenced paper.

  S = (skew(I(qr1)+I(qr2)) I(qr1)-I(qr2) 0_{3x3}             0_{3x1}      )
      (skew(I(qt1)+I(qt2)) I(qt1)-I(qt2) skew(I(qr1)+I(qr2)) I(qr1)-I(qr2))
  I(q) denotes the vector of the imaginary components of a quaternion.
  Note: The order of the blocks switched as we are using q = [x y z w]^T
  instead of q = [w x y z].T.
  """
  scalar_parts_1 = dq_1.scalar()
  scalar_parts_2 = dq_2.scalar()

  assert np.allclose(
      scalar_parts_1.dq, scalar_parts_2.dq,
      atol=5e-2), (
      "\ndq1:\n{},\nscalar_parts_1:\n{},\ndq2:\n{},\nscalar_parts_2:\n{}\n"
      "Scalar parts should always be equal.".format(dq_1, scalar_parts_1, dq_2,
                                                    scalar_parts_2))

  s_matrix = np.zeros([6, 8])
  s_matrix[0:3, 0:3] = skew_from_vector(dq_1.q_rot.q[0:-1] + dq_2.q_rot.q[0:-1])
  s_matrix[0:3, 3] = dq_1.q_rot.q[0:-1] - dq_2.q_rot.q[0:-1]
  s_matrix[3:6, 0:3] = skew_from_vector(dq_1.q_dual.q[0:-1] +
                                        dq_2.q_dual.q[0:-1])
  s_matrix[3:6, 3] = dq_1.q_dual.q[0:-1] - dq_2.q_dual.q[0:-1]
  s_matrix[3:6, 4:7] = skew_from_vector(dq_1.q_rot.q[0:-1] + dq_2.q_rot.q[0:-1])
  s_matrix[3:6, 7] = dq_1.q_rot.q[0:-1] - dq_2.q_rot.q[0:-1]
  # print("S: \n{}".format(s_matrix))

  rank_s_matrix = np.linalg.matrix_rank(s_matrix)
  assert rank_s_matrix <= 6, s_matrix
  return s_matrix.copy()


def setup_t_matrix(dq_W_E_vec, dq_B_H_vec):
  """This sets up the [6nx8] T matrix consisting of multiple S matrices for the
  different pose pairs. See Equation (33) of the referenced paper.

  T = (S_1.T S_2.T ... S_n.T).T
  """
  n_quaternions = len(dq_W_E_vec)
  t_matrix = np.zeros([6 * n_quaternions, 8])
  for i in range(n_quaternions):
    t_matrix[i * 6:i * 6 + 6, :] = setup_s_matrix(dq_W_E_vec[i], dq_B_H_vec[i])

  rank_t_matrix = np.linalg.matrix_rank(t_matrix, tol=5e-2)
  U, s, V = np.linalg.svd(t_matrix)
  # print("t_matrix: \n{}".format(t_matrix))
  # print("Rank(t_matrix): {}".format(rank_t_matrix))
  # assert rank_t_matrix == 6, ("T should have rank 6 otherwise we can not find "
  #                             "a rigid transform.", rank_t_matrix, s)
  return t_matrix.copy()


def compute_hand_eye_calibration(dq_B_H_vec_inliers, dq_W_E_vec_inliers,
                                 scalar_part_tolerance=1e-2,
                                 enforce_same_non_dual_scalar_sign=True):
  """
  Do the actual hand eye-calibration as described in the referenced paper.
  Assumes the outliers have already been removed and the scalar parts of
  each pair are a match.
  """
  n_quaternions = len(dq_B_H_vec_inliers)

  # Verify that the first pose is at the origin.
  assert np.allclose(dq_B_H_vec_inliers[0].dq,
                     [0., 0., 0., 1.0, 0., 0., 0., 0.],
                     atol=1.e-8), dq_B_H_vec_inliers[0]
  assert np.allclose(dq_W_E_vec_inliers[0].dq,
                     [0., 0., 0., 1.0, 0., 0., 0., 0.],
                     atol=1.e-8), dq_W_E_vec_inliers[0]

  if enforce_same_non_dual_scalar_sign:
    for i in range(n_quaternions):
      dq_W_E = dq_W_E_vec_inliers[i]
      dq_B_H = dq_B_H_vec_inliers[i]
      if ((dq_W_E.q_rot.w < 0. and dq_B_H.q_rot.w > 0.) or
              (dq_W_E.q_rot.w > 0. and dq_B_H.q_rot.w < 0.)):
        dq_W_E_vec_inliers[i].dq = -dq_W_E_vec_inliers[i].dq.copy()

  # 0. Stop alignment if there are still pairs that do not have matching
  # scalar parts.
  for j in range(n_quaternions):
    dq_B_H = dq_W_E_vec_inliers[j]
    dq_W_E = dq_B_H_vec_inliers[j]

    scalar_parts_B_H = dq_B_H.scalar()
    scalar_parts_W_E = dq_W_E.scalar()

    assert np.allclose(scalar_parts_B_H.dq, scalar_parts_W_E.dq,
                       atol=scalar_part_tolerance), (
        "Mismatch of scalar parts of dual quaternion at idx {}:"
        " dq_B_H: {} dq_W_E: {}".format(j, dq_B_H, dq_W_E))

  # 1.
  # Construct 6n x 8 matrix T
  t_matrix = setup_t_matrix(dq_B_H_vec_inliers, dq_W_E_vec_inliers)

  # 2.
  # Compute SVD of T and check if only two singular values are almost equal to
  # zero. Take the corresponding right-singular vectors (v_7 and v_8)
  U, s, V = np.linalg.svd(t_matrix)

  # Check if only the last two singular values are almost zero.
  bad_singular_values = False
  for i, singular_value in enumerate(s):
    if i < 6:
      if singular_value < 5e-1:
        bad_singular_values = True
    else:
      if singular_value > 5e-1:
        bad_singular_values = True
  v_7 = V[6, :].copy()
  v_8 = V[7, :].copy()
  # print("v_7: {}".format(v_7))
  # print("v_8: {}".format(v_8))

  # 3.
  # Compute the coefficients of (35) and solve it, finding two solutions for s.
  u_1 = v_7[0:4].copy()
  u_2 = v_8[0:4].copy()
  v_1 = v_7[4:8].copy()
  v_2 = v_8[4:8].copy()
  # print("u_1: {}, \nu_2: {}, \nv_1: {}, \nv_2: {}".format(u_1, u_2, v_1, v_2))

  a = np.dot(u_1.T, v_1)
  assert a != 0.0, "This would involve division by zero."
  b = np.dot(u_1.T, v_2) + np.dot(u_2.T, v_1)
  c = np.dot(u_2.T, v_2)
  # print("a: {}, b: {}, c: {}".format(a, b, c))
  square_root_term = b * b - 4.0 * a * c

  if square_root_term < -1e-2:
    assert False, "square_root_term is too negative: {}".format(
        square_root_term)
  if square_root_term < 0.0:
    square_root_term = 0.0
  s_1 = (-b + np.sqrt(square_root_term)) / (2.0 * a)
  s_2 = (-b - np.sqrt(square_root_term)) / (2.0 * a)
  # print("s_1: {}, s_2: {}".format(s_1, s_2))

  # 4.
  # For these two s values, compute s^2*u_1^T*u_1 + 2*s*u_1^T*u_2 + u_2^T*u_2
  # From these choose the largest to compute lambda_2 and then lambda_1
  solution_1 = s_1 * s_1 * np.dot(u_1.T, u_1) + 2.0 * \
      s_1 * np.dot(u_1.T, u_2) + np.dot(u_2.T, u_2)
  solution_2 = s_2 * s_2 * np.dot(u_1.T, u_1) + 2.0 * \
      s_2 * np.dot(u_1.T, u_2) + np.dot(u_2.T, u_2)

  if solution_1 > solution_2:
    assert solution_1 > 0.0, solution_1
    lambda_2 = np.sqrt(1.0 / solution_1)
    lambda_1 = s_1 * lambda_2
  else:
    assert solution_2 > 0.0, solution_2
    lambda_2 = np.sqrt(1.0 / solution_2)
    lambda_1 = s_2 * lambda_2
  # print("lambda_1: {}, lambda_2: {}".format(lambda_1, lambda_2))

  # 5.
  # The result is lambda_1*v_7 + lambda_2*v_8
  dq_H_E = DualQuaternion.from_vector(lambda_1 * v_7 + lambda_2 * v_8)
  # Normalize the output, to get rid of numerical errors.
  dq_H_E.normalize()

  if (dq_H_E.q_rot.w < 0.):
    dq_H_E.dq = -dq_H_E.dq.copy()
  return (dq_H_E, s, bad_singular_values)


def prefilter_using_screw_axis(dq_W_E_vec_in, dq_B_H_vec_in, dot_product_threshold=0.95):
  dq_W_E_vec = copy.deepcopy(dq_W_E_vec_in)
  dq_B_H_vec = copy.deepcopy(dq_B_H_vec_in)
  n_quaternions = len(dq_W_E_vec)
  i = 0
  while i < len(dq_W_E_vec):
    dq_W_E_i = dq_W_E_vec[i]
    dq_B_H_i = dq_B_H_vec[i]
    screw_axis_W_E_i, rotation_W_E_i, translation_W_E_i = dq_W_E_i.screw_axis()
    screw_axis_B_H_i, rotation_B_H_i, translation_B_H_i = dq_B_H_i.screw_axis()

    if (np.linalg.norm(screw_axis_W_E_i) <= 1.e-12 or np.linalg.norm(screw_axis_B_H_i) <= 1.e-12):
      dq_W_E_vec.pop(i)
      dq_B_H_vec.pop(i)
    else:
      screw_axis_W_E_i = screw_axis_W_E_i / np.linalg.norm(screw_axis_W_E_i)
      screw_axis_B_H_i = screw_axis_B_H_i / np.linalg.norm(screw_axis_B_H_i)

      # TODO(ntonci): Add a check for small motion

      j = i + 1
      while j < len(dq_W_E_vec):
        dq_W_E_j = dq_W_E_vec[j]
        dq_B_H_j = dq_B_H_vec[j]
        screw_axis_W_E_j, rotation_W_E_j, translation_W_E_j = dq_W_E_j.screw_axis()
        screw_axis_B_H_j, rotation_B_H_j, translation_B_H_j = dq_B_H_j.screw_axis()

        if (np.linalg.norm(screw_axis_W_E_j) <= 1.e-12 or np.linalg.norm(screw_axis_B_H_j) <= 1.e-12):
          dq_W_E_vec.pop(j)
          dq_B_H_vec.pop(j)
        else:
          screw_axis_W_E_j = screw_axis_W_E_j / np.linalg.norm(screw_axis_W_E_j)
          screw_axis_B_H_j = screw_axis_B_H_j / np.linalg.norm(screw_axis_B_H_j)

          if (np.inner(screw_axis_W_E_i, screw_axis_W_E_j) > dot_product_threshold):
            dq_W_E_vec.pop(j)
            dq_B_H_vec.pop(j)
          elif (np.inner(screw_axis_B_H_i, screw_axis_B_H_j) > dot_product_threshold):
            dq_W_E_vec.pop(j)
            dq_B_H_vec.pop(j)
          else:
            j += 1
      i += 1

  assert i >= 2, "Not enough distinct poses found."
  return dq_W_E_vec, dq_B_H_vec


def compute_pose_error(pose_A, pose_B):
  """
  Compute the error norm of position and orientation.
  """
  error_position = np.linalg.norm(pose_A[0:3] - pose_B[0:3], ord=2)

  # Construct quaternions to compare.
  quaternion_A = Quaternion(q=pose_A[3:7])
  quaternion_A.normalize()
  if quaternion_A.w < 0:
    quaternion_A.q = -quaternion_A.q
  quaternion_B = Quaternion(q=pose_B[3:7])
  quaternion_B.normalize()
  if quaternion_B.w < 0:
    quaternion_B.q = -quaternion_B.q

  # Sum up the square of the orientation angle error.
  error_angle_rad = angle_between_quaternions(
      quaternion_A, quaternion_B)
  error_angle_degrees = math.degrees(error_angle_rad)
  if error_angle_degrees > 180.0:
    error_angle_degrees = math.fabs(360.0 - error_angle_degrees)

  return (error_position, error_angle_degrees)


def evaluate_alignment(poses_A, poses_B, config, visualize=False):
  """
  Takes aligned poses and compares position and orientation.
  Returns the RMSE of position and orientation as well as a bool vector,
  indicating which pairs are below the error thresholds specified in the
  configuration:
    ransac_orientation_error_threshold_deg
    ransac_position_error_threshold_m
  """

  assert np.array_equal(poses_A.shape, poses_B.shape), (
      "Two pose vector of different size cannot be evaluated. "
      "Size pose A: {} Size pose B: {}".format(poses_A.shape, poses_B.shape))
  assert poses_A.shape[1] == 7, "poses_A are not valid poses!"
  assert poses_B.shape[1] == 7, "poses_B are not valid poses!"
  assert isinstance(config, HandEyeConfig)

  num_poses = poses_A.shape[0]

  inlier_list = [False] * num_poses

  errors_position = np.zeros((num_poses, 1))
  errors_orientation = np.zeros((num_poses, 1))
  for i in range(0, num_poses):
    (error_position,
     error_angle_degrees) = compute_pose_error(poses_A[i, :], poses_B[i, :])

    if (error_angle_degrees < config.ransac_orientation_error_threshold_deg and
            error_position < config.ransac_position_error_threshold_m):
      inlier_list[i] = True

    errors_position[i] = error_position
    errors_orientation[i] = error_angle_degrees

  rmse_pose_accumulator = np.sum(np.square(errors_position))
  rmse_orientation_accumulator = np.sum(np.square(errors_orientation))

  # Compute RMSE.
  rmse_pose = math.sqrt(rmse_pose_accumulator / num_poses)
  rmse_orientation = math.sqrt(rmse_orientation_accumulator / num_poses)

  # Plot the error.
  if visualize:
    plot_alignment_errors(errors_position, rmse_pose, errors_orientation,
                          rmse_orientation, blocking=True)

  return (rmse_pose, rmse_orientation, inlier_list)


def get_aligned_poses(dq_B_H_vec, dq_W_E_vec, dq_H_E_estimated):
  """

  """

  assert len(dq_W_E_vec) == len(dq_B_H_vec)

  # Compute aligned poses.
  dq_E_H_estimated = dq_H_E_estimated.inverse()
  dq_E_H_estimated.normalize()
  dq_E_H_estimated.enforce_positive_q_rot_w()

  dq_W_H_vec = []
  for i in range(0, len(dq_B_H_vec)):
    dq_W_H = dq_W_E_vec[i] * dq_E_H_estimated
    dq_W_H.normalize()

    if ((dq_W_H.q_rot.w < 0. and dq_B_H_vec[i].q_rot.w > 0.) or
            (dq_W_H.q_rot.w > 0. and dq_B_H_vec[i].q_rot.w < 0.)):
      dq_W_H.dq = -dq_W_H.dq.copy()

    dq_W_H_vec.append(dq_W_H)

  dq_W_H_vec = align_paths_at_index(dq_W_H_vec)

  # Convert to poses.
  poses_W_H = np.array([dq_W_H_vec[0].to_pose().T])
  for i in range(1, len(dq_W_H_vec)):
    poses_W_H = np.append(poses_W_H, np.array(
        [dq_W_H_vec[i].to_pose().T]), axis=0)
  poses_B_H = np.array([dq_B_H_vec[0].to_pose().T])
  for i in range(1, len(dq_B_H_vec)):
    poses_B_H = np.append(poses_B_H, np.array(
        [dq_B_H_vec[i].to_pose().T]), axis=0)

  return (poses_B_H.copy(), poses_W_H.copy())


def compute_hand_eye_calibration_BASELINE(dq_B_H_vec, dq_W_E_vec, config):
  """
  Do the actual hand eye-calibration as described in the referenced paper.

  Outputs a tuple containing:
   - success
   - dq_H_E
   - (best_rmse_position, best_rmse_orientation)
   - best_num_inliers
   - num_poses_after_filtering
   - runtime
   - singular_values
   - bad_singular_values
  """
  assert len(dq_W_E_vec) == len(dq_B_H_vec)
  num_poses = len(dq_W_E_vec)

  start_time = timeit.default_timer()

  # Enforce the same sign of the rotation quaternion.
  for i in range(num_poses):
    dq_B_H = dq_B_H_vec[i]
    dq_W_E = dq_W_E_vec[i]
    if ((dq_W_E.q_rot.w < 0. and dq_B_H.q_rot.w > 0.) or
            (dq_W_E.q_rot.w > 0. and dq_B_H.q_rot.w < 0.)):
      dq_W_E_vec[i].dq = -dq_W_E_vec[i].dq.copy()

  # 0.0 Reject pairs whose motion is not informative,
  # i.e. their screw axis dot product is large
  if config.prefilter_poses_enabled:
    dq_B_H_vec_filtered, dq_W_E_vec_filtered = prefilter_using_screw_axis(
        dq_B_H_vec, dq_W_E_vec, config.prefilter_dot_product_threshold)
  else:
    dq_B_H_vec_filtered = dq_B_H_vec
    dq_W_E_vec_filtered = dq_W_E_vec
  num_poses_after_filtering = len(dq_W_E_vec_filtered)

  best_idx = -1
  best_num_inliers = config.min_num_inliers - 1
  best_dq_W_E_vec_inlier = []
  best_dq_B_H_vec_inlier = []

  if config.enable_exhaustive_search:
    print("Do exhaustive search to find biggest subset of inliers...")
  else:
    print("Search for first set of inliers bigger than {}...".format(
        config.min_num_inliers))

  # 0.1 Reject pairs where scalar parts of dual quaternions do not match.
  # Loop over all the indices to find an index of a pose pair.
  for j in range(num_poses_after_filtering):
    # Re-align all dual quaternion to the j-th dual quaternion.
    dq_W_E_vec_aligned = align_paths_at_index(dq_W_E_vec_filtered, j)
    dq_B_H_vec_aligned = align_paths_at_index(dq_B_H_vec_filtered, j)

    dq_W_E_vec_inlier = []
    dq_B_H_vec_inlier = []

    # Loop over the indices again starting at the first index to find either:
    # - The first set of inliers of at least size min_num_inliers
    #       OR
    # - The largest set of inliers using an exhaustive search
    for i in range(0, num_poses_after_filtering):
      dq_W_E = dq_W_E_vec_aligned[i]
      dq_B_H = dq_B_H_vec_aligned[i]
      scalar_parts_W_E = dq_W_E.scalar()
      scalar_parts_B_H = dq_B_H.scalar()
      # Append the inliers to the filtered dual quaternion vectors.
      if np.allclose(scalar_parts_W_E.dq, scalar_parts_B_H.dq, atol=1e-2):
        dq_W_E_vec_inlier.append(dq_W_E)
        dq_B_H_vec_inlier.append(dq_B_H)

    assert len(dq_W_E_vec_inlier) == len(dq_B_H_vec_inlier)

    if config.enable_exhaustive_search:
      has_the_most_inliers = (len(dq_W_E_vec_inlier) > best_num_inliers)
      if has_the_most_inliers:
        best_num_inliers = len(dq_W_E_vec_inlier)
        best_idx = j
        best_dq_W_E_vec_inlier = copy.deepcopy(dq_W_E_vec_inlier)
        best_dq_B_H_vec_inlier = copy.deepcopy(dq_B_H_vec_inlier)
        print("Found new best start idx: {} number of inliers: {}".format(
            best_idx, best_num_inliers))
    else:
      has_enough_inliers = (len(dq_W_E_vec_inlier) > config.min_num_inliers)
      if has_enough_inliers:
        best_idx = j
        best_num_inliers = len(dq_W_E_vec_inlier)
        break

      assert (j + 1) < num_poses_after_filtering, (
          "Reached over all filtered poses and couldn't find "
          "enough inliers. num_samples: {}, num_inliers: {}".format(
              num_poses_after_filtering, len(dq_W_E_vec_inlier)))

  if config.enable_exhaustive_search:
    assert best_idx != -1, "Not enough inliers found!"
    dq_W_E_vec_inlier = best_dq_W_E_vec_filtered
    dq_B_H_vec_inlier = best_dq_B_H_vec_inlier

  aligned_dq_B_H = align_paths_at_index(dq_B_H_vec_inlier, best_idx)
  aligned_dq_W_E = align_paths_at_index(dq_W_E_vec_inlier, best_idx)

  print("Best start idx: {}".format(best_idx))
  print("Removed {} outliers from the (prefiltered) poses.".format(
      len(dq_B_H_vec_filtered) - len(dq_B_H_vec_inlier)))
  print("Running the hand-eye calibration with the remaining {} pairs of "
        "poses".format(len(dq_B_H_vec_inlier)))

  try:
    # Compute hand-eye calibration on the inliers.
    (dq_H_E_estimated,
     singular_values,
     bad_singular_values) = compute_hand_eye_calibration(
        dq_B_H_vec_inlier, dq_W_E_vec_inlier,
        config.hand_eye_calibration_scalar_part_equality_tolerance)
    dq_H_E_estimated.normalize()
  except:
    print("\n\n Hand-eye calibration FAILED! "
          "algorithm_name: {} exception: \n\n".format(
              config.algorithm_name, sys.exc_info()[0]))
    end_time = timeit.default_timer()
    runtime = end_time - start_time
    return (False, None, (None, None),
            None, num_poses_after_filtering, runtime, None, None)

  # Evaluate hand-eye calibration either on all poses aligned by the
  # sample index or only on the inliers.
  if config.ransac_evaluate_refined_model_on_inliers_only:
    (poses_B_H, poses_W_H) = get_aligned_poses(dq_B_H_vec_inlier,
                                               dq_W_E_vec_inlier,
                                               dq_H_E_estimated)
  else:
    # TODO(mfehr): There is some redundancy here, fix it!
    aligned_dq_B_H = align_paths_at_index(dq_B_H_vec, best_idx)
    aligned_dq_W_E = align_paths_at_index(dq_W_E_vec, best_idx)
    (poses_B_H, poses_W_H) = get_aligned_poses(aligned_dq_B_H,
                                               aligned_dq_W_E,
                                               dq_H_E_estimated)

  (rmse_position,
   rmse_orientation,
   inlier_flags) = evaluate_alignment(poses_B_H, poses_W_H, config, config.visualize)

  end_time = timeit.default_timer()
  runtime = end_time - start_time

  pose_vec = dq_H_E_estimated.to_pose()
  print("Solution found by aligned based on idx: {}\n"
        "\t\tNumber of inliers: {}\n"
        "\t\tRMSE position:     {:10.4f}\n"
        "\t\tRMSE orientation:  {:10.4f}\n"
        "\t\tdq_H_E:    {}\n"
        "\t\tpose_H_E:  {}\n"
        "\t\tTranslation norm:  {:10.4f}".format(
            best_idx, best_num_inliers, rmse_position,
            rmse_orientation, dq_H_E_estimated,
            pose_vec, np.linalg.norm(pose_vec[0:3])))

  return (True, dq_H_E_estimated,
          (rmse_position, rmse_orientation),
          best_num_inliers, num_poses_after_filtering, runtime, singular_values, bad_singular_values)


def compute_hand_eye_calibration_RANSAC(dq_B_H_vec, dq_W_E_vec, config):
  """
  Runs various RANSAC-based hand-eye calibration algorithms.
   - RANSAC using the position/orientation error to determine model inliers
   - RANSAC using the scalar part equality constraint to determine model inliers
   For evaluation purposes:
   - Exhaustive search versions of both algorithms above.

   Outputs a tuple containing:
    - success
    - dq_H_E
    - (best_rmse_position, best_rmse_orientation)
    - best_num_inliers
    - num_poses_after_filtering
    - runtime
    - singular_values
    - bad_singular_values
  """
  assert len(dq_W_E_vec) == len(dq_B_H_vec)

  start_time = timeit.default_timer()

  num_poses = len(dq_W_E_vec)
  assert config.ransac_sample_size < num_poses, (
      "The RANSAC sample size ({}) is bigger than the number "
      "of poses ({})!".format(config.ransac_sample_size, num_poses))

  # Reject pairs whose motion is not informative,
  # i.e. their screw axis dot product is large
  if config.prefilter_poses_enabled:
    dq_B_H_vec_filtered, dq_W_E_vec_filtered = prefilter_using_screw_axis(
        dq_B_H_vec, dq_W_E_vec, config.prefilter_dot_product_threshold)
    assert len(dq_W_E_vec_filtered) == len(dq_B_H_vec_filtered)
    assert len(dq_B_H_vec) == num_poses
    assert len(dq_W_E_vec) == num_poses
    num_poses_after_filtering = len(dq_W_E_vec_filtered)
  else:
    dq_B_H_vec_filtered = dq_B_H_vec
    dq_W_E_vec_filtered = dq_W_E_vec
    num_poses_after_filtering = num_poses

  print("Ignore {} poses based on the screw axis".format(
      num_poses - num_poses_after_filtering))
  print("Drawing samples from remaining {} poses".format(
      num_poses_after_filtering))

  indices_set = set(range(0, num_poses_after_filtering))

  # Result variables:
  best_inlier_idx_set = None
  best_num_inliers = 0
  best_rmse_position = np.inf
  best_rmse_orientation = np.inf
  best_estimated_dq_H_E = None
  best_singular_values = None
  best_singular_value_status = True

  all_sample_combinations = []
  max_number_samples = np.inf
  if not config.enable_exhaustive_search:
    print("Running RANSAC...")
    print("Inlier classification method: {}".format(
        config.ransac_inlier_classification))
  else:
    all_sample_combinations = list(
        itertools.combinations(indices_set, config.ransac_sample_size))
    max_number_samples = len(all_sample_combinations)
    print("Running exhaustive search, exploring {} "
          "sample combinations...".format(max_number_samples))

  sample_number = 0
  full_iterations = 0
  prerejected_samples = 0
  while ((not config.enable_exhaustive_search and
          full_iterations < config.ransac_max_number_iterations) or
         (config.enable_exhaustive_search and
          sample_number < max_number_samples)):

    # Get sample, either at:
    #  - random (RANSAC)
    #  - from the list of all possible samples (exhaustive search)
    if config.enable_exhaustive_search:
      sample_indices = list(all_sample_combinations[sample_number])
    else:
      sample_indices = random.sample(indices_set,
                                     config.ransac_sample_size)
    sample_number += 1

    # Extract sampled poses.
    samples_dq_W_E = [dq_W_E_vec_filtered[idx] for idx in sample_indices]
    samples_dq_B_H = [dq_B_H_vec_filtered[idx] for idx in sample_indices]
    assert len(samples_dq_W_E) == len(samples_dq_B_H)
    assert len(samples_dq_W_E) == config.ransac_sample_size

    if config.ransac_sample_size > 1:
      # Transform all sample poses such that the first pose becomes the origin.
      aligned_samples_dq_B_H = align_paths_at_index(samples_dq_B_H,
                                                    align_index=0)
      aligned_samples_dq_W_E = align_paths_at_index(samples_dq_W_E,
                                                    align_index=0)
      assert len(aligned_samples_dq_B_H) == len(aligned_samples_dq_W_E)

      # Reject the sample early if not even the samples have a
      # similar scalar part. This should speed up RANSAC and is required, as
      # the hand eye calibration does not accept outliers.
      good_sample = True
      for i in range(0, config.ransac_sample_size):
        scalar_parts_W_E = aligned_samples_dq_W_E[i].scalar()
        scalar_parts_B_H = aligned_samples_dq_B_H[i].scalar()
        if not np.allclose(
                scalar_parts_W_E.dq, scalar_parts_B_H.dq,
                atol=config.ransac_sample_rejection_scalar_part_equality_tolerance):
          good_sample = False
          prerejected_samples += 1
          break

      if not good_sample:
        continue

    # Transform all poses based on the first sample pose and rearange poses,
    # such that the first sample pose is the first pose.
    aligned_dq_B_H = align_paths_at_index(dq_B_H_vec, sample_indices[0])
    aligned_dq_W_E = align_paths_at_index(dq_W_E_vec, sample_indices[0])
    assert len(aligned_dq_B_H) == num_poses
    assert len(aligned_dq_W_E) == num_poses

    # Compute model and determine inliers
    dq_H_E_initial = None
    num_inliers = 0
    inlier_dq_B_H = []
    inlier_dq_W_E = []
    if config.ransac_inlier_classification == "rmse_threshold":
      assert config.ransac_sample_size >= 2, (
          "Cannot compute the hand eye calibration with a "
          "sample size of less than 2!")
      try:
        # Compute initial hand-eye calibration on SAMPLES only.
        (dq_H_E_initial,
         singular_values,
         bad_singular_values) = compute_hand_eye_calibration(
            aligned_samples_dq_B_H, aligned_samples_dq_W_E,
            config.hand_eye_calibration_scalar_part_equality_tolerance)
        dq_H_E_initial.normalize()
      except:
        print("\n\n Hand-eye calibration FAILED! "
              "algorithm_name: {} exception: \n\n".format(
                  config.algorithm_name, sys.exc_info()[0]))
        continue

      # Inliers are determined by evaluating the hand-eye calibration computed
      # based on the samples on all the poses and thresholding the RMSE of the
      # position/orientation.
      (poses_B_H, poses_W_H) = get_aligned_poses(aligned_dq_B_H,
                                                 aligned_dq_W_E,
                                                 dq_H_E_initial)
      (rmse_position,
       rmse_orientation,
       inlier_flags) = evaluate_alignment(poses_B_H, poses_W_H, config)
      assert inlier_flags[0], ("The sample idx used for alignment should be "
                               "part of the inlier set!")

      num_inlier_removed_due_to_scalar_part_inequality = 0
      for i in range(0, num_poses):
        if inlier_flags[i]:
          scalar_parts_W_E = aligned_dq_W_E[i].scalar()
          scalar_parts_B_H = aligned_dq_B_H[i].scalar()
          if not np.allclose(
                  scalar_parts_W_E.dq, scalar_parts_B_H.dq,
                  atol=config.ransac_sample_rejection_scalar_part_equality_tolerance):
            num_inlier_removed_due_to_scalar_part_inequality += 1
            inlier_flags[i] = False

      if num_inlier_removed_due_to_scalar_part_inequality > 0:
        print("WARNING: At least one inlier selected based on the "
              "position/orientation error did not pass the scalar part "
              "equality test! Use tighter values for "
              "ransac_position_error_threshold_m and "
              "ransac_orientation_error_threshold_deg. "
              "Inliers removed: {}".format(
                  num_inlier_removed_due_to_scalar_part_inequality))
        # TODO(mfehr): Find a good way to tune the parameters.
        continue

    elif config.ransac_inlier_classification == "scalar_part_equality":
      # Inliers are determined without computing an initial model but by simply
      # selecting all poses that have a matching scalar part.
      inlier_flags = [False] * num_poses
      for i in range(0, num_poses):
        scalar_parts_B_H = aligned_dq_B_H[i].scalar()
        scalar_parts_W_E = aligned_dq_W_E[i].scalar()
        if np.allclose(scalar_parts_W_E.dq, scalar_parts_B_H.dq,
                       atol=config.ransac_sample_rejection_scalar_part_equality_tolerance):
          inlier_flags[i] = True

    else:
      assert False, "Unkown ransac inlier classification."

    # Filter poses based on inlier flags.
    inlier_dq_B_H = list(compress(aligned_dq_B_H, inlier_flags))
    inlier_dq_W_E = list(compress(aligned_dq_W_E, inlier_flags))
    assert len(inlier_dq_B_H) == len(inlier_dq_W_E)
    num_inliers = len(inlier_dq_B_H)

    # Reject sample if not enough inliers.
    if num_inliers < config.min_num_inliers:
      print("==> Not enough inliers ({})".format(num_inliers))
      continue

    if (config.ransac_model_refinement or dq_H_E_initial is None):
      try:
        # Refine hand-calibration using all inliers.
        (dq_H_E_refined,
         singular_values,
         bad_singular_values) = compute_hand_eye_calibration(
            inlier_dq_B_H, inlier_dq_W_E,
            config.hand_eye_calibration_scalar_part_equality_tolerance)
        dq_H_E_refined.normalize()
      except:
        print("\n\n Hand-eye calibration FAILED! "
              "algorithm_name: {} exception: \n\n".format(
                  config.algorithm_name, sys.exc_info()[0]))
        continue
    else:
      assert dq_H_E_initial is not None
      dq_H_E_refined = dq_H_E_initial

    # Rerun evaluation to determine the RMSE for the refined
    # hand-eye calibration.
    if config.ransac_evaluate_refined_model_on_inliers_only:
      (poses_B_H, poses_W_H) = get_aligned_poses(inlier_dq_B_H,
                                                 inlier_dq_W_E,
                                                 dq_H_E_refined)
    else:
      (poses_B_H, poses_W_H) = get_aligned_poses(aligned_dq_B_H,
                                                 aligned_dq_W_E,
                                                 dq_H_E_refined)

    (rmse_position_refined,
     rmse_orientation_refined,
     inlier_flags) = evaluate_alignment(poses_B_H, poses_W_H, config)

    if (rmse_position_refined < best_rmse_position and
            rmse_orientation_refined < best_rmse_orientation):
      best_estimated_dq_H_E = dq_H_E_refined
      best_rmse_position = rmse_position_refined
      best_rmse_orientation = rmse_orientation_refined
      best_inlier_idx_set = sample_indices
      best_num_inliers = num_inliers
      best_singular_values = singular_values
      best_singular_value_status = bad_singular_values

      print("Found a new best sample: {}\n"
            "\t\tNumber of inliers: {}\n"
            "\t\tRMSE position:     {:10.4f}\n"
            "\t\tRMSE orientation:  {:10.4f}\n"
            "\t\tdq_H_E_initial:    {}\n"
            "\t\tdq_H_E_refined:    {}".format(
                sample_indices, num_inliers, rmse_position_refined,
                rmse_orientation_refined, dq_H_E_initial, dq_H_E_refined))
    else:
      print("Rejected sample: {}\n"
            "\t\tNumber of inliers: {}\n"
            "\t\tRMSE position:     {:10.4f}\n"
            "\t\tRMSE orientation:  {:10.4f}".format(
                sample_indices, num_inliers, rmse_position_refined,
                rmse_orientation_refined))

    full_iterations += 1

    # Abort RANSAC early based on prior about the outlier probability.
    if (not config.enable_exhaustive_search and config.ransac_enable_early_abort):
      s = config.ransac_sample_size
      w = (1.0 - config.ransac_outlier_probability)  # Inlier probability
      w_pow_s = w ** s
      required_iterations = (math.log(1. - config.ransac_success_probability_threshold) /
                             math.log(1. - w_pow_s))
      if (full_iterations > required_iterations):
        print("Reached a {}% probability that RANSAC succeeded in finding a sample "
              "containing only inliers, aborting ...".format(
                  config.ransac_success_probability_threshold * 100.))
        break
  if not config.enable_exhaustive_search:
    print("Finished RANSAC.")
    print("RANSAC iterations: {}".format(full_iterations))
    print("RANSAC early rejected samples: {}".format(prerejected_samples))
  else:
    print("Finished exhaustive search!")

  if best_estimated_dq_H_E is None:
    print("!!! RANSAC couldn't find a solution !!!")
    end_time = timeit.default_timer()
    runtime = end_time - start_time
    return (False, None, (best_rmse_position, best_rmse_orientation),
            best_num_inliers, num_poses_after_filtering, runtime,
            best_singular_values, best_singular_value_status)

  # Visualize best alignment.
  if config.visualize:
    aligned_dq_W_E = align_paths_at_index(dq_W_E_vec)
    aligned_dq_B_H = align_paths_at_index(dq_B_H_vec)

    (poses_B_H, poses_W_H) = get_aligned_poses(aligned_dq_B_H,
                                               aligned_dq_W_E,
                                               best_estimated_dq_H_E)
    (rmse_position_all,
     rmse_orientation_all,
     inlier_flags) = evaluate_alignment(poses_B_H, poses_W_H, config, config.visualize)

    every_nth_element = config.visualize_plot_every_nth_pose
    plot_poses([poses_B_H[:: every_nth_element],
                poses_W_H[:: every_nth_element]],
               True, title="3D Poses After Alignment")

  pose_vec = best_estimated_dq_H_E.to_pose()
  print("Solution found with sample: {}\n"
        "\t\tNumber of inliers: {}\n"
        "\t\tRMSE position:     {:10.4f}\n"
        "\t\tRMSE orientation:  {:10.4f}\n"
        "\t\tdq_H_E_refined:    {}\n"
        "\t\tpose_H_E_refined:  {}\n"
        "\t\tTranslation norm:  {:10.4f}".format(
            sample_indices, best_num_inliers, best_rmse_position,
            best_rmse_orientation, best_estimated_dq_H_E,
            pose_vec, np.linalg.norm(pose_vec[0:3])))

  if best_singular_values is not None:
    if best_singular_value_status:
      print("The singular values of this solution are bad. "
            "Either the smallest two are too big or the first 6 "
            "are too small! singular values: {}".format(best_singular_values))

  end_time = timeit.default_timer()
  runtime = end_time - start_time

  return (True, best_estimated_dq_H_E, (best_rmse_position, best_rmse_orientation),
          best_num_inliers, num_poses_after_filtering, runtime, best_singular_values, best_singular_value_status)
