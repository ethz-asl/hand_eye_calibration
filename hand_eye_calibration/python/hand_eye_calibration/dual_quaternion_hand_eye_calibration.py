
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import tf

from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.quaternion import (Quaternion, angle_between_quaternions)

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


class Arrow3D(FancyArrowPatch):

  def __init__(self, xs, ys, zs, *args, **kwargs):
    FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
    self._verts3d = xs, ys, zs

  def draw(self, renderer):
    xs3d, ys3d, zs3d = self._verts3d
    xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
    FancyArrowPatch.draw(self, renderer)


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
  assert np.allclose(dq_vec_starting_at_origin[align_index].dq,
                     [0., 0., 0., 1.0, 0., 0., 0., 0.],
                     atol=1.e-8), dq_vec_starting_at_origin[0]
  return dq_vec_starting_at_origin


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


def align(dq_W_E_vec, dq_B_H_vec, enforce_same_non_dual_scalar_sign=True, min_num_inliers=2, exhaustive_search=False):
  """Do the actual hand eye-calibration as described in the referenced paper."""
  n_quaternions = len(dq_W_E_vec)

  if enforce_same_non_dual_scalar_sign:
    for i in range(n_quaternions):
      dq_W_E = dq_W_E_vec[i]
      dq_B_H = dq_B_H_vec[i]
      if ((dq_W_E.q_rot.w < 0. and dq_B_H.q_rot.w > 0.) or
              (dq_W_E.q_rot.w > 0. and dq_B_H.q_rot.w < 0.)):
        dq_W_E_vec[i].dq = -dq_W_E_vec[i].dq.copy()

  best_idx = -1
  best_num_inliers = min_num_inliers - 1
  best_dq_W_E_vec_filtered = []
  best_dq_B_H_vec_filtered = []

  if exhaustive_search:
    print("Do exhaustive search to find biggest subset of inliers...")
  else:
    print("Search for first set of inliers bigger than {}...".format(min_num_inliers))

  # 0. Reject pairs where scalar parts of dual quaternions do not match.
  # Loop over all the indices to find an index of a pose pair.
  for j in range(n_quaternions):
    # Re-align all dual quaternion to the j-th dual quaternion.
    dq_W_E_vec = align_paths_at_index(dq_W_E_vec, j)
    dq_B_H_vec = align_paths_at_index(dq_B_H_vec, j)

    dq_W_E_vec_filtered = []
    dq_B_H_vec_filtered = []

    # Loop over the indices again starting at the first index to find either:
    # - The first set of inliers of at least size min_num_inliers
    #       OR
    # - The largest set of inliers using an exhaustive search
    for i in range(0, n_quaternions):
      dq_W_E = dq_W_E_vec[i]
      dq_B_H = dq_B_H_vec[i]
      scalar_parts_W_E = dq_W_E.scalar()
      scalar_parts_B_H = dq_B_H.scalar()
      # Append the inliers to the filtered dual quaternion vectors.
      if np.allclose(scalar_parts_W_E.dq, scalar_parts_B_H.dq, atol=1e-2):
        dq_W_E_vec_filtered.append(dq_W_E)
        dq_B_H_vec_filtered.append(dq_B_H)

    assert len(dq_W_E_vec_filtered) == len(dq_B_H_vec_filtered)

    if exhaustive_search:
      has_the_most_inliers = (len(dq_W_E_vec_filtered) > best_num_inliers)
      if has_the_most_inliers:
        best_num_inliers = len(dq_W_E_vec_filtered)
        best_idx = j
        best_dq_W_E_vec_filtered = copy.deepcopy(dq_W_E_vec_filtered)
        best_dq_B_H_vec_filtered = copy.deepcopy(dq_B_H_vec_filtered)
        print("Found new best start idx: {} number of inliers: {}".format(best_idx, best_num_inliers))
    else:
      has_enough_inliers = (len(dq_W_E_vec_filtered) > min_num_inliers)
      if has_enough_inliers:
        best_idx = j
        break

      if j + 1 >= n_quaternions:
        assert False, "Not enough inliers found."

  if exhaustive_search:
    assert best_idx != -1, "Not enough inliers found!"
    dq_W_E_vec_filtered = best_dq_W_E_vec_filtered
    dq_B_H_vec_filtered = best_dq_B_H_vec_filtered

  print("Best start idx: {}".format(best_idx))
  print("Removed {} outliers from the initial set of poses.".format(
      len(dq_W_E_vec) - len(dq_W_E_vec_filtered)))
  print("Running the hand-eye calibration with the remaining {} pairs of "
        "poses".format(len(dq_W_E_vec_filtered)))

  # 1.
  # Construct 6n x 8 matrix T
  t_matrix = setup_t_matrix(dq_W_E_vec_filtered, dq_B_H_vec_filtered)

  # 2.
  # Compute SVD of T and check if only two singular values are almost equal to
  # zero. Take the corresponding right-singular vectors (v_7 and v_8)
  U, s, V = np.linalg.svd(t_matrix)
  print("singular values: {}".format(s))

  # Check if only the last two singular values are almost zero.
  # for i, singular_value in enumerate(s):
  #   if i < 6:
  #     assert (singular_value >= 5e-2), s
  #   else:
  #     assert (singular_value < 5e-2), s
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
  return dq_H_E


def draw_poses(poses, additional_poses=None, plot_arrows=True):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  positions = ax.plot(xs=poses[:, 0], ys=poses[:, 1],
                      zs=poses[:, 2], color='blue')
  for pose in poses:
    # Position point
    ax.plot([pose[0]], [pose[1]], [pose[2]], 'o',
            markersize=10, color='blue', alpha=0.5)
    if not plot_arrows:
      continue
    t = tf.transformations.quaternion_matrix(pose[3:7].copy())
    # Add orientation arrow.
    x_arrow = np.array([1, 0, 0, 0]).copy()
    x_arrow_rotated = np.dot(t, x_arrow)
    a = Arrow3D(
        [pose[0], pose[0] + x_arrow_rotated[0]
         ], [pose[1], pose[1] + x_arrow_rotated[1]],
        [pose[2], pose[2] + x_arrow_rotated[2]],
        mutation_scale=20,
        lw=3,
        arrowstyle="-|>",
        color="b")
    ax.add_artist(a)

    y_arrow = np.array([0, 1, 0, 0]).copy()
    y_arrow_rotated = np.dot(t, y_arrow)
    a = Arrow3D(
        [pose[0], pose[0] + y_arrow_rotated[0]
         ], [pose[1], pose[1] + y_arrow_rotated[1]],
        [pose[2], pose[2] + y_arrow_rotated[2]],
        mutation_scale=20,
        lw=3,
        arrowstyle="-|>",
        color="c")
    ax.add_artist(a)
  if additional_poses is not None:
    positions_2 = ax.plot(
        xs=additional_poses[:, 0],
        ys=additional_poses[:, 1],
        zs=additional_poses[:, 2],
        color='red')
    plt.legend(iter(positions + positions_2), ('pos1', 'pos2'))
    for pose in additional_poses:
      # Position point
      ax.plot([pose[0]], [pose[1]], [pose[2]], 'o',
              markersize=10, color='red', alpha=0.5)
      if not plot_arrows:
        continue
      # Add orientation arrow.
      x_arrow = np.array([1, 0, 0, 0]).copy()
      t = tf.transformations.quaternion_matrix(pose[3:7].copy())
      arrow_rotated = np.dot(t, x_arrow)
      a = Arrow3D(
          [pose[0], pose[0] + arrow_rotated[0]
           ], [pose[1], pose[1] + arrow_rotated[1]],
          [pose[2], pose[2] + arrow_rotated[2]],
          mutation_scale=20,
          lw=3,
          arrowstyle="-|>",
          color="r")
      ax.add_artist(a)
      y_arrow = np.array([0, 1, 0, 0]).copy()
      y_arrow_rotated = np.dot(t, y_arrow)
      a = Arrow3D(
          [pose[0], pose[0] + y_arrow_rotated[0]
           ], [pose[1], pose[1] + y_arrow_rotated[1]],
          [pose[2], pose[2] + y_arrow_rotated[2]],
          mutation_scale=20,
          lw=3,
          arrowstyle="-|>",
          color="y")
      ax.add_artist(a)
    # TODO(ff): Connect the corresponding points.

  plt.show(block=True)


def evaluate_alignment(poses_A, poses_B, visualize=False):
  assert np.array_equal(poses_A.shape, poses_B.shape), (
      "Two pose vector of different size cannot be evaluated. "
      "Size pose A: {} Size pose B: {}".format(poses_A.shape, poses_B.shape))
  assert poses_A.shape[1] == 7, "poses_A are not valid poses!"
  assert poses_B.shape[1] == 7, "poses_B are not valid poses!"

  num_poses = poses_A.shape[0]

  errors_position = np.zeros((num_poses, 1))
  errors_orientation = np.zeros((num_poses, 1))
  for i in range(0, num_poses):
    # Sum up the squared norm of the pose error.
    errors_position[i] = np.linalg.norm(poses_A[i, 0:3] - poses_B[i, 0:3], ord=2)

    # Construct quaternions to compare.
    quaternion_A = Quaternion(q=poses_A[i, 3:7])
    quaternion_A.normalize()
    if quaternion_A.w < 0:
      quaternion_A.q = -quaternion_A.q
    quaternion_B = Quaternion(q=poses_B[i, 3:7])
    quaternion_B.normalize()
    if quaternion_B.w < 0:
      quaternion_B.q = -quaternion_B.q

    # Sum up the square of the orientation angle error.
    error_angle_rad = angle_between_quaternions(
        quaternion_A, quaternion_B)
    error_angle_degrees = math.degrees(error_angle_rad)
    if error_angle_degrees > 180.0:
      error_angle_degrees = math.fabs(360.0 - error_angle_degrees)
    errors_orientation[i] = error_angle_degrees

  rmse_pose_accumulator = np.sum(np.square(errors_position))
  rmse_orientation_accumulator = np.sum(np.square(errors_orientation))

  # Compute RMSE.
  rmse_pose = math.sqrt(rmse_pose_accumulator / num_poses)
  rmse_orientation = math.sqrt(rmse_orientation_accumulator / num_poses)

  # Plot the error.
  if visualize:
    title_position = 1.05
    fig = plt.figure()
    a1 = fig.add_subplot(2, 1, 1)
    fig.suptitle("Alignment Evaluation", fontsize='24')
    a1.set_title(
        "Red = Position Error Norm [m] - Black = RMSE", y=title_position)
    plt.plot(errors_position, c='r')
    plt.plot(rmse_pose * np.ones((num_poses, 1)), c='k')
    a2 = fig.add_subplot(2, 1, 2)
    a2.set_title(
        "Red = Absolute Orientation Error [Degrees] - Black = RMSE", y=title_position)
    plt.plot(errors_orientation, c='r')
    plt.plot(rmse_orientation * np.ones((num_poses, 1)), c='k')
    if plt.get_backend() == 'TkAgg':
      mng = plt.get_current_fig_manager()
      max_size = mng.window.maxsize()
      max_size = (max_size[0], max_size[1] * 0.45)
      mng.resize(*max_size)
    fig.tight_layout()
    plt.subplots_adjust(left=0.025, right=0.975, top=0.8, bottom=0.05)
    plt.show()
  return (rmse_pose, rmse_orientation)
