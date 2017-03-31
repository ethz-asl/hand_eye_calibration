import tf
import numpy as np

from hand_eye_calibration.dual_quaternion_hand_eye_calibration import (
    align, draw_poses, make_paths_start_at_origin,
    compute_dual_quaternions_with_offset)
from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.quaternion import Quaternion


def random_quaternion():
  quaternion = tf.transformations.random_quaternion()
  if quaternion[3] < 0.0:
    quaternion = -quaternion
  assert np.isclose(np.linalg.norm(quaternion), 1.0, atol=1.e-8)
  return quaternion.copy()


def random_rotation():
  return tf.transformations.random_rotation_matrix()


def random_translation():
  translation = tf.transformations.random_vector(3)
  translation_transformation = tf.transformations.translation_matrix(
      translation)
  return translation_transformation


def rand_transform():
  return np.dot(random_translation(), random_rotation()).copy()


def random_transform_as_dual_quaternion(
        enforce_positive_non_dual_scalar_sign=True):
  T_rand = rand_transform()
  dq_rand = DualQuaternion.from_transformation_matrix(T_rand)
  if(enforce_positive_non_dual_scalar_sign):
    if(dq_rand.q_rot.w < 0):
      dq_rand.dq = -dq_rand.dq.copy()
  return dq_rand.copy()


def generate_test_paths(n_samples, dq_H_E, dq_B_W, paths_start_at_origin=True):
  dq_B_H_vec = generate_test_path(n_samples)

  if paths_start_at_origin:
    dq_B_H_vec = make_paths_start_at_origin(dq_B_H_vec)

  # Generate other trajectory with contant offset.
  dq_W_E_vec = compute_dual_quaternions_with_offset(dq_B_H_vec, dq_H_E, dq_B_W)

  if paths_start_at_origin:
    dq_W_E_vec = make_paths_start_at_origin(dq_W_E_vec)

  return dq_B_H_vec, dq_W_E_vec


def generate_test_path(n_samples):
  # Create a sine for x, cos for y and linear motion for z.
  # Rotate around the curve, while keeping the x-axis perpendicular to the
  # curve.
  dual_quaternions = []
  t = np.linspace(0, 1, num=n_samples)
  x = np.cos(4 * np.pi * t)
  y = -np.sin(4 * np.pi * t)
  z = t * 5
  theta = 4 * np.pi * t

  for i in range(n_samples):
    q_tmp = tf.transformations.quaternion_from_euler(
        0., -theta[i] * 0.1, -theta[i], 'rxyz')
    q = Quaternion(q=q_tmp)
    q.normalize()
    if q.w < 0.0:
      q = -q
    dq = DualQuaternion.from_pose(x[i], y[i], z[i], q.x, q.y, q.z, q.w)
    dual_quaternions.append(dq)

  return dual_quaternions
