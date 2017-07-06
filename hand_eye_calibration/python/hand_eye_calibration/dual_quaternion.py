from numbers import Number

import numpy as np
import numpy.testing as npt

from quaternion import Quaternion


class DualQuaternion(object):
  """ Clifford dual quaternion denoted as dq = q_rot + epsilon * q_dual.

  Can be instantiated by:
  >>> dq = DualQuaternion(q_rot, q_dual)
  >>> dq = DualQuaternion.from_vector([qrx, qry, qrz, qrw, qtx, qty, qtz, qtw])
  >>> dq = DualQuaternion.from_pose(x, y, z, q_x, q_y, q_z, q_w)
  >>> dq = DualQuaternion.from_pose_vector([x, y, z, q_x, q_y, q_z, q_w])
  >>> dq = DualQuaternion(q_rot, q_dual)
  >>> # Given a [4x4] transformation matrix T.
  >>> dq = DualQuaternion.from_transformation_matrix(T)
  """
  dq = np.array([0., 0., 0., 1.0, 0., 0., 0., 0.]).T

  def __init__(self, q_rot, q_dual):
    for i in [q_rot, q_dual]:
      assert isinstance(
          i, Quaternion), "q_rot and q_dual should be quaternions."

    self.dq = np.array([0., 0., 0., 1.0, 0., 0., 0., 0.]).T

    # Assign real part (rotation).
    self.dq[0:4] = q_rot.q.copy()

    # Assign dual part (translation).
    self.dq[4:8] = q_dual.q.copy()

    self.assert_normalization()

  def __str__(self):
    return "[q_rot: {}, q_dual: {}]".format(np.str(self.q_rot), np.str(self.q_dual))

  def __repr__(self):
    return ("<Dual quaternion q_rot {} q_dual {}>").format(self.q_rot, self.q_dual)

  def __add__(self, other):
    """ Dual quaternion addition. """
    dq_added = self.dq + other.dq
    return DualQuaternion.from_vector(dq_added)

  def __sub__(self, other):
    """ Dual quaternion subtraction. """
    dq_sub = self.dq - other.dq
    return DualQuaternion.from_vector(dq_sub)

  def __mul__(self, other):
    """ Dual quaternion multiplication.

    The multiplication with a scalar returns the dual quaternion with all
    elements multiplied by the scalar.

    The multiplication of two dual quaternions dq1 and dq2 as:
    q1_rot * q2_rot + epsilon * (q1_rot * q2_trans + q1_trans * q2_rot),
    where dq1 and dq2 are defined as:
    dq1 = q1_rot + epsilon * q1_trans,
    dq2 = q2_rot + epsilon * q2_trans.
    """
    if isinstance(other, DualQuaternion):
      rotational_part = self.q_rot * other.q_rot
      translational_part = (self.q_rot * other.q_dual +
                            self.q_dual * other.q_rot)
      return DualQuaternion(rotational_part.copy(), translational_part.copy())
    elif isinstance(other, Number):
      dq = self.dq.copy()
      dq_out = dq * np.float64(other)
      return DualQuaternion.from_vector(dq_out)
    else:
      assert False, ("Multiplication is only defined for scalars or dual " "quaternions.")

  def __rmul__(self, other):
    """ Scalar dual quaternion multiplication.

    The multiplication with a scalar returns the dual quaternion with all
    elements multiplied by the scalar.
    """
    if isinstance(other, Number):
      dq = self.dq.copy()
      dq_out = np.float64(other) * dq
      return DualQuaternion.from_vector(dq_out)
    else:
      assert False, ("Multiplication is only defined for scalars or dual " "quaternions.")

  def __truediv__(self, other):
    """ Quaternion division with either scalars or quaternions.

    The division with a scalar returns the dual quaternion with all
    translational elements divided by the scalar.

    The division with a dual quaternion returns dq = dq1/dq2 = dq1 * dq2^-1,
    hence other divides on the right.
    """
    # TODO(ff): Check if this is correct.
    print("WARNING: This might not be properly implemented.")
    if isinstance(other, DualQuaternion):
      return self * other.inverse()
    elif isinstance(other, Number):
      dq = self.dq.copy()
      dq_out = dq / np.float64(other)
      return DualQuaternion.from_vector(dq_out)
    else:
      assert False, "Division is only defined for scalars or dual quaternions."

  def __div__(self, other):
    """ Quaternion division with either scalars or quaternions.

    The division with a scalar returns the dual quaternion with all
    translational elements divided by the scalar.

    The division with a dual quaternion returns dq = dq1 / dq2 = dq1 * dq2^-1.
    """
    return self.__truediv__(other)

  def __eq__(self, other):
    """ Check equality. """
    if isinstance(other, DualQuaternion):
      return np.allclose(self.dq, other.dq)
    else:
      return False

  @classmethod
  def from_vector(cls, dq):
    dual_quaternion_vector = None
    if isinstance(dq, np.ndarray):
      dual_quaternion_vector = dq.copy()
    else:
      dual_quaternion_vector = np.array(dq)
    return cls(Quaternion(q=dual_quaternion_vector[0:4]),
               Quaternion(q=dual_quaternion_vector[4:8]))

  @classmethod
  def from_pose(cls, x, y, z, rx, ry, rz, rw):
    """ Create a normalized dual quaternion from a pose. """
    qr = Quaternion(rx, ry, rz, rw)
    qr.normalize()
    qt = (Quaternion(x, y, z, 0) * qr) * 0.5
    return cls(qr, qt)

  @classmethod
  def from_pose_vector(cls, pose):
    """ Create a normalized dual quaternion from a pose vector. """
    return cls.from_pose(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6])

  @classmethod
  def from_transformation_matrix(cls, transformation_matrix):
    q_rot = Quaternion.from_rotation_matrix(transformation_matrix[0:3, 0:3])

    pose_vec = np.zeros(7)
    pose_vec[3:7] = q_rot.q
    pose_vec[0:3] = transformation_matrix[0:3, 3]

    return cls.from_pose_vector(pose_vec)

  @classmethod
  def identity(cls):
    identity_q_rot = Quaternion(0., 0., 0., 1.)
    identity_q_dual = Quaternion(0., 0., 0., 0.)
    return cls(identity_q_rot, identity_q_dual)

  # def conjugate_translation(self):
  #   """ Dual quaternion translation conjugate. """
  #   return DualQuaternion(self.q_rot.conjugate(), -self.q_dual.conjugate())

  def conjugate(self):
    """ Dual quaternion multiplication conjugate. """
    return DualQuaternion(self.q_rot.conjugate(), self.q_dual.conjugate())

  def inverse(self):
    """ Dual quaternion inverse. """
    assert self.norm()[0] > 1e-8
    return DualQuaternion(self.q_rot.inverse(),
                          -self.q_rot.inverse() * self.q_dual * self.q_rot.inverse())

  def enforce_positive_q_rot_w(self):
    """ Enforce a positive real part of the rotation quaternion. """
    assert self.norm()[0] > 1e-8
    if self.q_rot.w < 0.0:
      self.dq = -self.dq

  def norm(self):
    """ The norm of a dual quaternion. """
    assert self.q_rot.norm() > 1e-8, (
        "Dual quaternion has rotational part equal to zero, hence the norm is"
        "not defined.")
    real_norm = self.q_rot.norm()
    dual_norm = np.dot(self.q_rot.q, self.q_dual.q) / real_norm
    return (real_norm, dual_norm)

  def is_normalized(self):
    real_part = np.absolute(self.norm()[0] - 1.0) < 1e-8
    dual_part = np.absolute(self.norm()[1]) < 1e-8
    return real_part and dual_part

  def assert_normalization(self):
    assert self.is_normalized, "Something went wrong, the dual quaternion is not normalized!"

  def normalize(self):
    """ Normalize the dual quaternion. """
    real_norm = self.q_rot.norm()
    self.dq[0:4] = self.q_rot.q / real_norm
    self.dq[4:8] = self.q_dual.q / real_norm
    self.assert_normalization()

  def scalar(self):
    """ The scalar part of the dual quaternion.

    Defined as: scalar(dq) := 0.5*(dq+dq.conjugate())
    """
    scalar_part = 0.5 * (self + self.conjugate())
    npt.assert_allclose(
        scalar_part.dq[[0, 1, 2, 4, 5, 6]], np.zeros(6), atol=1e-6)
    return scalar_part.copy()

  def screw_axis(self):
    """ The rotation, translation and screw axis from the dual quaternion. """
    rotation = 2. * np.degrees(np.arccos(self.q_rot.w))
    rotation = np.mod(rotation, 360.)

    if (rotation > 1.e-12):
      translation = -2. * self.q_dual.w / np.sin(rotation / 2. * np.pi / 180.)
      screw_axis = self.q_rot.q[0:3] / np.sin(rotation / 2. * np.pi / 180.)
    else:
      translation = 2. * np.sqrt(np.sum(np.power(self.q_dual.q[0:3], 2.)))
      if (translation > 1.e-12):
        screw_axis = 2. * self.q_dual.q[0:3] / translation
      else:
        screw_axis = np.zeros(3)

    # TODO(ntonci): Add axis point for completeness

    return screw_axis, rotation, translation

  def passive_transform_point(self, point):
    """ Applies the passive transformation of the dual quaternion to a point.
    """
    # TODO(ff): Check if the rotation is in the right direction.
    point_dq = DualQuaternion.from_pose(
        point[0], point[1], point[2], 0, 0, 0, 1)
    dq_in_new_frame = self * point_dq
    return dq_in_new_frame.to_pose()[0:3]

  def active_transform_point(self, point):
    """ Applies the active transformation of the dual quaternion to a point.
    """
    return self.inverse().passive_transform_point(point)

  # TODO(ff): Implement translational velocity, rotational velocity.
  # See for instance:
  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3576712/pdf/fnbeh-07-00007.pdf

  def to_matrix(self):
    """ Returns a [4x4] transformation matrix. """
    self.normalize()
    matrix_out = np.identity(4)
    matrix_out[0:3, 0:3] = self.q_rot.to_rotation_matrix()
    matrix_out[0:3, 3] = self.to_pose()[0:3]
    return matrix_out.copy()

  def to_pose(self):
    """ Returns a [7x1] pose vector.

    In the form: pose = [x, y, z, qx, qy, qz, qw].T.
    """
    self.normalize()

    pose = np.zeros(7)
    q_rot = self.q_rot
    if (q_rot.w < 0.):
      q_rot = -q_rot
    translation = (2.0 * self.q_dual) * q_rot.conjugate()

    pose[0:3] = translation.q[0:3].copy()
    pose[3:7] = q_rot.q.copy()
    return pose.copy()

  def copy(self):
    """ Copy dual quaternion. """
    return DualQuaternion.from_vector(self.dq)

  @property
  def q_rot(self):
    return Quaternion(q=self.dq[0:4])

  @property
  def q_dual(self):
    return Quaternion(q=self.dq[4:8])

  @property
  def r_x(self):
    return self.dq[0]

  @property
  def r_y(self):
    return self.dq[1]

  @property
  def r_z(self):
    return self.dq[2]

  @property
  def r_w(self):
    return self.dq[3]

  @property
  def t_x(self):
    return self.dq[4]

  @property
  def t_y(self):
    return self.dq[5]

  @property
  def t_z(self):
    return self.dq[6]

  @property
  def t_w(self):
    return self.dq[7]
