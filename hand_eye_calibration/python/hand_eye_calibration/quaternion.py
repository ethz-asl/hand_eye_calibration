from numbers import Number

import numpy as np
import random


class Quaternion(object):
  """ Hamiltonian quaternion denoted as q = [x y z w].T.

  Can be instantiated by:
  >>> q = Quaternion(x, y, z, w)
  >>> q = Quaternion(q=[x, y, z, w])
  >>> q = Quaternion(q=np.array([x, y, z, w]))
  """
  q = np.array([0.0, 0.0, 0.0, 1.0]).T

  def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0, q=None):
    if q is None:
      for i in [x, y, z, w]:
        assert isinstance(i, Number), "x, y, z, w should be scalars."
      self.q = np.array([x, y, z, w]).T
    elif isinstance(q, np.ndarray):
      self.q = q.copy()
    else:
      print("This is not supported. Type of q is {}".format(type(q)))
      assert False

  def __str__(self):
    return np.str(self.q)

  def __repr__(self):
    return "<Quaternion x:{} y:{} z:{} w:{}>".format(self.x, self.y, self.z, self.w)

  def __add__(self, other):
    """ Quaternion addition. """
    q_added = self.q + other.q
    return Quaternion(q=q_added)

  def __neg__(self):
    return Quaternion(q=-self.q)

  def __sub__(self, other):
    """ Quaternion subtraction. """
    q_added = self.q - other.q
    return Quaternion(q=q_added)

  def __mul__(self, other):
    """ Scalar and Hamilton quaternion product.

    The multiplication with a scalar returns the quaternion with all elements
    multiplied by the scalar.

    The multiplication with a quaternion returns the Hamilton product.
    """
    if isinstance(other, Quaternion):
      x = (self.w * other.x + self.x * other.w +
           self.y * other.z - self.z * other.y)
      y = (self.w * other.y - self.x * other.z +
           self.y * other.w + self.z * other.x)
      z = (self.w * other.z + self.x * other.y -
           self.y * other.x + self.z * other.w)
      w = (self.w * other.w - self.x * other.x -
           self.y * other.y - self.z * other.z)
      return Quaternion(x, y, z, w)
    elif isinstance(other, Number):
      q = self.q.copy()
      q_out = q * np.float64(other)
      return Quaternion(q=q_out)
    else:
      assert False, "Multiplication is only defined for scalars or quaternions."

  def __rmul__(self, other):
    """ Scalar quaternion multiplication.

    The multiplication with a scalar returns the quaternion with all elements
    multiplied by the scalar.
    """
    if isinstance(other, Number):
      q = self.q.copy()
      q_out = np.float64(other) * q
      return Quaternion(q=q_out)
    else:
      assert False, "Multiplication is only defined for scalars or quaternions."

  def __truediv__(self, other):
    """ Quaternion division with either scalars or quaternions.

    The division with a scalar returns the quaternion with all elements divided
    by the scalar.

    The division with a quaternion returns q = q1 / q2 = q1 * q2^-1.
    """
    if isinstance(other, Quaternion):
      return self * other.inverse()
    elif isinstance(other, Number):
      q = self.q.copy()
      q_out = q / np.float64(other)
      return Quaternion(q=q_out)
    else:
      assert False, "Division is only defined for scalars or quaternions."

  def __div__(self, other):
    return self.__truediv__(other)

  @classmethod
  def from_rotation_matrix(cls, rotation_matrix):
    w = np.sqrt(
        1.0 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
    w4 = 4.0 * w
    x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / w4
    y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / w4
    z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / w4
    return cls(x, y, z, w)

  @classmethod
  def from_angle_axis(cls, angle, axis):
    q = cls(axis[0], axis[1], axis[2], 0.0)
    q_norm = q.norm()
    if q_norm > 1e-16:
      q *= np.sin(angle / 2.0) / q_norm
    q.q[3] = np.cos(angle / 2.0)
    return cls(q=q.q.copy())

  @classmethod
  def get_random(cls, min_angle=0., max_angle=np.math.pi):
    axis = np.random.uniform(-1.0, 1.0, 3)
    axis /= np.linalg.norm(axis)
    assert np.isclose(np.linalg.norm(axis), 1., atol=1e-8)

    angle = random.uniform(
        min_angle, max_angle)
    q = cls.from_angle_axis(angle, axis)
    return cls(q=q.q.copy())

  def angle_axis(self):
    """ Returns the axis and angle of a quaternion.

    The output format is np.array([x, y, z, angle]).
    """
    # If there is no rotation return rotation about x-axis with zero angle.
    if np.isclose(self.w, 1., atol=1.e-12):
      return(np.array([1., 0., 0., 0.]))

    angle = 2. * np.arccos(self.w)
    x = self.x / np.sqrt(1. - self.w * self.w)
    y = self.y / np.sqrt(1. - self.w * self.w)
    z = self.z / np.sqrt(1. - self.w * self.w)
    return np.array([x, y, z, angle])

  def copy(self):
    """ Copy quaternion. """
    return Quaternion(q=self.q.copy())

  def conjugate(self):
    """ Quaternion conjugate. """
    return Quaternion(-self.x, -self.y, -self.z, self.w)

  def squared_norm(self):
    """ The squared norm of a quaternion. """
    return np.dot(self.q.T, self.q)

  def inverse(self):
    """ Quaternion inverse. """
    return self.conjugate() / self.squared_norm()

  def norm(self):
    """ The norm of a quaternion. """
    return np.sqrt(np.dot(self.q.T, self.q))

  def normalize(self):
    """ Normalize the quaternion. """
    norm = self.norm()
    if norm > 1e-8:
      self.q = self.q / norm
    # assert norm > 1e-16

  def rotate_vector(self, vector):
    """ Rotate a vector by the quaternion. """
    vector_rotated = np.zeros(3)
    vector_rotated[0] = ((1 - 2 * self.y**2 - 2 * self.z**2) * vector[0] +
                         2 * (self.x * self.y + self.w * self.z) * vector[1] +
                         2 * (self.x * self.z - self.w * self.y) * vector[2])
    vector_rotated[1] = (2 * (self.x * self.y - self.w * self.z) * vector[0] +
                         (1 - 2 * self.x**2 - 2 * self.z**2) * vector[1] +
                         2 * (self.y * self.z + self.w * self.x) * vector[2])
    vector_rotated[2] = (2 * (self.x * self.z + self.w * self.y) * vector[0] +
                         2 * (self.y * self.z - self.w * self.x) * vector[1] +
                         (1 - 2 * self.x**2 - 2 * self.y**2) * vector[2])
    return vector_rotated.copy()

  def to_rotation_matrix(self):
    """ Return the [3x3] rotation matrix. """
    # 1 - 2*qy^2 - 2*qz^2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw
    # 2*qx*qy + 2*qz*qw, 1 - 2*qx^2 - 2*qz^2, 2*qy*qz - 2*qx*qw
    # 2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx^2 - 2*qy^2
    rotation_matrix = np.zeros([3, 3])
    rotation_matrix[0, 0] = 1 - 2 * self.y**2 - 2 * self.z**2
    rotation_matrix[0, 1] = 2 * self.x * self.y - 2 * self.z * self.w
    rotation_matrix[0, 2] = 2 * self.x * self.z + 2 * self.y * self.w
    rotation_matrix[1, 0] = 2 * self.x * self.y + 2 * self.z * self.w
    rotation_matrix[1, 1] = 1 - 2 * self.x**2 - 2 * self.z**2
    rotation_matrix[1, 2] = 2 * self.y * self.z - 2 * self.x * self.w
    rotation_matrix[2, 0] = 2 * self.x * self.z - 2 * self.y * self.w
    rotation_matrix[2, 1] = 2 * self.y * self.z + 2 * self.x * self.w
    rotation_matrix[2, 2] = 1 - 2 * self.x**2 - 2 * self.y**2
    return rotation_matrix.copy()

  def to_transformation_matrix(self):
    """ Return the [4x4] transformation matrix (zero translation). """
    transformation_matrix = np.identity(4)
    transformation_matrix[0:3, 0:3] = self.to_rotation_matrix()
    return transformation_matrix.copy()

  @property
  def x(self):
    return self.q[0]

  @property
  def y(self):
    return self.q[1]

  @property
  def z(self):
    return self.q[2]

  @property
  def w(self):
    return self.q[3]


def quaternion_slerp(q_1, q_2, fraction):
  """ Quaternion slerp between q_1 and q_2 at fraction ([0,1]).

  Spherical linear quaternion interpolation method.
  q_1 and q_2 should be unit quaternions.
  """
  assert fraction >= 0.0 and fraction <= 1.0, "fraction should be in [0,1]."
  assert isinstance(q_1, Quaternion), "q_1 should be of Quaternion type."
  assert isinstance(q_2, Quaternion), "q_2 should be of Quaternion type."
  assert np.isclose(
      q_1.norm(), 1.0, atol=1.e-8), "Slerp should only be used with unit quaternions."
  assert np.isclose(
      q_2.norm(), 1.0, atol=1.e-8), "Slerp should only be used with unit quaternions."

  q1 = q_1.q.copy()
  q2 = q_2.q.copy()
  if fraction == 0.0:
    return Quaternion(q=q1)
  if fraction == 1.0:
    return Quaternion(q=q2)

  dot_product = np.dot(q1, q2)
  if dot_product < 0.0:
    dot_product = -dot_product
    q1 *= -1

  # Stay within the domain of acos().
  dot_product = np.clip(dot_product, -1.0, 1.0)

  # Angle between q_1 and q_2.
  theta_prime = np.arccos(dot_product)
  theta = theta_prime * fraction

  q_3 = Quaternion(q=q2 - q1 * dot_product)
  q_3.normalize()

  return Quaternion(q=q1 * np.cos(theta)) + q_3 * np.sin(theta)


def quaternion_lerp(q_1, q_2, fraction):
  """ Quaternion lerp.

  Linear quaternion interpolation method.
  """
  assert fraction >= 0.0 and fraction <= 1.0, "fraction should be in [0,1]."
  assert isinstance(q_1, Quaternion), "q_1 should be of Quaternion type."
  assert isinstance(q_2, Quaternion), "q_2 should be of Quaternion type."
  assert np.isclose(
      q_1.norm(), 1.0, atol=1.e-8), "Slerp should only be used with unit quaternions."
  assert np.isclose(
      q_2.norm(), 1.0, atol=1.e-8), "Slerp should only be used with unit quaternions."

  q1 = q_1.q.copy()
  q2 = q_2.q.copy()
  if fraction == 0.0:
    return Quaternion(q=q1)
  if fraction == 1.0:
    return Quaternion(q=q2)

  dot_product = np.dot(q1, q2)
  return Quaternion(q=q1 + fraction * (q2 - q1))


def quaternion_nlerp(q_1, q_2, fraction):
  """ Normalized quaternion lerp. """
  q = quaternion_lerp(q_1, q_2, fraction)
  q.normalize()
  return q


def quaternions_interpolate(q_left, t_left, q_right, t_right, times):
  """ Returns an array of the interpolated quaternions between q_left/t_left and
  q_right/t_right at times.
  """

  times_scaled = (times - t_left) / (t_right - t_left)
  return [quaternion_slerp(q_left, q_right, time) for time in times_scaled]


def angle_between_quaternions(q1, q2):
  """ Returns the angle between two quaternions, q1 and q2. """
  if np.allclose(q1.q, q2.q, atol=1.e-12):
    return 0.0
  return 2. * np.arccos((q1 * q2.inverse()).w)


def angular_velocity_between_quaternions(q1, q2, t):
  """ Returns the angular velocity resulting from transitioning from q1 to q2
  in t. """
  q1_q2_inv = q1 * q2.inverse()
  # Ensure positive w.
  if q1_q2_inv.w < 0.:
    q1_q2_inv = -q1_q2_inv
  angle_axis = q1_q2_inv.angle_axis()
  return 0.5 / t * angle_axis[3] * angle_axis[0:3]
