
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import copy
import numpy as np
import tf

from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.quaternion import Quaternion


class Arrow3D(FancyArrowPatch):

  def __init__(self, xs, ys, zs, *args, **kwargs):
    FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
    self._verts3d = xs, ys, zs

  def draw(self, renderer):
    xs3d, ys3d, zs3d = self._verts3d
    xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
    FancyArrowPatch.draw(self, renderer)


def plot_poses(poses, additional_poses=None, plot_arrows=True, title=""):
  title_position = 1.05
  fig = plt.figure()
  if title:
    fig.suptitle(title, fontsize='24')
  ax = fig.add_subplot(111, projection='3d')

  # Make copy, otherwise the data is manipulated.
  poses_A = poses.copy()

  # Tranform position from device into world frame.
  for pose in poses_A:
    quaternion = Quaternion(q=pose[3:7])
    pose[0:3] = quaternion.rotate_vector(pose[0:3])

  # Compute dimensions for visualization and legend.
  min_A = np.amin(poses_A[:, 0:3])
  max_A = np.amax(poses_A[:, 0:3])
  span_of_trajectories = max_A - min_A
  positions = ax.plot(xs=poses_A[:, 0], ys=poses_A[:, 1],
                      zs=poses_A[:, 2], color='blue')
  plt.legend(iter(positions), ('3D poses A'))

  poses_B = None
  if additional_poses is not None:
    poses_B = additional_poses.copy()

    # Tranform position from device into world frame.
    if poses_B is not None:
      for pose in poses_B:
        quaternion = Quaternion(q=pose[3:7])
        pose[0:3] = quaternion.rotate_vector(pose[0:3])

    # Compute dimensions for visualization and legend.
    min_B = np.amin(poses_B[:, 0:3])
    max_B = np.amax(poses_B[:, 0:3])
    span_of_trajectories = max(span_of_trajectories, max_B - min_B)

    positions_2 = ax.plot(
        xs=poses_B[:, 0],
        ys=poses_B[:, 1],
        zs=poses_B[:, 2],
        color='red')
    plt.legend(iter(positions + positions_2), ('3D poses A', '3D poses B'))

  # Arrows are about 1% of the span of the trajectories.
  arrow_size = span_of_trajectories * 0.01
  print("Plot arrows of size: {}m".format(arrow_size))

  for pose in poses_A:
    # Position point
    ax.plot([pose[0]], [pose[1]], [pose[2]], 'o',
            markersize=5, color='blue', alpha=0.5)
    if not plot_arrows:
      continue
    t = tf.transformations.quaternion_matrix(pose[3:7].copy())
    # Add orientation arrow.
    x_arrow = np.array([1, 0, 0, 0]).copy()
    x_arrow_rotated = np.dot(t, x_arrow)
    x_arrow_rotated *= arrow_size
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
    y_arrow_rotated *= arrow_size
    a = Arrow3D(
        [pose[0], pose[0] + y_arrow_rotated[0]
         ], [pose[1], pose[1] + y_arrow_rotated[1]],
        [pose[2], pose[2] + y_arrow_rotated[2]],
        mutation_scale=20,
        lw=3,
        arrowstyle="-|>",
        color="c")
    ax.add_artist(a)

  if poses_B is not None:
    for pose in poses_B:

      # Position point
      ax.plot([pose[0]], [pose[1]], [pose[2]], 'o',
              markersize=5, color='red', alpha=0.5)
      if not plot_arrows:
        continue
      # Add orientation arrow.
      x_arrow = np.array([1, 0, 0, 0]).copy()
      t = tf.transformations.quaternion_matrix(pose[3:7].copy())
      x_arrow_rotated = np.dot(t, x_arrow)
      x_arrow_rotated *= arrow_size
      a = Arrow3D(
          [pose[0], pose[0] + x_arrow_rotated[0]
           ], [pose[1], pose[1] + x_arrow_rotated[1]],
          [pose[2], pose[2] + x_arrow_rotated[2]],
          mutation_scale=20,
          lw=3,
          arrowstyle="-|>",
          color="r")
      ax.add_artist(a)
      y_arrow = np.array([0, 1, 0, 0]).copy()
      y_arrow_rotated = np.dot(t, y_arrow)
      y_arrow_rotated *= arrow_size
      a = Arrow3D(
          [pose[0], pose[0] + y_arrow_rotated[0]
           ], [pose[1], pose[1] + y_arrow_rotated[1]],
          [pose[2], pose[2] + y_arrow_rotated[2]],
          mutation_scale=20,
          lw=3,
          arrowstyle="-|>",
          color="y")
      ax.add_artist(a)

  plt.show(block=True)


def plot_alignment_errors(errors_position, rmse_pose, errors_orientation, rmse_orientation):
  assert np.array_equal(errors_position.shape, errors_orientation.shape)

  num_error_values = errors_position.shape[0]

  title_position = 1.05
  fig = plt.figure()
  a1 = fig.add_subplot(2, 1, 1)
  fig.suptitle("Alignment Evaluation", fontsize='24')
  a1.set_title(
      "Red = Position Error Norm [m] - Black = RMSE", y=title_position)
  plt.plot(errors_position, c='r')
  plt.plot(rmse_pose * np.ones((num_error_values, 1)), c='k')
  a2 = fig.add_subplot(2, 1, 2)
  a2.set_title(
      "Red = Absolute Orientation Error [Degrees] - Black = RMSE", y=title_position)
  plt.plot(errors_orientation, c='r')
  plt.plot(rmse_orientation * np.ones((num_error_values, 1)), c='k')
  if plt.get_backend() == 'TkAgg':
    mng = plt.get_current_fig_manager()
    max_size = mng.window.maxsize()
    max_size = (max_size[0], max_size[1] * 0.45)
    mng.resize(*max_size)
  fig.tight_layout()
  plt.subplots_adjust(left=0.025, right=0.975, top=0.8, bottom=0.05)
  plt.show(block=True)
