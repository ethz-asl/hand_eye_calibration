
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import (proj3d, Axes3D)

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


def compute_bbox_3D(poses_list):
  bbox_min = np.zeros((len(poses_list), 3))
  bbox_max = np.zeros((len(poses_list), 3))
  for i in range(0, len(poses_list)):
    poses = poses_list[i]
    bbox_min[i, :] = np.amin(poses[:, 0:3], axis=0)
    bbox_max[i, :] = np.amax(poses[:, 0:3], axis=0)
  return (np.amax(bbox_max, axis=0), np.amin(bbox_min, axis=0))


def plot_poses(poses_list, plot_arrows=True, title="", blocking=True):
  title_position = 1.05
  fig = plt.figure()
  plt.clf()
  ax = Axes3D(fig)

  if title:
    fig.suptitle(title, fontsize='24')

  colors = ['r', 'g', 'b', 'c', 'm', 'k']
  num_colors = len(colors)

  assert len(poses_list) < num_colors, (
      "Need to define more colors to plot more trajectories!")

  (bbox_max, bbox_min) = compute_bbox_3D(poses_list)

  arrow_size = np.linalg.norm(bbox_max - bbox_min) * 0.05
  arrow_width = 2

  axis_min = np.amin(bbox_min)
  axis_max = np.amax(bbox_max)
  ax.set_xlim3d(axis_min, axis_max)
  ax.set_ylim3d(axis_min, axis_max)
  ax.set_zlim3d(axis_min, axis_max)

  for i in range(0, len(poses_list)):
    poses = poses_list[i].copy()

    # Plot line.
    positions = ax.plot(xs=poses[:, 0], ys=poses[:, 1],
                        zs=poses[:, 2], color=colors[i])

    for pose in poses:
      # Position point
      ax.plot([pose[0]], [pose[1]], [pose[2]], 'o',
              markersize=5, color=colors[i], alpha=0.5)
      if not plot_arrows:
        continue

      rotation_quaternion = Quaternion(q=pose[3:7])

      x_rotated = rotation_quaternion.rotate_vector([1, 0, 0, 0])
      x_rotated *= arrow_size
      a = Arrow3D(
          [pose[0], pose[0] + x_rotated[0]
           ], [pose[1], pose[1] + x_rotated[1]],
          [pose[2], pose[2] + x_rotated[2]],
          mutation_scale=20,
          lw=arrow_width,
          arrowstyle="-|>",
          color="r")
      ax.add_artist(a)

      y_rotated = rotation_quaternion.rotate_vector([0, 1, 0, 0])
      y_rotated *= arrow_size
      a = Arrow3D(
          [pose[0], pose[0] + y_rotated[0]
           ], [pose[1], pose[1] + y_rotated[1]],
          [pose[2], pose[2] + y_rotated[2]],
          mutation_scale=20,
          lw=arrow_width,
          arrowstyle="-|>",
          color="g")
      ax.add_artist(a)

      z_rotated = rotation_quaternion.rotate_vector([0, 0, 1, 0])
      z_rotated *= arrow_size
      a = Arrow3D(
          [pose[0], pose[0] + z_rotated[0]
           ], [pose[1], pose[1] + z_rotated[1]],
          [pose[2], pose[2] + z_rotated[2]],
          mutation_scale=20,
          lw=arrow_width,
          arrowstyle="-|>",
          color="b")
      ax.add_artist(a)

  ax.auto_scale_xyz([axis_min, axis_max], [
      axis_min, axis_max], [axis_min, axis_max])
  plt.show(block=blocking)


def plot_alignment_errors(errors_position, rmse_pose, errors_orientation,
                          rmse_orientation, blocking=True):
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
  plt.show(block=blocking)
