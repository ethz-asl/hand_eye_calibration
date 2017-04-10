# Plotting tools for time alignment.

from matplotlib import pylab as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib

font = FontProperties()
font.set_size('small')
font.set_family('serif')
font.set_weight('light')
font.set_style('normal')


def plot_results(times_A, times_B, signal_A, signal_B,
                 convoluted_signals, time_offset, block=True):

  fig = plt.figure()

  title_position = 1.05

  matplotlib.rcParams.update({'font.size': 20})

  # fig.suptitle("Time Alignment", fontsize='24')
  a1 = plt.subplot(1, 3, 1)

  a1.get_xaxis().get_major_formatter().set_useOffset(False)

  plt.ylabel('angular velocity norm [rad]')
  plt.xlabel('time [s]')
  a1.set_title(
      "Before Time Alignment", y=title_position)
  plt.hold("on")

  min_time = min(np.amin(times_A), np.amin(times_B))
  times_A_zeroed = times_A - min_time
  times_B_zeroed = times_B - min_time

  plt.plot(times_A_zeroed, signal_A, c='r')
  plt.plot(times_B_zeroed, signal_B, c='b')

  times_A_shifted = times_A + time_offset

  a3 = plt.subplot(1, 3, 2)
  a3.get_xaxis().get_major_formatter().set_useOffset(False)
  plt.ylabel('correlation')
  plt.xlabel('sample idx offset')
  a3.set_title(
      "Correlation Result \n[Ideally has a single dominant peak.]",
      y=title_position)
  plt.hold("on")
  plt.plot(np.arange(-len(signal_A) + 1, len(signal_B)), convoluted_signals)

  a2 = plt.subplot(1, 3, 3)
  a2.get_xaxis().get_major_formatter().set_useOffset(False)
  plt.ylabel('angular velocity norm [rad]')
  plt.xlabel('time [s]')
  a2.set_title(
      "After Time Alignment", y=title_position)
  plt.hold("on")
  min_time = min(np.amin(times_A_shifted), np.amin(times_B))
  times_A_shifted_zeroed = times_A_shifted - min_time
  times_B_zeroed = times_B - min_time
  plt.plot(times_A_shifted_zeroed, signal_A, c='r')
  plt.plot(times_B_zeroed, signal_B, c='b')

  plt.subplots_adjust(left=0.04, right=0.99, top=0.8, bottom=0.15)

  if plt.get_backend() == 'TkAgg':
    mng = plt.get_current_fig_manager()
    max_size = mng.window.maxsize()
    max_size = (max_size[0], max_size[1] * 0.45)
    mng.resize(*max_size)
  plt.show(block=block)


def plot_time_stamped_poses(title,
                            time_stamped_poses_A,
                            time_stamped_poses_B,
                            block=True):
  fig = plt.figure()

  title_position = 1.05

  fig.suptitle(title + " [A = top, B = bottom]", fontsize='24')

  a1 = plt.subplot(2, 2, 1)
  a1.set_title(
      "Orientation \nx [red], y [green], z [blue], w [cyan]",
      y=title_position)
  plt.plot(time_stamped_poses_A[:, 4], c='r')
  plt.plot(time_stamped_poses_A[:, 5], c='g')
  plt.plot(time_stamped_poses_A[:, 6], c='b')
  plt.plot(time_stamped_poses_A[:, 7], c='c')

  a2 = plt.subplot(2, 2, 2)
  a2.set_title(
      "Position (eye coordinate frame) \nx [red], y [green], z [blue]", y=title_position)
  plt.plot(time_stamped_poses_A[:, 1], c='r')
  plt.plot(time_stamped_poses_A[:, 2], c='g')
  plt.plot(time_stamped_poses_A[:, 3], c='b')

  a3 = plt.subplot(2, 2, 3)
  plt.plot(time_stamped_poses_B[:, 4], c='r')
  plt.plot(time_stamped_poses_B[:, 5], c='g')
  plt.plot(time_stamped_poses_B[:, 6], c='b')
  plt.plot(time_stamped_poses_B[:, 7], c='c')

  a4 = plt.subplot(2, 2, 4)
  plt.plot(time_stamped_poses_B[:, 1], c='r')
  plt.plot(time_stamped_poses_B[:, 2], c='g')
  plt.plot(time_stamped_poses_B[:, 3], c='b')

  plt.subplots_adjust(left=0.025, right=0.975, top=0.8, bottom=0.05)

  if plt.get_backend() == 'TkAgg':
    mng = plt.get_current_fig_manager()
    max_size = mng.window.maxsize()
    max_size = (max_size[0], max_size[1] * 0.45)
    mng.resize(*max_size)
  plt.show(block=block)


def plot_angular_velocities(title,
                            angular_velocities,
                            angular_velocities_filtered,
                            block=True):
  fig = plt.figure()

  title_position = 1.05

  fig.suptitle(title, fontsize='24')

  a1 = plt.subplot(1, 2, 1)
  a1.set_title(
      "Angular Velocities Before Filtering \nvx [red], vy [green], vz [blue]",
      y=title_position)
  plt.plot(angular_velocities[:, 0], c='r')
  plt.plot(angular_velocities[:, 1], c='g')
  plt.plot(angular_velocities[:, 2], c='b')

  a2 = plt.subplot(1, 2, 2)
  a2.set_title(
      "Angular Velocities After Filtering \nvx [red], vy [green], vz [blue]", y=title_position)
  plt.plot(angular_velocities_filtered[:, 0], c='r')
  plt.plot(angular_velocities_filtered[:, 1], c='g')
  plt.plot(angular_velocities_filtered[:, 2], c='b')

  plt.subplots_adjust(left=0.025, right=0.975, top=0.8, bottom=0.05)

  if plt.get_backend() == 'TkAgg':
    mng = plt.get_current_fig_manager()
    max_size = mng.window.maxsize()
    max_size = (max_size[0], max_size[1] * 0.45)
    mng.resize(*max_size)
  plt.show(block=block)


def plot_input_data(quaternions_A,
                    quaternions_B,
                    quaternions_A_interp,
                    quaternions_B_interp,
                    angular_velocity_norms_A,
                    angular_velocity_norms_B,
                    angular_velocity_norms_A_filtered,
                    angular_velocity_norms_B_filtered,
                    block=True):
  quat_A = np.array([quat.q for quat in quaternions_A])
  quat_A_interp = np.array([quat.q for quat in quaternions_A_interp])
  quat_B = np.array([quat.q for quat in quaternions_B])
  quat_B_interp = np.array([quat.q for quat in quaternions_B_interp])

  fig = plt.figure()

  title_position = 1.05

  fig.suptitle("Input Data [A = top, B = bottom]", fontsize='24')
  a1 = plt.subplot(2, 4, 1)
  a1.set_title(
      "Input quaternions \nx [red], y [green], z [blue], w [cyan]",
      y=title_position)
  plt.plot(quat_A[:, 0], c='r')
  plt.plot(quat_A[:, 1], c='g')
  plt.plot(quat_A[:, 2], c='b')
  plt.plot(quat_A[:, 3], c='c')
  a2 = plt.subplot(2, 4, 2)
  a2.set_title(
      "Interpolated quaternions \nx [red], y [green], z [blue], w [cyan]",
      y=title_position)
  plt.plot(quat_A_interp[:, 0], c='r')
  plt.plot(quat_A_interp[:, 1], c='g')
  plt.plot(quat_A_interp[:, 2], c='b')
  plt.plot(quat_A_interp[:, 3], c='c')
  a3 = plt.subplot(2, 4, 3)
  a3.set_title("Norm of angular velocity", y=title_position)
  plt.plot(angular_velocity_norms_A, c='r')
  a4 = plt.subplot(2, 4, 4)
  a4.set_title(
      "Norm of angular velocity \n[median filter]", y=title_position)
  plt.plot(angular_velocity_norms_A_filtered, c='r')

  plt.subplot(2, 4, 5)
  plt.plot(quat_B[:, 0], c='r')
  plt.plot(quat_B[:, 1], c='g')
  plt.plot(quat_B[:, 2], c='b')
  plt.plot(quat_B[:, 3], c='c')
  plt.subplot(2, 4, 6)
  plt.plot(quat_B_interp[:, 0], c='r')
  plt.plot(quat_B_interp[:, 1], c='g')
  plt.plot(quat_B_interp[:, 2], c='b')
  plt.plot(quat_B_interp[:, 3], c='c')
  plt.subplot(2, 4, 7)
  plt.plot(angular_velocity_norms_B, c='b')
  plt.subplot(2, 4, 8)
  plt.plot(angular_velocity_norms_B_filtered, c='b')

  plt.subplots_adjust(left=0.025, right=0.975, top=0.8, bottom=0.05)

  if plt.get_backend() == 'TkAgg':
    mng = plt.get_current_fig_manager()
    max_size = mng.window.maxsize()
    max_size = (max_size[0], max_size[1] * 0.45)
    mng.resize(*max_size)
  plt.show(block=block)
