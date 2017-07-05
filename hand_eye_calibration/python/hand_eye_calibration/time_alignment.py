# Expect time stamped transformations
# Output aligned and resampled poses

from quaternion import (angular_velocity_between_quaternions,
                        quaternions_interpolate, Quaternion)
from time_alignment_plotting_tools import (plot_results, plot_input_data,
                                           plot_time_stamped_poses,
                                           plot_angular_velocities)
from scipy import signal
import numpy as np
import bisect


class FilteringConfig:

  def __init__(self):
    self.smoothing_kernel_size_A = 25
    self.clipping_percentile_A = 99.5
    self.smoothing_kernel_size_B = 25
    self.clipping_percentile_B = 99.0


def filter_and_smooth_angular_velocity(angular_velocity,
                                       low_pass_kernel_size, clip_percentile, plot=False):
  """Reduce the noise in a velocity signal."""

  max_value = np.percentile(angular_velocity, clip_percentile)
  print("Clipping angular velocity norms to {} rad/s ...".format(max_value))
  angular_velocity_clipped = np.clip(angular_velocity, -max_value, max_value)
  print("Done clipping angular velocity norms...")

  low_pass_kernel = np.ones((low_pass_kernel_size, 1)) / low_pass_kernel_size
  print("Smoothing with kernel size {} samples...".format(low_pass_kernel_size))

  angular_velocity_smoothed = signal.correlate(angular_velocity_clipped,
                                               low_pass_kernel, 'same')

  print("Done smoothing angular velocity norms...")

  if plot:
    plot_angular_velocities("Angular Velocities", angular_velocity,
                            angular_velocity_smoothed, True)

  return angular_velocity_smoothed.copy()


def calculate_time_offset_from_signals(times_A, signal_A,
                                       times_B, signal_B,
                                       plot=False, block=True):
  """ Calculates the time offset between signal A and signal B. """
  convoluted_signals = signal.correlate(signal_B, signal_A)
  dt_A = np.mean(np.diff(times_A))
  offset_indices = np.arange(-len(signal_A) + 1, len(signal_B))
  max_index = np.argmax(convoluted_signals)
  offset_index = offset_indices[max_index]
  time_offset = dt_A * offset_index + times_B[0] - times_A[0]
  if plot:
    plot_results(times_A, times_B, signal_A, signal_B, convoluted_signals,
                 time_offset, block=block)
  return time_offset


def resample_quaternions_from_samples(times, quaternions, samples):
  """
  Resample the quaternions at the times specified in 'samples'.
  Uses SLERP for quaternion interpolation.
  """
  interp_quaternions = []
  for sample in samples:
    assert sample <= times[-1], (sample, times)
    assert sample >= times[0], (sample, times[0])

    right_idx = bisect.bisect_left(times, sample)
    if (np.isclose(sample, times[right_idx], atol=1e-16)):
      interp_quaternions.append(quaternions[right_idx])
    else:
      left_idx = right_idx - 1
      assert right_idx < times.shape[0], end_idx
      assert left_idx >= 0, left_idx
      sample_times = []
      sample_times.append(sample)
      quaternion_interp = quaternions_interpolate(
          quaternions[left_idx], times[left_idx], quaternions[right_idx],
          times[right_idx], sample_times)
      interp_quaternions += quaternion_interp
  assert len(interp_quaternions) == samples.shape[0], str(
      len(interp_quaternions)) + " vs " + str(samples.shape[0])
  return interp_quaternions


def resample_quaternions(times, quaternions, dt):
  """
  Resample the quaternions based on the new interval dt within the interval
  spanned by the first and last time stamp in 'times'.
  Uses SLERP for quaternion interpolation.
  """
  interval = times[-1] - times[0]
  samples = np.linspace(times[0], times[-1], interval / dt + 1)
  return (resample_quaternions_from_samples(times, quaternions, samples),
          samples)


def compute_angular_velocity_norms(quaternions, samples, smoothing_kernel_size, clipping_percentile, plot=False):
  angular_velocity_norms = []
  angular_velocity_size = (len(quaternions) - 1)
  angular_velocity = np.zeros((angular_velocity_size, 3))
  for i in range(0, angular_velocity_size):
    dt = samples[i + 1] - samples[i]
    assert dt > 0.
    angular_velocity[i, :] = angular_velocity_between_quaternions(
        quaternions[i], quaternions[i + 1], dt)

  angular_velocity_filtered = filter_and_smooth_angular_velocity(
      angular_velocity, smoothing_kernel_size, clipping_percentile, plot)

  for i in range(0, angular_velocity_size):
    angular_velocity_norms.append(
        np.linalg.norm(angular_velocity_filtered[i, :]))

  assert len(angular_velocity_norms) == (len(quaternions) - 1)
  return angular_velocity_norms


def calculate_time_offset(times_A, quaternions_A, times_B, quaternions_B, filtering_config, plot=False):
  """
  Calculate the time offset between the stamped quaternions_A and quaternions_B.

  We generate fake angular rotations by taking the derivatives of the
  quaternions. From these derivatives we take the norm and then we apply a
  convolution to compute the best time alignment for the two sets of poses.
  Note that the accuracy of the time alignment is limited by the higher frequency
  of the two signals, i.e. by the smallest time interval between two poses.
  """
  time_offset = 0.0

  # Get the two mean time steps. Take the smaller one for the interpolation.
  dt_A = np.mean(np.diff(times_A))
  dt_B = np.mean(np.diff(times_B))
  if dt_A >= dt_B:
    dt = dt_A
  else:
    dt = dt_B

  # Using the smaller mean time step resample the poses inbetween
  # measurements.
  (quaternions_A_interp, samples_A) = resample_quaternions(times_A,
                                                           quaternions_A, dt)
  (quaternions_B_interp, samples_B) = resample_quaternions(times_B,
                                                           quaternions_B, dt)

  # Compute angular velocity norms for the resampled orientations.
  angular_velocity_norms_A = compute_angular_velocity_norms(
      quaternions_A_interp, samples_A,
      filtering_config.smoothing_kernel_size_A,
      filtering_config.clipping_percentile_A, plot)
  angular_velocity_norms_B = compute_angular_velocity_norms(
      quaternions_B_interp, samples_B,
      filtering_config.smoothing_kernel_size_B,
      filtering_config.clipping_percentile_B, plot)

  # Adapt samples to filtered data.
  length_before = len(samples_A)
  samples_A = samples_A[:len(angular_velocity_norms_A)]
  samples_B = samples_B[:len(angular_velocity_norms_B)]
  assert len(samples_A) == length_before - 1

  # Plot the intput as it goes through the interpolation and filtering.
  if plot:
    plot_input_data(quaternions_A, quaternions_B, quaternions_A_interp,
                    quaternions_B_interp, angular_velocity_norms_A,
                    angular_velocity_norms_B,
                    angular_velocity_norms_A,
                    angular_velocity_norms_B, False)

  # Comput time offset.
  time_offset = calculate_time_offset_from_signals(
      samples_A, angular_velocity_norms_A, samples_B,
      angular_velocity_norms_B, plot, True)

  return time_offset


def interpolate_poses_from_samples(time_stamped_poses, samples):
  """
  Interpolate time stamped poses at the time stamps provided in samples.
  The poses are expected in the following format:
    [timestamp [s], x [m], y [m], z [m], qx, qy, qz, qw]

  We apply linear interpolation to the position and use SLERP for
  the quaternion interpolation.
  """

  # Extract the quaternions from the poses.
  quaternions = []
  for pose in time_stamped_poses[:, 1:]:
    quaternions.append(Quaternion(q=pose[3:]))

  # interpolate the quaternions.
  quaternions_interp = resample_quaternions_from_samples(
      time_stamped_poses[:, 0], quaternions, samples)

  # Interpolate the position and assemble the full aligned pose vector.
  num_poses = samples.shape[0]
  aligned_poses = np.zeros((num_poses, time_stamped_poses.shape[1]))
  aligned_poses[:, 0] = samples[:]

  for i in [1, 2, 3]:
    aligned_poses[:, i] = np.interp(
        samples,
        np.asarray(time_stamped_poses[:, 0]).ravel(),
        np.asarray(time_stamped_poses[:, i]).ravel())

  for i in range(0, num_poses):
    aligned_poses[i, 4:8] = quaternions_interp[i].q

  return aligned_poses.copy()


def compute_aligned_poses(time_stamped_poses_A,
                          time_stamped_poses_B,
                          time_offset,
                          plot=False):
  """
  time_stamped_poses should have the following format:
    [timestamp [s], x [m], y [m], z [m], qx, qy, qz, qw]
  """

  time_stamped_poses_A_shifted = time_stamped_poses_A.copy()

  # Apply time offset.
  time_stamped_poses_A_shifted[:, 0] += time_offset

  # Compute common time interval.
  start_time = max(
      time_stamped_poses_A_shifted[0, 0], time_stamped_poses_B[0, 0])
  end_time = min(
      time_stamped_poses_A_shifted[-1, 0], time_stamped_poses_B[-1, 0])
  interval = end_time - start_time

  # Resample at the lower frequency to prevent introducing more noise.
  dt_A = np.mean(np.diff(time_stamped_poses_A_shifted[:, 0]))
  dt_B = np.mean(np.diff(time_stamped_poses_B[:, 0]))
  if dt_A >= dt_B:
    dt = dt_A
    timestamps_low = time_stamped_poses_A_shifted[:, 0].T
    timestamps_high = time_stamped_poses_B[:, 0].T
  else:
    dt = dt_B
    timestamps_low = time_stamped_poses_B[:, 0].T
    timestamps_high = time_stamped_poses_A_shifted[:, 0].T

  # Create samples at time stamps from lower frequency signal, check if there
  # are timely close samples available from the other signal.
  # Interpolate the signals to match the time stamps of the low frequency
  # signal.
  samples = []
  max_time_stamp_difference = 0.1
  for timestamp in timestamps_low:
    if (timestamp < start_time):
      continue
    idx = bisect.bisect_left(timestamps_high, timestamp)
    if idx >= timestamps_high.shape[0] - 1:
      print("Omitting timestamps at the end of the high frequency poses.")
      break
    # Check if times are identical.
    if timestamp == timestamps_high[idx]:
      samples.append(timestamp)
      continue
    left_idx = idx - 1

    if timestamp < timestamps_high[left_idx]:
      continue

    right_idx = idx
    assert right_idx < timestamps_high.shape[
        0], (right_idx, timestamps_high.shape[0])
    if ((timestamp - timestamps_high[left_idx]) < max_time_stamp_difference and
            (timestamps_high[right_idx] - timestamp) <= max_time_stamp_difference):
      samples.append(timestamp)
      continue

  samples = np.array(samples)

  # Uncomment if you want to have equally spaced samples in time.
  # samples = np.linspace(start_time, end_time, interval / dt + 1)

  aligned_poses_A = interpolate_poses_from_samples(time_stamped_poses_A_shifted,
                                                   samples)
  aligned_poses_B = interpolate_poses_from_samples(time_stamped_poses_B,
                                                   samples)
  assert aligned_poses_A.shape == aligned_poses_B.shape
  assert np.allclose(aligned_poses_A[:, 0], aligned_poses_B[:, 0], atol=1e-8)
  print("Found {} matching poses.".format(aligned_poses_A.shape[0]))

  if plot:
    plot_time_stamped_poses("Time Aligned Aligned Poses", aligned_poses_A,
                            aligned_poses_B)

  return (aligned_poses_A, aligned_poses_B)
