#!/usr/bin/env python
import unittest

from hand_eye_calibration.time_alignment import (
    calculate_time_offset_from_signals)
from hand_eye_calibration.quaternion import (
    Quaternion, quaternions_interpolate, angular_velocity_between_quaternions)
from hand_eye_calibration.test_tools import (
    introduce_data_drops, DataDropConfig)

from scipy import signal
import numpy as np
import numpy.testing as npt
import math
import matplotlib.pyplot as plt
import copy


def plot_alignment(data_no_drops, data_with_drops, blocking=False):
  plt.subplot(2, 1, 1)
  plt.plot(data_no_drops)
  plt.subplot(2, 1, 2)
  plt.plot(data_with_drops)
  plt.show(block=blocking)
  plt.close()


class TimeAlignment(unittest.TestCase):
  make_plots_blocking = False
  n_samples = 100.
  q1_initial = Quaternion(0, 0, 0, 1)
  q1_final = Quaternion(np.sqrt(2.) / 2., 0, 0, np.sqrt(2.) / 2.)
  ts = np.linspace(0, (n_samples + 1) / n_samples, n_samples)
  t1s = ts

  q2_initial = Quaternion(np.sqrt(2.) / 2., 0, 0, np.sqrt(2.) / 2.)
  q2_final = Quaternion(1., 0, 0, 0.)
  t2s = ts + 1.5 + signal.gaussian(len(ts), 0.1)

  # To generate varying angular velocities, we assign new random times
  # between the different quaternions.
  quat_interpolate_times = np.random.rand(n_samples)
  quat_interpolate_times.sort()
  quat_interpolate_times2 = (
      quat_interpolate_times *
      (quat_interpolate_times[-1] - quat_interpolate_times[0]) + t2s[0])
  q1s = quaternions_interpolate(
      q1_initial, t1s[0], q1_final, t1s[-1], quat_interpolate_times)
  q2s = quaternions_interpolate(
      q2_initial, t2s[0], q2_final, t2s[-1], quat_interpolate_times2)

  # ts_betwenn_quaternions = ts
  angular_velocity1_norms = []
  for i in range(len(q1s) - 1):
    angular_velocity = angular_velocity_between_quaternions(
        q1s[i], q1s[i + 1],
        t1s[i + 1] - t1s[i])
    angular_velocity1_norms.append(np.linalg.norm(angular_velocity))
  angular_velocity2_norms = []
  for i in range(len(q2s) - 1):
    angular_velocity = angular_velocity_between_quaternions(
        q2s[i], q2s[i + 1],
        t2s[i + 1] - t2s[i])
    angular_velocity2_norms.append(np.linalg.norm(angular_velocity))
  angular_velocity1_norms += signal.gaussian(len(angular_velocity1_norms), 1)

  dx = np.mean(np.diff(t1s))

  def test_time_alignment(self):
    time_offset = calculate_time_offset_from_signals(
        self.t1s[0:-1], self.angular_velocity1_norms, self.t2s[0:-1],
        self.angular_velocity2_norms, plot=True, block=False)
    print(time_offset)
    # TODO(ff): Finish this test.

  def test_time_alignment_from_sample_csv_poses(self):
    # TODO(ff): Read stamped poses from csv files.
    # Then call calculate_time_offset.
    pass

  def test_introduce_data_drops(self):
    test_size = 1000
    test = [math.sin(float(x)) for x in np.linspace(0, 2 * math.pi, test_size)]

    test_before = copy.deepcopy(test)

    config = DataDropConfig()
    config.max_percentage_for_single_drop = 5.0
    config.overall_drop_percentage = 20.0

    print("test size before dropping: {}".format(len(test)))
    set_to_none = True
    introduce_data_drops(test, config, set_to_none)
    print("test size after dropping: {}".format(len(test)))

    expected_test_size = float(test_size) - \
        ((config.overall_drop_percentage / 100.0) * float(test_size))

    print("expected_test_size: {}".format(expected_test_size))
    # assert abs(len(test) - expected_test_size) < 1e-8
    plot_alignment(test_before, test, blocking=self.make_plots_blocking)

    def test_introduce_data_drops_with_time_alignment(self):

      angular_velocity1_norms_before = copy.deepcopy(angular_velocity1_norms)
      angular_velocity2_norms_before = copy.deepcopy(angular_velocity2_norms)

      time_offset = calculate_time_offset_from_signals(
          self.t1s[0:-1], self.angular_velocity1_norms, self.t2s[0:-1],
          self.angular_velocity2_norms, plot=True, block=True)
      print(time_offset)

      config = DataDropConfig()
      config.max_percentage_for_single_drop = 5.0
      config.overall_drop_percentage = 20.0

      set_to_none = False
      introduce_data_drops(test, config, set_to_none)

      expected_test_size = float(test_size) - \
          ((config.overall_drop_percentage / 100.0) * float(test_size))

      print("expected_test_size: {}".format(expected_test_size))

      # assert abs(len(test) - expected_test_size) < 1e-8
      plot_alignment(test_before, test, blocking=self.make_plots_blocking)


if __name__ == '__main__':
  unittest.main()
