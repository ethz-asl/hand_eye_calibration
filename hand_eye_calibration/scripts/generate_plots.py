#!/usr/bin/env python
from matplotlib import pylab as plt

import argparse
import numpy as np


def generate_box_plot(methods, position_rmses, orientation_rmses):

  num_methods = len(methods)
  x_ticks = np.linspace(0, 1, num_methods)

  width = 0.3 / num_methods
  spacing = 0.3 / num_methods
  fig, ax1 = plt.subplots()
  ax1.set_ylabel('RMSE position [m]', color='b')
  ax1.tick_params('y', colors='b')
  fig.suptitle("Hand-Eye Calibration Method Comparison", fontsize='24')
  ax1.boxplot(position_rmses, 0, '', positions=x_ticks - spacing, widths=width)
  ax2 = ax1.twinx()
  ax2.set_ylabel('RMSE Orientation [$^\circ$]', color='g')
  ax2.tick_params('y', colors='g')
  bp_orientation = ax2.boxplot(
      orientation_rmses, 0, '', positions=x_ticks + spacing, widths=width)
  plt.setp(bp_orientation['boxes'], color='green')
  plt.setp(bp_orientation['whiskers'], color='green')
  plt.setp(bp_orientation['fliers'], color='red', marker='+')

  plt.xticks(x_ticks, methods)
  plt.xlim(x_ticks[0] - 2.5 * spacing, x_ticks[-1] + 2.5 * spacing)

  plt.show()


def generate_time_plot(methods, datasets, runtimes):
  # TODO(ff):
  # - Collect the runtimes of all runs.

  num_methods = len(methods)
  num_datasets = len(datasets)
  x_ticks = np.linspace(0, 1, num_methods)

  width = 0.6 / num_methods / num_datasets
  spacing = 0.6 / num_methods / num_datasets
  fig, ax1 = plt.subplots()
  ax1.set_ylabel('RMSE position [m]', color='b')
  ax1.tick_params('y', colors='b')
  fig.suptitle("Hand-Eye Calibration Method Timings", fontsize='24')
  for i, runtimes in enumerate(datasets):
    ax1.boxplot(runtime, 0, '', positions=x_ticks - spacing, widths=width)

  plt.xticks(x_ticks, methods)
  plt.xlim(x_ticks[0] - 2.5 * spacing, x_ticks[-1] + 2.5 * spacing)

  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--csv_file_names', type=str,
                      required=True, nargs='+',
                      help='CSV file name from which to generate the box plots.')

  args = parser.parse_args()
  print("box_plot.py: Generating box plots from csv file: {}".format(
      args.csv_file_names))
  get_header = False
  data = np.empty(0, dtype=str)
  for csv_file_name in args.csv_file_names:
    if get_header:
      header = np.genfromtxt(csv_file_name, dtype=str,
                             max_rows=1, delimiter=',')
      get_header = True
    body = np.genfromtxt(csv_file_name, dtype=None,
                         skip_header=1, delimiter=',')
    np.append(data, body)

  print(header)
  print(data)
  assert(False)
  methods = []
  datasets = []
  position_rmses = []
  orientation_rmses = []
  runtimes = []
  max_index = 0
  for row in data:
    method = row[0]
    b_h_filename = row[4]
    w_e_filename = row[5]
    success = row[6]
    position_rmse = row[7]
    orientation_rmse = row[8]
    runtime = row[12]
    if not success:
      # TODO(ff): Think of what to do here.
      continue
    if method in methods:
      index = methods.index(method)
    else:
      methods.append(method)
      index = max_index
      position_rmses.append([])
      orientation_rmses.append([])
      runtimes.append([])
      max_index += 1
    position_rmses[index].append(position_rmse)
    orientation_rmses[index].append(orientation_rmse)
    runtimes[index].append(runtime)
  print("Plotting the results of the follwoing methods: \n\t{}".format(
      ', '.join(methods)))

  generate_box_plot(methods, position_rmses, orientation_rmses)
