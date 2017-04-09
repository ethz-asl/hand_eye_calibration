#!/usr/bin/env python
from matplotlib import pylab as plt

import argparse
import numpy as np

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--csv_file_name', required=True,
                      help='CSV file name from which to generate the box plots.')

  args = parser.parse_args()
  print("box_plot.py: Generating box plots from csv file: {}".format(
      args.csv_file_name))

  header = np.genfromtxt(args.csv_file_name, dtype=str,
                         max_rows=1, delimiter=',')
  data = np.genfromtxt(args.csv_file_name, dtype=None,
                       skip_header=1, delimiter=',')

  # print(header)
  # print(data)

  methods = []
  position_rmses = []
  orientation_rmses = []
  max_index = 0
  for row in data:
    method = row[0]
    success = row[6]
    position_rmse = row[7]
    orientation_rmse = row[8]
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
      max_index += 1
    position_rmses[index].append(position_rmse)
    orientation_rmses[index].append(orientation_rmse)
  print("Plotting the results of the follwoing methods: \n\t{}".format(
      ', '.join(methods)))

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
