#!/usr/bin/env python
from matplotlib import pylab as plt
from matplotlib.font_manager import FontProperties
import matplotlib

import argparse
import matplotlib.patches as mpatches
import numpy as np


font = FontProperties()
font.set_size('small')
font.set_family('serif')
font.set_weight('light')
font.set_style('normal')


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


def generate_time_plot(methods, datasets, runtimes_per_method, colors):
  num_methods = len(methods)
  num_datasets = len(datasets)
  x_ticks = np.linspace(0, 1, num_methods)

  width = 0.6 / num_methods / num_datasets
  spacing = 0.4 / num_methods / num_datasets
  line_width = 2
  fig, ax1 = plt.subplots()
  ax1.set_ylabel('Time [s]', color='b')
  ax1.tick_params('y', colors='b')
  ax1.set_yscale('log')
  fig.suptitle("Hand-Eye Calibration Method Timings", fontsize='24')
  handles = []
  for i, dataset in enumerate(datasets):
    runtimes = [runtimes_per_method[dataset][method] for method in methods]
    bp = ax1.boxplot(
        runtimes, 0, '',
        positions=(x_ticks + (i - num_datasets / 2. + 0.5) *
                   spacing * 2),
        widths=width)
    plt.setp(bp['boxes'], color=colors[i], linewidth=line_width)
    plt.setp(bp['whiskers'], color=colors[i], linewidth=line_width)
    plt.setp(bp['fliers'], color=colors[i],
             marker='+', linewidth=line_width)
    plt.setp(bp['medians'], color=colors[i],
             marker='+', linewidth=line_width)
    plt.setp(bp['caps'], color=colors[i], linewidth=line_width)
    handles.append(mpatches.Patch(color=colors[i], label=dataset))
  plt.legend(handles=handles, loc=2)

  plt.xticks(x_ticks, methods)
  plt.xlim(x_ticks[0] - 2.5 * spacing * num_datasets,
           x_ticks[-1] + 2.5 * spacing * num_datasets)

  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--csv_file_names', type=str,
                      required=True, nargs='+',
                      help='CSV file name from which to generate the box plots.')

  args = parser.parse_args()
  print("box_plot.py: Generating box plots from csv file: {}".format(
      args.csv_file_names))
  font = {'family': 'serif',
          'weight': 'light',
          'size': 18,
          'style': 'normal'}

  matplotlib.rc('font', **font)

  font = FontProperties()
  font.set_size('small')
  font.set_family('serif')
  font.set_weight('light')
  font.set_style('normal')
  colors = ['b', 'g', 'c', 'm', 'y', 'k', 'r']
  dt = np.dtype([
      ('algorithm_name', np.str_, 50),
      ('pose_pair_num', np.uint, 1),
      ('iteration_num', np.uint, 1),
      ('prefiltering', np.bool, 1),
      ('poses_B_H_csv_file', np.str_, 50),
      ('poses_W_E_csv_file', np.str_, 50),
      ('success', np.bool, 1),
      ('position_rmse', np.float64, 1),
      ('orientation_rmse', np.float64, 1),
      ('num_inliers', np.uint, 1),
      ('num_input_poses', np.uint, 1),
      ('num_posesafter_filtering', np.uint, 1),
      ('runtime_s', np.float64, 1),
      ('loop_error_position_m', np.float64, 1),
      ('loop_error_orientation_deg', np.float64, 1),
      ('singular_values', np.str_, 300),
      ('bad_singular_values', np.bool, 1),
      ('dataset', np.str_, 50)])

  get_header = True
  print("Evaluating the following result files: {}".format(args.csv_file_names))
  for csv_file_name in args.csv_file_names:
    if get_header:
      header = np.genfromtxt(csv_file_name, dtype=str,
                             max_rows=1, delimiter=',')
      get_header = False
      data = np.genfromtxt(csv_file_name, dtype=dt,
                           skip_header=1, delimiter=',').copy()
    else:
      # print(header)
      body = np.genfromtxt(csv_file_name, dtype=dt,
                           skip_header=1, delimiter=',')
      data = np.append(data, body.copy())

  methods = []
  datasets = []
  position_rmses_per_method = {}
  orientation_rmses_per_method = {}
  position_rmses = []
  orientation_rmses = []
  runtimes = []
  runtimes_per_method = {}

  max_index = 0
  max_index_ds = 0
  for row in data:
    method = row[0]
    b_h_filename = row[4]
    w_e_filename = row[5]
    success = row[6]
    position_rmse = row[7]
    orientation_rmse = row[8]
    runtime = row[12]
    dataset = row[17]

    if not success:
      # TODO(ff): Think of what to do here.
      continue

    if dataset in datasets:
      index_ds = datasets.index(dataset)
    else:
      index_ds = max_index_ds
      datasets.append(dataset)
      max_index_ds += 1
    if method in methods:
      index = methods.index(method)
    else:
      index = max_index
      methods.append(method)
      position_rmses.append([])
      orientation_rmses.append([])
      runtimes.append([])
      max_index += 1
    if dataset not in position_rmses_per_method:
      position_rmses_per_method[dataset] = dict()
    if method not in position_rmses_per_method[dataset]:
      position_rmses_per_method[dataset][method] = list()
    position_rmses_per_method[dataset][method].append(position_rmse)

    if dataset not in orientation_rmses_per_method:
      orientation_rmses_per_method[dataset] = dict()
    if method not in orientation_rmses_per_method[dataset]:
      orientation_rmses_per_method[dataset][method] = list()
    orientation_rmses_per_method[dataset][method].append(orientation_rmse)

    if dataset not in runtimes_per_method:
      runtimes_per_method[dataset] = dict()
    if method not in runtimes_per_method[dataset]:
      runtimes_per_method[dataset][method] = list()
    runtimes_per_method[dataset][method].append(runtime)

    position_rmses[index].append(position_rmse)
    orientation_rmses[index].append(orientation_rmse)
    runtimes[index].append(runtime)
  print("Plotting the results of the follwoing methods: \n\t{}".format(
      ', '.join(methods)))
  print("Creating plots for the following datasets:\n{}".format(datasets))
  # for dataset in datasets:
  #   generate_box_plot(
  #       methods,
  #       [position_rmses_per_method[dataset][method] for method in methods],
  #       [orientation_rmses_per_method[dataset][method] for method in methods])
  generate_time_plot(methods, datasets, runtimes_per_method, colors)
