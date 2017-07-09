#!/usr/bin/env python
from matplotlib import pylab as plt
from matplotlib.font_manager import FontProperties
import matplotlib

import argparse
import matplotlib.patches as mpatches
import numpy as np

from hand_eye_calibration_experiments.experiment_plotting_tools import (
    collect_data_from_csv)


font = FontProperties()
font.set_size('small')
font.set_family('serif')
font.set_weight('light')
font.set_style('normal')
line_width = 2


def generate_box_plot(dataset, methods, position_rmses, orientation_rmses):

  num_methods = len(methods)
  x_ticks = np.linspace(0., 1., num_methods)

  width = 0.3 / num_methods
  spacing = 0.3 / num_methods
  fig, ax1 = plt.subplots()
  ax1.set_ylabel('RMSE position [m]', color='b')
  ax1.tick_params('y', colors='b')
  fig.suptitle(
      "Hand-Eye Calibration Method Error {}".format(dataset), fontsize='24')
  bp_position = ax1.boxplot(position_rmses, 0, '',
                            positions=x_ticks - spacing, widths=width)
  plt.setp(bp_position['boxes'], color='blue', linewidth=line_width)
  plt.setp(bp_position['whiskers'], color='blue', linewidth=line_width)
  plt.setp(bp_position['fliers'], color='blue',
           marker='+', linewidth=line_width)
  plt.setp(bp_position['caps'], color='blue', linewidth=line_width)
  plt.setp(bp_position['medians'], color='blue', linewidth=line_width)
  ax2 = ax1.twinx()
  ax2.set_ylabel('RMSE Orientation [$^\circ$]', color='g')
  ax2.tick_params('y', colors='g')
  bp_orientation = ax2.boxplot(
      orientation_rmses, 0, '', positions=x_ticks + spacing, widths=width)
  plt.setp(bp_orientation['boxes'], color='green', linewidth=line_width)
  plt.setp(bp_orientation['whiskers'], color='green', linewidth=line_width)
  plt.setp(bp_orientation['fliers'], color='green',
           marker='+')
  plt.setp(bp_orientation['caps'], color='green', linewidth=line_width)
  plt.setp(bp_orientation['medians'], color='green', linewidth=line_width)

  plt.xticks(x_ticks, methods)
  plt.xlim(x_ticks[0] - 2.5 * spacing, x_ticks[-1] + 2.5 * spacing)

  plt.show()


def generate_time_plot(methods, datasets, runtimes_per_method, colors):
  num_methods = len(methods)
  num_datasets = len(datasets)
  x_ticks = np.linspace(0., 1., num_methods)

  width = 0.6 / num_methods / num_datasets
  spacing = 0.4 / num_methods / num_datasets
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
  [methods, datasets, position_rmses_per_method, orientation_rmses_per_method,
   position_rmses, orientation_rmses, runtimes, runtimes_per_method,
   spoiled_data] = collect_data_from_csv(args.csv_file_names)
  print("Plotting the results of the follwoing methods: \n\t{}".format(
      ', '.join(methods)))
  print("Creating plots for the following datasets:\n{}".format(datasets))
  for dataset in datasets:
    generate_box_plot(
        dataset,
        methods,
        [position_rmses_per_method[dataset][method] for method in methods],
        [orientation_rmses_per_method[dataset][method] for method in methods])
  generate_time_plot(methods, datasets, runtimes_per_method, colors)
