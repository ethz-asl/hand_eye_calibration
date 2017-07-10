#!/usr/bin/env python
from matplotlib import pylab as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

import argparse
import bisect
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


def generate_optimization_circle_error_plot(
        min_max_step_time_spoil, min_max_step_translation_spoil,
        min_max_step_angle_spoil, data):
  [loop_errors_position_m, loop_errors_orientation_deg,
   spoiled_initial_guess_angle_offsets,
   spoiled_initial_guess_translation_norm_offsets,
   spoiled_initial_guess_time_offsets] = data

  assert len(loop_errors_position_m) == len(loop_errors_orientation_deg)

  times = np.arange(min_max_step_time_spoil['start'],
                    min_max_step_time_spoil['end'] +
                    min_max_step_time_spoil['step'],
                    min_max_step_time_spoil['step'])
  translation_norms = np.arange(min_max_step_translation_spoil['start'],
                                min_max_step_translation_spoil['end'] +
                                min_max_step_translation_spoil['step'],
                                min_max_step_translation_spoil['step'])
  angles = np.arange(min_max_step_angle_spoil['start'],
                     min_max_step_angle_spoil['end'] +
                     min_max_step_angle_spoil['step'],
                     min_max_step_angle_spoil['step'])

  time_steps = len(times)
  translation_steps = len(translation_norms)
  angle_steps = len(angles)

  error_matrix = np.zeros((translation_steps, angle_steps))
  error_bins_translations = [[[[] for i in range(angle_steps)]
                              for j in range(translation_steps)
                              ] for k in range(time_steps)]

  error_bins_angles = [[[[] for i in range(angle_steps)]
                        for j in range(translation_steps)
                        ] for k in range(time_steps)]

  # Assign all circle errors to bins of times, angles and translation_norms,
  # and calculate the mean.
  for (time_spoil, angle_spoil, translation_spoil, loop_error_position_m,
       loop_error_orientation_deg) in zip(
          spoiled_initial_guess_time_offsets,
          spoiled_initial_guess_angle_offsets,
          spoiled_initial_guess_translation_norm_offsets,
          loop_errors_position_m, loop_errors_orientation_deg):
    t_idx = bisect.bisect_right(times, time_spoil)
    a_idx = bisect.bisect_right(angles, angle_spoil)
    trans_idx = bisect.bisect_right(
        translation_norms, translation_spoil)
    assert t_idx > 0
    assert a_idx > 0, ("angle" + str(angle_spoil))
    assert trans_idx > 0
    assert t_idx < len(times), ("time" + str(time_spoil) + str(times))
    assert a_idx < len(angles), ("angle" + str(angle_spoil) + str(angles))
    assert trans_idx < len(translation_norms), ("trans" +
                                                str(translation_spoil) +
                                                str(translation_norms))

    error_bins_translations[t_idx - 1][trans_idx - 1][a_idx - 1].append(
        loop_error_position_m)
    error_bins_angles[t_idx - 1][trans_idx - 1][a_idx - 1].append(
        loop_error_orientation_deg)

  for t_idx, t in enumerate(times[:-1]):
    error_matrix_translations = np.zeros(
        (len(translation_norms) - 1, len(angles) - 1))
    error_matrix_angles = np.zeros(
        (len(translation_norms) - 1, len(angles) - 1))
    for trans_idx, trans in enumerate(translation_norms[:-1]):
      for a_idx, a in enumerate(angles[:-1]):
        mean_translation = np.mean(
            np.array(error_bins_translations[t_idx][trans_idx][a_idx]))
        error_matrix_translations[trans_idx, a_idx] = mean_translation.copy()
        mean_angles = np.mean(
            np.array(error_bins_angles[t_idx][trans_idx][a_idx]))
        error_matrix_angles[trans_idx, a_idx] = mean_angles.copy()
    fig, axes = plt.subplots(1, 2)
    (ax1, ax2) = axes

    x_tick_labels = (
        '[' + np.char.array(angles[:-1] * 180 / np.pi) + ',' +
        np.char.array(angles[1:] * 180 / np.pi) + ')')
    y_tick_labels = (
        '[' + np.char.array(translation_norms[:-1]) + ',' +
        np.char.array(translation_norms[1:]) + ')')
    x_ticks = range(angle_steps - 1)
    y_ticks = range(translation_steps - 1)
    plt.setp(axes, xticks=x_ticks, xticklabels=x_tick_labels,
             yticks=y_ticks, yticklabels=y_tick_labels)
    spoil_time_frame = '[' + str(t) + ',' + str(times[t_idx + 1]) + ')'

    ax1.set_xlabel('Perturbation Angle [deg]', color='k')
    ax1.set_ylabel('Perturbation Translation Norm [m]', color='k')
    ax1.set_title(
        'Translational error [m] $t_{spoil} \in ' + spoil_time_frame + '$')
    cax1 = ax1.imshow(error_matrix_translations, interpolation='nearest')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(cax1, ax=ax1, cax=cax)
    ax1.spines['left'].set_position(('outward', 10))
    ax1.spines['bottom'].set_position(('outward', 10))
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    ax2.set_xlabel('Perturbation Angle [deg]', color='k')
    ax2.set_ylabel('Perturbation Translation [m]', color='k')
    ax2.set_title(
        'Angular error [deg] $t_{spoil} \in ' + spoil_time_frame + '$')
    cax2 = ax2.imshow(error_matrix_angles, interpolation='nearest')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(cax2, ax=ax2, cax=cax)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_position(('outward', 10))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_visible(False)
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

  min_max_step_times = {
      'start': 0.0,
      'end': 1.0,
      'step': 0.2,
  }

  min_max_step_translation = {
      'start': 0.0,
      'end': 0.1,
      'step': 0.02,
  }
  min_max_step_angle = {
      'start': 0.0,
      'end': 20.0 / 180 * np.pi,
      'step': 5.0 / 180 * np.pi,
  }

  generate_optimization_circle_error_plot(
      min_max_step_times, min_max_step_translation, min_max_step_angle,
      spoiled_data)
