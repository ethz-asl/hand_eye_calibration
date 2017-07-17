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
        min_max_step_angle_spoil, data, plot_order, num_measurements_per_bin):
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

  # Fine tune the plot here.
  # cap_translations = True
  # cap_angles = True
  cap_translations = False
  cap_angles = False
  translation_cap_value = 1.0
  angle_cap_value = 10.0

  if plot_order == "translation_time_angle":
    loop_x = angles
    loop_y = times
    loop_plot = translation_norms
    x_spoils = spoiled_initial_guess_angle_offsets
    y_spoils = spoiled_initial_guess_time_offsets
    plot_spoils = spoiled_initial_guess_translation_norm_offsets
    perturbation_x_label = "Angle [deg]"
    perturbation_y_label = "Time [s]"
    perturbation_loop_label = "$|x_{spoil}|"

  elif plot_order == "angle_time_translation":
    loop_x = translation_norms
    loop_y = times
    loop_plot = angles
    x_spoils = spoiled_initial_guess_translation_norm_offsets
    y_spoils = spoiled_initial_guess_time_offsets
    plot_spoils = spoiled_initial_guess_angle_offsets
    perturbation_x_label = "Translation [m]"
    perturbation_y_label = "Time [s]"
    perturbation_loop_label = "$\\alpha_{spoil}"

  elif plot_order == "time_translation_angle":
    loop_x = angles
    loop_y = translation_norms
    loop_plot = times
    x_spoils = spoiled_initial_guess_angle_offsets
    y_spoils = spoiled_initial_guess_translation_norm_offsets
    plot_spoils = spoiled_initial_guess_time_offsets
    perturbation_x_label = "Angle [deg]"
    perturbation_y_label = "Translation [m]"
    perturbation_loop_label = "$t_{spoil}"

  x_steps = len(loop_x)
  y_steps = len(loop_y)
  plot_steps = len(loop_plot)

  error_bins_translations = [[[[] for i in range(x_steps)]
                              for j in range(y_steps)
                              ] for k in range(plot_steps)]

  error_bins_angles = [[[[] for i in range(x_steps)]
                        for j in range(y_steps)
                        ] for k in range(plot_steps)]

  # Assign all circle errors to bins of times, angles and translation_norms,
  # and calculate the mean.
  for (x_spoil, y_spoil, plot_spoil, loop_error_position_m,
       loop_error_orientation_deg) in zip(
          x_spoils,
          y_spoils,
          plot_spoils,
          loop_errors_position_m, loop_errors_orientation_deg):

    x_idx = bisect.bisect_right(loop_x, np.linalg.norm(x_spoil))
    y_idx = bisect.bisect_right(loop_y, np.linalg.norm(y_spoil))
    plot_idx = bisect.bisect_right(loop_plot, np.linalg.norm(plot_spoil))

    assert x_idx > 0
    assert y_idx > 0, ("y " + str(y_spoil))
    assert plot_idx > 0
    assert x_idx < x_steps, ("x " + str(x_spoil) + str(loop_x))
    assert y_idx < y_steps, ("y " + str(y_spoil) + str(loop_y))
    assert plot_idx < plot_steps, ("plot idx: " + str(plot_idx) +
                                   ", plot_spoil: " + str(plot_spoil) +
                                   ", plot range: " + str(loop_plot))
    error_bins_translations[plot_idx - 1][y_idx - 1][x_idx - 1].append(
        loop_error_position_m)
    error_bins_angles[plot_idx - 1][y_idx - 1][x_idx - 1].append(
        loop_error_orientation_deg)

  for plot_idx, plot_value in enumerate(loop_plot[:-1]):
    error_matrix_translations = np.zeros(
        (y_steps - 1, x_steps - 1))
    error_matrix_angles = np.zeros(
        (y_steps - 1, x_steps - 1))
    for y_idx, y in enumerate(loop_y[:-1]):
      for x_idx, x in enumerate(loop_x[:-1]):
        num_measurements_per_bin_translation = len(
            error_bins_translations[plot_idx][y_idx][x_idx])
        assert num_measurements_per_bin_translation == num_measurements_per_bin, (
            "len: " + str(num_measurements_per_bin_translation))
        num_measurements_per_bin_angles = len(
            error_bins_angles[plot_idx][y_idx][x_idx])
        assert num_measurements_per_bin_angles == num_measurements_per_bin, (
            "len: " + str(num_measurements_per_bin_angles))
        # mean_translation = np.max(
        mean_translation = np.mean(
            np.array(error_bins_translations[plot_idx][y_idx][x_idx]))
        if cap_translations:
          mean_translation = min(mean_translation, translation_cap_value)
        error_matrix_translations[y_idx, x_idx] = mean_translation
        # mean_angles = np.max(
        mean_angles = np.mean(
            np.array(error_bins_angles[plot_idx][y_idx][x_idx]))
        if cap_angles:
          mean_angles = min(mean_angles, angle_cap_value)
        error_matrix_angles[y_idx, x_idx] = mean_angles
    fig, axes = plt.subplots(1, 2)
    (ax1, ax2) = axes
    if (plot_order == "translation_time_angle" or
            plot_order == "time_translation_angle"):
      x_ticks_start = np.rint(loop_x[:-1] * 180 / np.pi).astype(int)
      x_ticks_end = np.rint(loop_x[1:] * 180 / np.pi).astype(int)
      x_tick_labels = (
          '[' + np.char.array(x_ticks_start) + ',' +
          np.char.array(x_ticks_end) + ')')
    else:
      x_tick_labels = (
          '[' + np.char.array(loop_x[:-1]) + ',' +
          np.char.array(loop_x[1:]) + ')')
    y_tick_labels = (
        '[' + np.char.array(loop_y[:-1]) + ',' +
        np.char.array(loop_y[1:]) + ')')
    x_ticks = range(x_steps - 1)
    y_ticks = range(y_steps - 1)
    plt.setp(axes, xticks=x_ticks, xticklabels=x_tick_labels,
             yticks=y_ticks, yticklabels=y_tick_labels)
    if plot_order == "angle_time_translation":
      spoil_plot_frame = ('[' +
                          str(plot_value * 180 / np.pi) + '^\circ,' +
                          str(loop_plot[plot_idx + 1] * 180 / np.pi) + '^\circ)')
    else:
      spoil_plot_frame = '[' + \
          str(plot_value) + ',' + str(loop_plot[plot_idx + 1]) + ')'

    ax1.set_xlabel('Perturbation ' + perturbation_x_label, color='k')
    ax1.set_ylabel('Perturbation ' + perturbation_y_label, color='k')
    ax1.set_title('Translational error [m], ' + perturbation_loop_label +
                  ' \in ' + spoil_plot_frame + '$')
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

    ax2.set_xlabel('Perturbation ' + perturbation_x_label, color='k')
    ax2.set_title('Angular error [deg], ' + perturbation_loop_label + ' \in ' +
                  spoil_plot_frame + '$')
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

  # Decide here what to plot against what.
  plot_order = "translation_time_angle"
  # plot_order = "angle_time_translation"
  # plot_order = "time_translation_angle"
  # plot_order = "time_translation_angle_max"
  num_measurements_per_bin = 60

  # Time angle
  if plot_order == "translation_time_angle":
    min_max_step_times = {
        'start': 0.0,
        'end': 0.23,
        'step': 0.03,
    }

    min_max_step_translation = {
        'start': 0.0,
        'end': 0.1,
        'step': 0.1,
    }
    min_max_step_angle = {
        'start': 0.0,
        'end': 120.0 / 180 * np.pi,
        'step': 15.0 / 180 * np.pi,
    }

  # Time translation
  elif plot_order == "angle_time_translation":
    min_max_step_times = {
        'start': 0.0,
        'end': 0.23,
        'step': 0.03,
    }

    min_max_step_translation = {
        'start': 0.0,
        'end': 0.8,
        'step': 0.1,
    }
    min_max_step_angle = {
        'start': 0.0,
        'end': 15.0 / 180 * np.pi,
        'step': 15.0 / 180 * np.pi,
    }

  # Angle translation
  elif plot_order == "time_translation_angle":
    min_max_step_times = {
        'start': 0.0,
        'end': 0.03,
        'step': 0.03,
    }

    min_max_step_translation = {
        'start': 0.0,
        'end': 0.8,
        'step': 0.1,
    }
    min_max_step_angle = {
        'start': 0.0,
        'end': 120.0 / 180 * np.pi,
        'step': 15.0 / 180 * np.pi,
    }

  # Angle translation complete
  elif plot_order == "time_translation_angle_max":
    min_max_step_times = {
        'start': 0.0,
        'end': 0.1,
        'step': 0.1,
    }

    min_max_step_translation = {
        'start': 0.0,
        'end': 1.0,
        'step': 0.1,
    }
    min_max_step_angle = {
        'start': 0.0,
        'end': 179.0 / 180 * np.pi,
        'step': 15.0 / 180 * np.pi,
    }
    plot_order = "time_translation_angle"
  # min_max_step_times = {
  #     'start': -0.0499999,
  #     'end': 0.05,
  #     'step': 0.05,
  # }
  #
  # min_max_step_translation = {
  #     'start': 0.0,
  #     'end': 0.1,
  #     'step': 0.02,
  # }
  # min_max_step_angle = {
  #     'start': 0.0,
  #     'end': 25.0 / 180 * np.pi,
  #     'step': 5.0 / 180 * np.pi,
  # }

  generate_optimization_circle_error_plot(
      min_max_step_times, min_max_step_translation, min_max_step_angle,
      spoiled_data, plot_order, num_measurements_per_bin)
