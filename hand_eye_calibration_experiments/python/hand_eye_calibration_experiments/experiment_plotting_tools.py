import numpy as np
import ast


def collect_data_from_csv(csv_file_names, get_header=True):
  dt = np.dtype([
      ('algorithm_name', np.str_, 50),
      ('num_pose_pairs', np.uint, 1),
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
      ('optimization_enabled', np.bool, 1),
      ('optimization_success', np.bool, 1),
      ('optimization_runtime_s', np.float64, 1),
      ('spoiled_initial_guess_angle_offset', np.float64, 1),
      ('spoiled_initial_guess_translation_offset', np.str_, 300),
      ('spoiled_initial_guess_time_offset', np.float64, 1),
      ('dataset', np.str_, 50)])

  print("Evaluating the following result files: {}".format(csv_file_names))
  for csv_file_name in csv_file_names:
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

  method_name_dict = {
      'baseline_filter_opt': 'PF_B_O',
      'baseline_no_filter_opt': 'B_O',
      'baseline_filter_no_opt': 'PF_B',
      'baseline_no_filter_no_opt': 'B',
      'EC_no_opt': 'PF_EC',
      'EC_opt': 'PF_EC_O',
      'ES_no_opt': 'PF_ES',
      'ES_opt': 'PF_ES_O',
      'RC_filter_no_opt': 'PF_RC',
      'RC_no_filter_no_opt': 'RC',
      'RC_filter_opt': 'PF_RC_O',
      'RC_no_filter_opt': 'RC_O',
      'RS_filter_no_opt': 'PF_RS',
      'RS_no_filter_no_opt': 'RS',
      'RS_filter_opt': 'PF_RS_O',
      'RS_no_filter_opt': 'RS_O',
      'optimization_w_spoiled_init_calibration': 'S_PF_ES_O'
  }

  methods = []
  datasets = []
  position_rmses_per_method = {}
  orientation_rmses_per_method = {}
  position_rmses = []
  orientation_rmses = []
  runtimes = []
  runtimes_per_method = {}
  loop_errors_position_m = []
  loop_errors_orientation_deg = []
  spoiled_initial_guess_angle_offsets = []
  spoiled_initial_guess_translation_norm_offsets = []
  spoiled_initial_guess_time_offsets = []

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
    loop_error_position_m = row[13]
    loop_error_orientation_deg = row[14]
    bad_singular_values = row[16]
    runtime_optimization = row[19]
    initial_guess_angle = 0 if row[20] is None else row[20]
    if row[21] != "None":
      initial_guess_string = ' '.join((row[21].lstrip()).split()).replace(
          " ", ",").replace("[,", "[").replace(",,", ",")
      initial_guess_translation = np.array(
          ast.literal_eval(initial_guess_string))
      initial_guess_translation_norm = np.linalg.norm(initial_guess_translation)
    else:
      initial_guess_translation_norm = 0
    initial_guess_time = 0 if row[22] is None else row[22]
    dataset = row[23]

    runtime += runtime_optimization
    method = method_name_dict[method]

    if not success or bad_singular_values:
      # TODO(ff): Create a plot with the success rate?
      # And for the initial guess spoil this should also be incorporated.
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

    loop_errors_position_m.append(loop_error_position_m)
    loop_errors_orientation_deg.append(loop_error_orientation_deg)
    spoiled_initial_guess_angle_offsets.append(initial_guess_angle)
    spoiled_initial_guess_translation_norm_offsets.append(
        initial_guess_translation_norm)
    spoiled_initial_guess_time_offsets.append(initial_guess_time)
  spoiled_data = [loop_errors_position_m, loop_errors_orientation_deg,
                  spoiled_initial_guess_angle_offsets,
                  spoiled_initial_guess_translation_norm_offsets,
                  spoiled_initial_guess_time_offsets]

  return [methods, datasets, position_rmses_per_method,
          orientation_rmses_per_method, position_rmses, orientation_rmses,
          runtimes, runtimes_per_method, spoiled_data]
