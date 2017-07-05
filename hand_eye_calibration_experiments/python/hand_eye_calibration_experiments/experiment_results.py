import numpy as np

from hand_eye_calibration.dual_quaternion_hand_eye_calibration import HandEyeConfig
from hand_eye_calibration.time_alignment import FilteringConfig


class ResultEntry:

  def __init__(self):
    self.algorithm_name = ""
    self.iteration_num = -1
    self.prefiltering_enabled = False
    self.dataset_names = []
    self.success = []
    self.rmse = []
    self.num_inliers = []
    self.num_initial_poses = []
    self.num_poses_kept = []
    self.runtimes = []
    self.loop_error_position = -1
    self.loop_error_orientation = -1
    self.singular_values = np.array()
    self.bad_singular_value = []
    self.optimization_enabled = False
    self.optimization_success = []

  def init_from_configs(self, name, time_alignment_config, hand_eye_config, optimization_config):
    self.algorithm_name = name
    self.prefiltering = hand_eye_config.prefilter_poses_enabled
    self.optimization_enabled = optimization_config.enable_optimization

  def get_header(self):
    return ("algorithm_name,"
            "pose_pair_num,"
            "iteration_num,"
            "prefiltering,"
            "poses_B_H_csv_file,"
            "poses_W_E_csv_file,"
            "success,"
            "position_rmse,"
            "orientation_rmse,"
            "num_inliers,"
            "num_input_poses,"
            "num_poses"
            "after_filtering,"
            "runtime_s,"
            "loop_error_position_m,"
            "loop_error_orientation_deg,"
            "singular_values,"
            "bad_singular_values,"
            "optimization_enabled,"
            "optimization_success\n")

  def write_pose_pair_to_csv_line(self, pose_pair_num):
    return "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
        algorithm_name,
        pose_pair_num,
        iteration_num,
        prefiltering_enabled,
        dataset_names[pose_pair_num][0],
        dataset_names[pose_pair_num][1],
        success[pose_pair_num],
        rmse[pose_pair_num][0],
        rmse[pose_pair_num][1],
        num_inliers[pose_pair_num],
        num_initial_poses[pose_pair_num],
        num_poses_kept[pose_pair_num],
        runtimes[pose_pair_num],
        loop_error_position,
        loop_error_orientation,
        np.array_str(
            singular_values[pose_pair_num], max_line_width=1000000),
        bad_singular_value[pose_pair_num],
        optimization_enabled,
        optimization_success[pose_pair_num])
