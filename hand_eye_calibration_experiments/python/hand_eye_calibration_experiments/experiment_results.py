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
    self.singular_values = []
    self.bad_singular_value = []
    self.optimization_enabled = False
    self.optimization_success = []

  def init_from_configs(self, name, time_alignment_config, hand_eye_config, optimization_config):
    self.algorithm_name = name
    self.prefiltering = hand_eye_config.prefilter_poses_enabled
    self.optimization_enabled = optimization_config.enable_optimization

  def check_length(self, num_pose_pairs):
    assert len(self.dataset_names) == num_pose_pairs
    assert len(self.success) == num_pose_pairs
    assert len(self.rmse) == num_pose_pairs
    assert len(self.num_inliers) == num_pose_pairs
    assert len(self.num_initial_poses) == num_pose_pairs
    assert len(self.num_poses_kept) == num_pose_pairs
    assert len(self.runtimes) == num_pose_pairs
    assert len(self.singular_values) == num_pose_pairs
    assert len(self.bad_singular_value) == num_pose_pairs
    assert len(self.optimization_success) == num_pose_pairs

  def get_header(self):
    return ("algorithm_name,"
            "num_pose_pairs,"
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

  def write_pose_pair_to_csv_line(self, num_pose_pairs):
    return "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
        self.algorithm_name,
        num_pose_pairs,
        self.iteration_num,
        self.prefiltering_enabled,
        self.dataset_names[num_pose_pairs][0],
        self.dataset_names[num_pose_pairs][1],
        self.success[num_pose_pairs],
        self.rmse[num_pose_pairs][0],
        self.rmse[num_pose_pairs][1],
        self.num_inliers[num_pose_pairs],
        self.num_initial_poses[num_pose_pairs],
        self.num_poses_kept[num_pose_pairs],
        self.runtimes[num_pose_pairs],
        self.loop_error_position,
        self.loop_error_orientation,
        np.array_str(
            self.singular_values[num_pose_pairs], max_line_width=1000000),
        self.bad_singular_value[num_pose_pairs],
        self.optimization_enabled,
        self.optimization_success[num_pose_pairs])
