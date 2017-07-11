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
    self.optimization_runtime = []
    self.spoiled_initial_guess_angle_offset = []
    self.spoiled_initial_guess_translation_offset = []
    self.spoiled_initial_guess_time_offset = []

  def init_from_configs(self, name, iteration, time_alignment_config, hand_eye_config, optimization_config):
    self.algorithm_name = name
    self.prefiltering = hand_eye_config.prefilter_poses_enabled
    self.optimization_enabled = optimization_config.enable_optimization
    self.iteration_num = iteration

  def check_length(self, num_pose_pairs):
    assert len(self.dataset_names) == num_pose_pairs, "{} vs {}".format(
        len(self.dataset_names), num_pose_pairs)
    assert len(self.success) == num_pose_pairs, "{} vs {}".format(
        len(self.success), num_pose_pairs)
    assert len(self.rmse) == num_pose_pairs, "{} vs {}".format(
        len(self.rmse), num_pose_pairs)
    assert len(self.num_inliers) == num_pose_pairs, "{} vs {}".format(
        len(self.num_inliers), num_pose_pairs)
    assert len(self.num_initial_poses) == num_pose_pairs, "{} vs {}".format(
        len(self.num_initial_poses), num_pose_pairs)
    assert len(self.num_poses_kept) == num_pose_pairs, "{} vs {}".format(
        len(self.num_poses_kept), num_pose_pairs)
    assert len(self.runtimes) == num_pose_pairs, "{} vs {}".format(
        len(self.runtimes), num_pose_pairs)
    assert len(self.singular_values) == num_pose_pairs, "{} vs {}".format(
        len(self.singular_values), num_pose_pairs)
    assert len(self.bad_singular_value) == num_pose_pairs, "{} vs {}".format(
        len(self.bad_singular_value), num_pose_pairs)
    assert len(self.optimization_success) == num_pose_pairs, "{} vs {}".format(
        len(self.optimization_success), num_pose_pairs)
    assert len(self.optimization_runtime) == num_pose_pairs, "{} vs {}".format(
        len(self.optimization_runtime), num_pose_pairs)
    assert len(self.spoiled_initial_guess_angle_offset) == num_pose_pairs, "{} vs {}".format(
        len(self.spoiled_initial_guess_angle_offset), num_pose_pairs)
    assert len(self.spoiled_initial_guess_translation_offset) == num_pose_pairs, "{} vs {}".format(
        len(self.spoiled_initial_guess_translation_offset), num_pose_pairs)
    assert len(self.spoiled_initial_guess_time_offset) == num_pose_pairs, "{} vs {}".format(
        len(self.spoiled_initial_guess_time_offset), num_pose_pairs)

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
            "optimization_success,"
            "optimization_runtime_s,"
            "spoiled_initial_guess_angle_offset,"
            "spoiled_initial_guess_translation_offset,"
            "spoiled_initial_guess_time_offset"
            "\n")

  def write_pose_pair_to_csv_line(self, pose_pair_idx):

    singular_values = self.singular_values[pose_pair_idx]

    return "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
        self.algorithm_name,
        pose_pair_idx,
        self.iteration_num,
        self.prefiltering_enabled,
        self.dataset_names[pose_pair_idx][0],
        self.dataset_names[pose_pair_idx][1],
        self.success[pose_pair_idx],
        self.rmse[pose_pair_idx][0],
        self.rmse[pose_pair_idx][1],
        self.num_inliers[pose_pair_idx],
        self.num_initial_poses[pose_pair_idx],
        self.num_poses_kept[pose_pair_idx],
        self.runtimes[pose_pair_idx],
        self.loop_error_position,
        self.loop_error_orientation,
        ("" if singular_values is None else np.array_str(
            singular_values, max_line_width=1000000)),
        self.bad_singular_value[pose_pair_idx],
        self.optimization_enabled,
        self.optimization_success[pose_pair_idx],
        self.optimization_runtime[pose_pair_idx],
        self.spoiled_initial_guess_angle_offset[pose_pair_idx],
        (None if self.spoiled_initial_guess_translation_offset[pose_pair_idx] is None else np.array_str(
            self.spoiled_initial_guess_translation_offset[pose_pair_idx], max_line_width=1000000)),
        self.spoiled_initial_guess_time_offset[pose_pair_idx])
