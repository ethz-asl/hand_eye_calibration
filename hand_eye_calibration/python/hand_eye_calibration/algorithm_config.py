
from hand_eye_calibration.dual_quaternion_hand_eye_calibration import HandEyeConfig
from hand_eye_calibration.time_alignment import FilteringConfig


def get_basic_config():

  ###########################
  ## TIME ALIGNMENT CONFIG ##
  ###########################

  time_alignment_config = FilteringConfig()
  time_alignment_config.smoothing_kernel_size_A = 25
  time_alignment_config.clipping_percentile_A = 99.5
  time_alignment_config.smoothing_kernel_size_B = 25
  time_alignment_config.clipping_percentile_B = 99.0

  #####################
  ## HAND EYE CONFIG ##
  #####################

  hand_eye_config = HandEyeConfig()

  # General config
  hand_eye_config.use_baseline_approach = False
  hand_eye_config.algorithm_name = ""
  hand_eye_config.enable_exhaustive_search = False
  hand_eye_config.min_num_inliers = 10

  # Select distinctive poses based on skrew axis
  hand_eye_config.prefilter_poses_enabled = False
  # 0.99 still works but only 0.95 makes exhaustive search tractable.
  hand_eye_config.prefilter_dot_product_threshold = 0.95

  # RANSAC
  hand_eye_config.ransac_sample_size = 3
  hand_eye_config.ransac_sample_rejection_scalar_part_equality_tolerance = 1e-2
  hand_eye_config.ransac_max_number_iterations = 20
  hand_eye_config.ransac_enable_early_abort = True
  hand_eye_config.ransac_outlier_probability = 0.5
  hand_eye_config.ransac_success_probability_threshold = 0.99
  hand_eye_config.ransac_inlier_classification = "scalar_part_equality"
  hand_eye_config.ransac_position_error_threshold_m = 0.02
  hand_eye_config.ransac_orientation_error_threshold_deg = 1.0
  hand_eye_config.ransac_model_refinement = True
  hand_eye_config.ransac_evaluate_refined_model_on_inliers_only = False

  # Hand-calibration
  hand_eye_config.hand_eye_calibration_scalar_part_equality_tolerance = 4e-2

  # Visualization
  hand_eye_config.visualize = False
  hand_eye_config.visualize_plot_every_nth_pose = 10

  return (time_alignment_config, hand_eye_config)


def get_RANSAC_classic_config(prefilter_poses):
  """
  Get config for the "RANSAC - Classic (pose inliers)" algorithm.
  """

  (time_alignment_config, hand_eye_config) = get_basic_config()

  if prefilter_poses:
    hand_eye_config.algorithm_name = "RC_filter"
  else:
    hand_eye_config.algorithm_name = "RC_no_filter"

  # Select distinctive poses based on skrew axis
  hand_eye_config.prefilter_poses_enabled = prefilter_poses

  # RANSAC
  hand_eye_config.ransac_sample_size = 3
  hand_eye_config.enable_exhaustive_search = False

  # Inlier/Outlier detection
  hand_eye_config.ransac_inlier_classification = "rmse_threshold"

  return (time_alignment_config, hand_eye_config)


def get_RANSAC_scalar_part_inliers_config(prefilter_poses):
  """
  Get config for the "RANSAC - scalar part inliers" algorithm.
  """
  (time_alignment_config, hand_eye_config) = get_basic_config()

  if prefilter_poses:
    hand_eye_config.algorithm_name = "RS_filter"
  else:
    hand_eye_config.algorithm_name = "RS_no_filter"

  # Select distinctive poses based on skrew axis
  hand_eye_config.prefilter_poses_enabled = prefilter_poses

  # RANSAC
  hand_eye_config.ransac_sample_size = 1
  hand_eye_config.enable_exhaustive_search = False

  # Inlier/Outlier detection
  hand_eye_config.ransac_inlier_classification = "scalar_part_equality"

  return (time_alignment_config, hand_eye_config)


def get_exhaustive_search_pose_inliers_config():
  """
  Get config for the "Exhaustive search - pose inliers" algorithm.
  """
  (time_alignment_config, hand_eye_config) = get_basic_config()

  hand_eye_config.algorithm_name = "EC"

  # Prefiltering is mandatory for exhaustive search,
  # otherwise it takes forever.
  hand_eye_config.prefilter_poses_enabled = True

  # RANSAC
  hand_eye_config.ransac_sample_size = 3
  hand_eye_config.enable_exhaustive_search = True

  # Inlier/Outlier detection
  hand_eye_config.ransac_inlier_classification = "rmse_threshold"

  return (time_alignment_config, hand_eye_config)


def get_exhaustive_search_scalar_part_inliers_config():
  """
  Get config for the "Exhaustive search - scalar part inliers" algorithm.
  """
  (time_alignment_config, hand_eye_config) = get_basic_config()

  hand_eye_config.algorithm_name = "ES"

  # Prefiltering is mandatory for exhaustive search,
  # otherwise it takes forever.
  hand_eye_config.prefilter_poses_enabled = True

  # RANSAC
  hand_eye_config.ransac_sample_size = 1
  hand_eye_config.enable_exhaustive_search = True

  # Inlier/Outlier detection
  hand_eye_config.ransac_inlier_classification = "scalar_part_equality"

  return (time_alignment_config, hand_eye_config)


def get_baseline_config(prefilter_poses):
  """
  Get config for the "Baseline" algorithm.
  """
  (time_alignment_config, hand_eye_config) = get_basic_config()

  if prefilter_poses:
    hand_eye_config.algorithm_name = "baseline_filter"
  else:
    hand_eye_config.algorithm_name = "baseline_no_filter"

  hand_eye_config.use_baseline_approach = True

  # Select distinctive poses based on skrew axis
  hand_eye_config.prefilter_poses_enabled = prefilter_poses

  hand_eye_config.enable_exhaustive_search = False

  return (time_alignment_config, hand_eye_config)
