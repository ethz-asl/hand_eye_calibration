import numpy as np

from hand_eye_calibration.dual_quaternion_hand_eye_calibration import HandEyeConfig
from hand_eye_calibration.time_alignment import FilteringConfig
from hand_eye_calibration.algorithm_config import (get_basic_config,
                                                   get_RANSAC_classic_config,
                                                   get_RANSAC_scalar_part_inliers_config,
                                                   get_exhaustive_search_pose_inliers_config,
                                                   get_exhaustive_search_scalar_part_inliers_config,
                                                   get_baseline_config)


class OptimizationConfig:

  def __init__(self):
    self.enable_optimization = True
    self.optimization_only = False


def get_all_configs():
  return [
      get_baseline_and_optimization_config(True, True),
      get_baseline_and_optimization_config(False, True),
      get_baseline_and_optimization_config(True, False),
      get_baseline_and_optimization_config(False, False),

      get_RC_and_optimization_config(True, False),
      get_RC_and_optimization_config(False, False),
      get_RC_and_optimization_config(True, True),
      get_RC_and_optimization_config(False, True),

      get_RS_and_optimization_config(True, False),
      get_RS_and_optimization_config(False, False),
      get_RS_and_optimization_config(True, True),
      get_RS_and_optimization_config(False, True),

      get_EC_and_optimization_config(False),
      get_EC_and_optimization_config(True),

      get_ES_and_optimization_config(False),
      get_ES_and_optimization_config(True),
  ]


def get_baseline_and_optimization_config(enable_filtering, enable_optimization):
  """
  Get configuration struct for end-to-end testing for:
    "baseline" algorithm
    with optional pre-filtering and optimization.
  """

  (time_alignment_config, hand_eye_config) = get_baseline_config(enable_filtering)

  optimiztion_config = OptimizationConfig()
  optimiztion_config.enable_optimization = enable_optimization
  optimiztion_config.optimization_only = False

  algorithm_name = (hand_eye_config.algorithm_name +
                    ("_opt" if enable_optimization else "_no_opt"))

  return (algorithm_name, time_alignment_config, hand_eye_config, optimiztion_config)


def get_RC_and_optimization_config(enable_filtering, enable_optimization):
  """
  Get configuration struct for end-to-end testing for:
    "RANSAC Classic (based on pose inlier)" algorithm
    with optional pre-filtering and optimization.
  """

  (time_alignment_config, hand_eye_config) = get_RANSAC_classic_config(enable_filtering)

  optimiztion_config = OptimizationConfig()
  optimiztion_config.enable_optimization = enable_optimization
  optimiztion_config.optimization_only = False

  algorithm_name = (hand_eye_config.algorithm_name +
                    ("_opt" if enable_optimization else "_no_opt"))

  return (algorithm_name, time_alignment_config, hand_eye_config, optimiztion_config)


def get_RS_and_optimization_config(enable_filtering, enable_optimization):
  """
  Get configuration struct for end-to-end testing for:
    "RANSAC based on scalar part inliers" algorithm
    with optional pre-filtering and optimization.
  """

  (time_alignment_config, hand_eye_config) = get_RANSAC_scalar_part_inliers_config(
      enable_filtering)

  optimiztion_config = OptimizationConfig()
  optimiztion_config.enable_optimization = enable_optimization
  optimiztion_config.optimization_only = False

  algorithm_name = (hand_eye_config.algorithm_name +
                    ("_opt" if enable_optimization else "_no_opt"))

  return (algorithm_name, time_alignment_config, hand_eye_config, optimiztion_config)


def get_EC_and_optimization_config(enable_optimization):
  """
  Get configuration struct for end-to-end testing for:
    "Exhaustive search version of RANSAC classic" algorithm
    with optional pre-filtering and optimization.
  """

  (time_alignment_config,
   hand_eye_config) = get_exhaustive_search_pose_inliers_config()

  optimiztion_config = OptimizationConfig()
  optimiztion_config.enable_optimization = enable_optimization
  optimiztion_config.optimization_only = False

  algorithm_name = (hand_eye_config.algorithm_name +
                    ("_opt" if enable_optimization else "_no_opt"))

  return (algorithm_name, time_alignment_config, hand_eye_config, optimiztion_config)


def get_ES_and_optimization_config(enable_optimization):
  """
  Get configuration struct for end-to-end testing for:
    "Exhaustive search version of RANSAC based on scalar part inliers" algorithm
    with optional pre-filtering and optimization.
  """

  (time_alignment_config,
   hand_eye_config) = get_exhaustive_search_scalar_part_inliers_config()

  optimiztion_config = OptimizationConfig()
  optimiztion_config.enable_optimization = enable_optimization
  optimiztion_config.optimization_only = False

  algorithm_name = (hand_eye_config.algorithm_name +
                    ("_opt" if enable_optimization else "_no_opt"))

  return (algorithm_name, time_alignment_config, hand_eye_config, optimiztion_config)


def get_optimization_with_spoiled_initial_calibration_config():
  """
  Get configuration struct for end-to-end testing for:
    "Optimization with spoiled initial calibration" algorithm.
  """

  (time_alignment_config, hand_eye_config) = get_exhaustive_search_scalar_part_inliers_config()

  optimiztion_config = OptimizationConfig()
  optimiztion_config.enable_optimization = True
  optimiztion_config.optimization_only = True

  algorithm_name = "optimization_w_spoiled_init_calibration"

  return (algorithm_name, time_alignment_config, hand_eye_config, optimiztion_config)
