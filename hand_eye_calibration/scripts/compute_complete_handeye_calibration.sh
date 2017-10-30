#!/bin/bash

DATASET_DIR_A=`basename $1`
DATASET_PATH_A=`dirname $1`

DATASET_DIR_B=`basename $2`
DATASET_PATH_B=`dirname $2`

ALIGNED_POSES_A="${DATASET_PATH_A}/aligned_${DATASET_DIR_A}"
ALIGNED_POSES_B="${DATASET_PATH_B}/aligned_${DATASET_DIR_B}"


rosrun hand_eye_calibration compute_aligned_poses.py \
--poses_B_H_csv_file $1  \
--poses_W_E_csv_file $2 \
--aligned_poses_B_H_csv_file $ALIGNED_POSES_A \
--aligned_poses_W_E_csv_file $ALIGNED_POSES_B \
--time_offset_output_csv_file time_offset.csv

rosrun hand_eye_calibration compute_hand_eye_calibration.py \
--aligned_poses_B_H_csv_file $ALIGNED_POSES_A  \
--aligned_poses_W_E_csv_file $ALIGNED_POSES_B  \
--time_offset_input_csv_file time_offset.csv \
--calibration_output_json_file calibration.json \
--visualize=0

rosrun hand_eye_calibration_batch_estimation batch_estimator \
--v 1 \
--pose1_csv $1 \
--pose2_csv $2 \
--init_guess_file calibration.json \
--output_file calibration_optimized.json
