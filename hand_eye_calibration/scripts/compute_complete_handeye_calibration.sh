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
--aligned_poses_W_E_csv_file $ALIGNED_POSES_B

rosrun hand_eye_calibration compute_hand_eye_calibration.py \
--aligned_poses_B_H_csv_file $ALIGNED_POSES_A  \
--aligned_poses_W_E_csv_file $ALIGNED_POSES_B  \
--visualize=1
