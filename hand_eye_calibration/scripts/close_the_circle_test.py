#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import json
import argparse
import numpy as np
import csv

from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.extrinsic_calibration import ExtrinsicCalibration
from hand_eye_calibration.bash_utils import run

DRY_RUN = False


def readArrayFromCsv(csv_file):
  with open(csv_file, 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    return np.array(list(csv_reader), dtype=float)


def getMTimes(inputs):
  return [os.path.getmtime(i) for i in inputs if os.path.exists(i)]


def requiresUpdate(inputs, outputs):
  its = getMTimes(inputs)
  ots = getMTimes(outputs)
  return len(ots) == 0 or (len(its) > 0 and max(its) > min(ots))


def computeCircle(name, calibs):
  c = calibs[0]
  for cc in calibs[1:]:
    c = c * cc
  print ("Circle %s :" % name, c)


def calibrateTwo(group, a_in, b_in, a, b):
  if not os.path.exists(group):
    os.mkdir(group)
  a_aligned = group + '/' + a + "_aligned.csv"
  b_aligned = group + '/' + b + "_aligned.csv"
  time_offset_file = group + '/' + 'time_offset.csv'
  pose_file = group + '/' + 'pose.csv'

  if requiresUpdate([a_in, b_in], [a_aligned, b_aligned, time_offset_file]):
    run("rosrun hand_eye_calibration compute_aligned_poses.py \
      --poses_B_H_csv_file %s  \
      --poses_W_E_csv_file %s \
      --aligned_poses_B_H_csv_file %s \
      --aligned_poses_W_E_csv_file %s \
      --time_offset_output_csv_file %s"
        % (a_in, b_in, a_aligned, b_aligned, time_offset_file), DRY_RUN)

  if requiresUpdate([a_aligned, b_aligned], [pose_file]):
    run("rosrun hand_eye_calibration compute_hand_eye_calibration.py \
      --aligned_poses_B_H_csv_file %s \
      --aligned_poses_W_E_csv_file %s  \
      --extrinsics_output_csv_file %s"
        % (a_aligned, b_aligned, pose_file), DRY_RUN)

  init_guess_file = group + "_init_guess.json"
  if requiresUpdate([time_offset_file, pose_file], [init_guess_file]):
    time_offset = float(readArrayFromCsv(time_offset_file)[0, 0])
    pose = readArrayFromCsv(pose_file).reshape((7,))
    calib = ExtrinsicCalibration(
        time_offset, DualQuaternion.from_pose_vector(pose))
    calib.writeJson(init_guess_file)

  calib_file = group + ".json"
  if requiresUpdate([a_in, b_in, init_guess_file], [calib_file]):
    run("rosrun hand_eye_calibration_batch_estimation batch_estimator -v 1 \
      --pose1_csv=%s --pose2_csv=%s \
      --init_guess_file=%s \
      --output_file=%s"
        % (a_in, b_in, init_guess_file, calib_file), DRY_RUN)

  cal = ExtrinsicCalibration.fromJson(calib_file)
  cal_init = ExtrinsicCalibration.fromJson(init_guess_file)
  print(cal_init, "->\n", cal)
  return cal, cal_init


if __name__ == '__main__':
  input = sys.argv[1:];
  names = [os.path.splitext(os.path.basename(a))[0] for a in input]
  print("Names:", names)

  if len(names) == 2:
    calibrateTwo("test", input[0], input[1], names[0], names[1])
  elif len(names) == 3:
    calibs = []
    calibs_init = []
    for ia, a in enumerate(names):
      ib = (ia + 1) % 3
      b = names[ib]
      print (a, b)
      calib, calib_init = calibrateTwo(
          "%d_%d" % (ia, ib), input[ia], input[ib], a, b)
      calibs.append(calib)
      calibs_init.append(calib_init)

    computeCircle("before NLOPT", calibs_init)
    computeCircle("after NLOPT ", calibs)
