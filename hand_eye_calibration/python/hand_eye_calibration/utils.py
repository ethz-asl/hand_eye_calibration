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


def readArrayFromCsv(csv_file):
  with open(csv_file, 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    return np.array(list(csv_reader), dtype=float)


def run(cmd, dry_run=False):
  if dry_run:
    print("Would normally run:", cmd)
    return
  print("Running:", cmd)
  import subprocess
  import shlex
  args = shlex.split(cmd)
  proc = subprocess.Popen(args)
  exit_code = proc.wait()
  if exit_code != 0:
    raise Exception("Cmd '%s' returned nozero exit code : %d" %
                    (cmd, exit_code))


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
