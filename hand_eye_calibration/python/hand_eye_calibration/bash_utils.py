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


def create_path(path):
  if not os.path.exists(os.path.dirname(path)):
    try:
      os.makedirs(os.path.dirname(path))
    except OSError as exc:  # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise
