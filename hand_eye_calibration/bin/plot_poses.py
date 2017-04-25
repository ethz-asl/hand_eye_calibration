#!/usr/bin/env python

import sys
sys.path.insert(0,  '/usr/local/lib/python2.7/dist-packages')

import matplotlib
matplotlib.use('Qt4Agg')


import sys
import numpy as np
from hand_eye_calibration.csv_io import read_time_stamped_poses_from_csv_file
from hand_eye_calibration.time_alignment_plotting_tools import (plot_results, plot_input_data,
                                           plot_time_stamped_poses,
                                           plot_angular_velocities)
from hand_eye_calibration.time_alignment import resample_quaternions, compute_angular_velocity_norms

plotInput = False

if __name__ == '__main__':
    time_stamped_poses_list = []
    angular_velocity_norms = []
    quaternions_list = []
    quaternions_interp_list = []
    
    dt = 0.02
    for f in sys.argv[1:]:
        time_stamped_poses, times, quaternions = read_time_stamped_poses_from_csv_file(f)
        time_stamped_poses_list.append(time_stamped_poses)
        quaternions_list.append(quaternions)
    
        if plotInput:
            quaternions_interp, samples = resample_quaternions(times, quaternions, dt)
            quaternions_interp_list.append(quaternions_interp)
            
            angular_velocity_norms.append(
                    compute_angular_velocity_norms(
                    quaternions_interp, samples,
                    25, 99))

    if plotInput:
        plot_input_data(quaternions_list[0], quaternions_list[1], quaternions_interp_list[0],
                  quaternions_interp_list[1], angular_velocity_norms[0],
                  angular_velocity_norms[1],
                  angular_velocity_norms[0],
                  angular_velocity_norms[1], True)
    
    print "Plotting poses:"
    plot_time_stamped_poses("Poses2", np.array(time_stamped_poses_list[0]), np.array(time_stamped_poses_list[1]))
