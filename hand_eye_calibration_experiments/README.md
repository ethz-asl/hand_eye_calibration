## End-to-End Experiment

**Usage:**

```
source $CATKIN_WS/devel/setup.bash
rosrun hand_eye_calibration_experiments compute_set_of_hand_eye_calibrations.py
          [-h]
          --aligned_poses_B_H_csv_files ALIGNED_POSES_B_H_CSV_FILES
          --result_directory RESULT_DIRECTORY
          [--visualize VISUALIZE]
          [--plot_every_nth_pose PLOT_EVERY_NTH_POSE]
          [--num_iterations NUM_ITERATIONS]
```

**Example:**

```
rosrun hand_eye_calibration_experiments compute_set_of_hand_eye_calibrations.py \ --aligned_poses_B_H_csv_files \ TANGO_exCALIBur_2/CALIGULA_2017-04-06-18-21-33.csv \ TANGO_exCALIBur_2/MARS_2017-04-06-18-23-37.csv \ TANGO_exCALIBur_2/NERO_2017-04-06-18-25-25.csv \
--num_iterations=10 \
--result_directory TANGO_exCALIBur_2/results \
--is_absolute_pose_sensor 0 0 0
```

## Optimization with Spoiled Initial Guess

**Usage:**

```
source $CATKIN_WS/devel/setup.bash
rosrun hand_eye_calibration_experiments optimization_experiments.py
          [-h]
          --aligned_poses_B_H_csv_files ALIGNED_POSES_B_H_CSV_FILES
          --result_directory RESULT_DIRECTORY
          [--visualize VISUALIZE]
          [--plot_every_nth_pose PLOT_EVERY_NTH_POSE]
          [--num_iterations NUM_ITERATIONS]
```

**Example:**

```
rosrun hand_eye_calibration_experiments optimization_experiments.py \ --aligned_poses_B_H_csv_files \ TANGO_exCALIBur_2/CALIGULA_2017-04-06-18-21-33.csv \ TANGO_exCALIBur_2/MARS_2017-04-06-18-23-37.csv \ TANGO_exCALIBur_2/NERO_2017-04-06-18-25-25.csv \
--num_iterations=10 \
--result_directory TANGO_exCALIBur_2/optimization_results \ --is_absolute_pose_sensor 0 0 0 \
```
