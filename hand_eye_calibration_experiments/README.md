**Usage:**

```
source $CATKIN_WS/devel/setup.bash
rosrun hand_eye_calibration_experiments compute_set_of_hand_eye_calibrations.py
          [-h]
          --aligned_poses_B_H_csv_files ALIGNED_POSES_B_H_CSV_FILES
          --result_file RESULT_FILE
          [--visualize VISUALIZE]
          [--plot_every_nth_pose PLOT_EVERY_NTH_POSE]
          [--num_iterations NUM_ITERATIONS]
```

**Example:**

```
source $CATKIN_WS/devel/setup.bash
rosrun hand_eye_calibration_experiments compute_set_of_hand_eye_calibrations.py --aligned_poses_B_H_csv_files=TANGO_exCALIBur_2/CALIGULA_2017-04-06-18-21-33.csv:TANGO_exCALIBur_2/MARS_2017-04-06-18-23-37.csv:TANGO_exCALIBur_2/NERO_2017-04-06-18-25-25.csv --visualize=True --plot_every_nth_pose=10 --num_iterations=2 --result_file TANGO_exCALIBur_2/result.csv

```
