# Hand-Eye Calibration

## Formats
Our hand-eye calibration expects timestamped poses with Hamiltonian quaternions in the following format, where [.] denotes the unit:
> p = [t[s], x[m], y[m], z[m], q_x, q_y, q_z, q_w]

Our scripts expect CSV files with the following format:
```
t, x, y, z, q_x, q_y, q_z, q_w
```

## Frames
In our hand-eye calibration we use the following frames:
- H: Hand — The frame of the robot end-effector (or the vicon output pose).
- B: Base — The robot's base frame, usually the end-effector poses are expressed with respect to this frame.
- E: Eye — The frame of the camera.
- W: World - The frame of the target.

**TODO:** Add the transformations here.

## Usage

All our tools can either be run via ROS, using

```
rosrun hand_eye_calibration <tool>.py [arguments]
```

or directly by changing into this directory (e.g. `~/catkin_ws/src/hand_eye_calibration/hand_eye_calibration`) and executing:

```
./bin/<tool>.py [arguments]
```

A typical use case consists of the following steps (here using ROS):
- Extract poses from tf (ROS transformation type) messages (with time stamps).
  ```
  rosrun hand_eye_calibration tf_to_csv.py --bag calibration.bag --tf_source_frame end_effector --tf_target_frame base_link --csv_output_file tf_poses_timestamped.csv
  ```
- Extract poses from images (with time stamps).
  ```
  rosrun hand_eye_calibration target_extractor_interface.py --bag calibration.bag --calib_file_camera calib/camera_intrinsics.yaml --calib_file_target calib/target.yaml --image_topic /camera/rgb/image_raw --output_file camera_poses_timestamped.csv
  ```
- Timely align the poses and interpolate the two sets at given time stamps.
  ```
  rosrun hand_eye_calibration compute_aligned_poses.py --poses_B_H_csv_file tf_poses_timestamped.csv --poses_W_E_csv_file camera_poses_timestamped.csv --aligned_poses_B_H_csv_file tf_aligned.csv --aligned_poses_W_E_csv_file camera_aligned.csv
  ```
- Perform the hand-eye calibration.
  ```
  rosrun hand_eye_calibration compute_hand_eye_calibration.py --aligned_poses_B_H_csv_file tf_aligned.csv --aligned_poses_W_E_csv_file camera_aligned.csv --visualize True
  ```

## Running the Tests
Tests are all python [unittests](https://docs.python.org/3.7/library/unittest.html) and can be run with the following command:
```
python test/test_<test_filename>.py
```
or you can directly invoke the tests with catkin:
```
catkin run_tests hand_eye_calibration
```
