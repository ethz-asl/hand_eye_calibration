# Hand-Eye-Calibration

## Description

Python tools to perform hand-eye calibration.

If you are using these tools, please cite our [paper](http://www.fsr.ethz.ch/papers/FSR_2017_paper_73.pdf):

```bibtex
@Inbook{Furrer2017FSR,
author="Furrer, Fadri
and Fehr, Marius
and Novkovic, Tonci
and Sommer, Hannes
and Gilitschenski, Igor
and Siegwart, Roland",
editor="Siegwart, Roland
and Hutter, Marco",
title="Evaluation of Combined Time-Offset Estimation and Hand-Eye Calibration on Robotic Datasets",
bookTitle="Field and Service Robotics: Results of the 11th International Conference",
year="2017",
publisher="Springer International Publishing",
address="Cham",
isbn="978-3-319-67361-5"
}
```

It includes time alignment of sets of poses, the implementation of a dual-quaternion based approach to solve the hand eye calibration, pre-filtering and filtering of poses, as well as the integration of a pose refinement step using batch optimization from [oomact](https://github.com/ethz-asl/oomact).

There are also classes that implement quaternions and dual-quaternions a set of plotting tools that were used to generate the plots in the paper.

The datasets where these algorithms are evaluated on can be found [here](http://projects.asl.ethz.ch/datasets/doku.php?id=handeyecalibration2017).

## Installation

### System Dependencies - Ubuntu 16.04

```bash
# Install ROS repository
sudo apt-get install software-properties-common libv4l-dev
sudo add-apt-repository "deb http://packages.ros.org/ros/ubuntu xenial main"
wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -O - | sudo apt-key add -
sudo apt-get update

# Install system depdendencies [INCOMPLETE]
sudo apt-get install ros-kinetic-desktop-full doxygen python-catkin-tools

```

### Workspace - OSX / Ubuntu 16.04 / Ubuntu 14.04
```bash
# Create catkin workspace.
export CATKIN_WS=~/catkin_ws
mkdir -p $CATKIN_WS/src
cd $CATKIN_WS
catkin init
catkin config --merge-devel
catkin config --extend /opt/ros/<YOUR_ROS_DISTRO>
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release

# Clone the repositories and its dependencies.
cd src
git clone https://github.com/ethz-asl/hand_eye_calibration.git
wstool init
wstool merge hand_eye_calibration/all_dependencies.rosinstall
wstool update -j 8

# Build hand_eye_calibration_package
catkin build hand_eye_calibration hand_eye_calibration_target_extractor hand_eye_calibration_batch_estimation
```

## Tutorial

### Formats
Our hand-eye calibration expects timestamped poses with Hamiltonian quaternions in the following format, where [.] denotes the unit:
> p = [t[s], x[m], y[m], z[m], q_x, q_y, q_z, q_w]

Our scripts expect CSV files with the following format:
```
t, x, y, z, q_x, q_y, q_z, q_w
```

### Frames
In our hand-eye calibration we use the following frames:
- H: Hand — The frame of the robot end-effector (or the vicon output pose).
- B: Base — The robot's base frame, usually the end-effector poses are expressed with respect to this frame.
- E: Eye — The frame of the camera.
- W: World - The frame of the target.

### Usage

All our tools can either be run via ROS, using

```bash
rosrun hand_eye_calibration <tool>.py [arguments]
```

or directly by changing into this directory (e.g. `~/catkin_ws/src/hand_eye_calibration/hand_eye_calibration`) and executing:

```bash
./bin/<tool>.py [arguments]
```

### Step-by-Step Calibration

A typical use case consists of the following steps (here using ROS):

- Extract poses from tf (ROS transformation type) messages (with time stamps):
  ```bash
  rosrun hand_eye_calibration tf_to_csv.py --bag calibration.bag --tf_source_frame end_effector --tf_target_frame base_link --csv_output_file tf_poses_timestamped.csv
  ```
- Extract poses from images (with time stamps):
  ```bash
  rosrun hand_eye_calibration target_extractor_interface.py \
    --bag calibration.bag \
    --calib_file_camera calib/camera_intrinsics.yaml \
    --calib_file_target calib/target.yaml \
    --image_topic /camera/rgb/image_raw \
    --output_file camera_poses_timestamped.csv
  ```
- Time alignment of the poses and interpolate the two sets at given time stamps:
  ```bash
  rosrun hand_eye_calibration compute_aligned_poses.py \
    --poses_B_H_csv_file tf_poses_timestamped.csv \
    --poses_W_E_csv_file camera_poses_timestamped.csv \ --aligned_poses_B_H_csv_file tf_aligned.csv \
    --aligned_poses_W_E_csv_file camera_aligned.csv \
    --time_offset_output_csv_file time_offset.csv
  ```
- Perform the dual-quaternion-based hand-eye calibration:
  ```bash
  rosrun hand_eye_calibration compute_hand_eye_calibration.py \
    --aligned_poses_B_H_csv_file tf_aligned.csv  \
    --aligned_poses_W_E_csv_file camera_aligned.csv \
    --time_offset_input_csv_file time_offset.csv \
    --calibration_output_json_file calibration.json \
    --visualize True
  ```
- Run optimization to refine the calibration:
 ```bash
  rosrun hand_eye_calibration_batch_estimation batch_estimator \
    --v 1 \
    --pose1_csv tf_poses_timestamped.csv \
    --pose2_csv camera_poses_timestamped.csv \
    --init_guess_file calibration.json \
    --output_file calibration_optimized.json
  ```

### End-to-End Calibration

If you already have the CSV files ready as described above you can use the end-to-end calibration script as follows:

```bash
rosrun hand_eye_calibration compute_complete_handeye_calibration.sh \
poses_B_H.csv poses_W_E.csv
```

### Running the Tests

Tests are all python [unittests](https://docs.python.org/3.7/library/unittest.html) and can be run with the following command:

```
python test/test_<test_filename>.py
```
or you can directly invoke the tests with catkin:
```
catkin run_tests hand_eye_calibration
```
