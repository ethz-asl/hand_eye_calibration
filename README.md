# hand_eye_calibration
Python tools to perform hand-eye calibration.

If you are using these tools, please cite our [paper](http://www.fsr.ethz.ch/papers/FSR_2017_paper_73.pdf):

> @Inbook{Furrer2017FSR,
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

It includes time alignment of sets of poses, the implementation of a dual-quaternion based approach to solve the hand eye calibration, pre-filtering and filtering of poses, as well as the integration of a pose refinement step using batch optimization from [oomact](https://github.com/ethz-asl/oomact).

There are also classes that implement quaternions and dual-quaternions a set of plotting tools that were used to generate the plots in the paper.

The datasets where these algorithms are evaluated on can be found [here](http://projects.asl.ethz.ch/datasets/doku.php?id=handeyecalibration2017).


## Installation

**System Dependencies - Ubuntu 16.04:**

**TODO** Complete

```bash
# Install ROS repository
sudo apt-get install software-properties-common
sudo add-apt-repository "deb http://packages.ros.org/ros/ubuntu xenial main"
wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -O - | sudo apt-key add -
sudo apt-get update

# Install system depdendencies
sudo apt-get install ros-kinetic-desktop-full doxygen python-catkin-tools


```

**System Dependencies - OSX:**

 **TODO**



**Workspace - OSX / Ubuntu 16.04 / Ubuntu 14.04:**
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
git clone git@github.com:ethz-asl/hand_eye_calibration.git
wstool init
wstool merge hand_eye_calibration/all_dependencies.rosinstall
wstool update -j 8

# Build hand_eye_calibration_package
catkin build hand_eye_calibration hand_eye_calibration_target_extractor hand_eye_calibration_batch_estimation
```
