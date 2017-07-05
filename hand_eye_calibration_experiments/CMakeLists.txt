cmake_minimum_required(VERSION 2.8.3)
project(hand_eye_calibration_experiments)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

catkin_python_setup()

cs_install()
cs_export()
