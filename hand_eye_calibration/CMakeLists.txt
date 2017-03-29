cmake_minimum_required(VERSION 2.8.3)
project(hand_eye_calibration)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(--std=c++11)

catkin_python_setup()

cs_add_executable(target_extractor src/target_extractor.cc)

cs_install()
cs_export()

## Add folders to be run by python nosetests
catkin_add_nosetests(test/test_quaternion.py)
catkin_add_nosetests(test/test_dual_quaternion.py)

