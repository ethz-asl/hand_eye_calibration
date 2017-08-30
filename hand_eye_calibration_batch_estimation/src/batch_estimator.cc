#include <fstream>
#include <limits>
#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Core>
#include <aslam/calibration/input/MotionCaptureSource.hpp>
#include <aslam/calibration/calibrator/CalibratorI.hpp>
#include <aslam/calibration/data/MeasurementsContainer.h>
#include <aslam/calibration/data/PoseMeasurement.h>
#include <aslam/calibration/model/FrameGraphModel.h>
#include <aslam/calibration/model/PoseTrajectory.h>
#include <aslam/calibration/model/sensors/PoseSensor.hpp>
#include <aslam/calibration/model/sensors/PositionSensor.hpp>
#include <ros/package.h>
#include <sm/boost/null_deleter.hpp>
#include <sm/BoostPropertyTree.hpp>
#include <sm/value_store/LayeredValueStore.hpp>
#include <sm/value_store/PrefixedValueStore.hpp>

constexpr const int kCsvOutputFixedPrecision = 18;
#define PACKAGE_PATH_MARKER "<PACKAGE>"

DEFINE_string(
    pose1_csv, "pose1.csv",
    "Pose1 input file. "
    "format: [frame_id, x[mm], y[mm], z[mm], x, y, z, w]");

DEFINE_string(
    pose2_csv, "pose2.csv",
    "Pose2 input file. "
    "format: [frame_id, x[mm], y[mm], z[mm], x, y, z, w]");

DEFINE_string(
  config_file,
  PACKAGE_PATH_MARKER "/conf/config.info",
  "Configuration file. The marker '" PACKAGE_PATH_MARKER "' gets replaced with this package's path.");

DEFINE_string(
  model_config,
  "",
  "Comma separated configuration strings shadowing the content of the configuration file in the model section (for EXPERTS, no validation yet).");

DEFINE_string(
  init_guess_file,
  "initial_guess.json",
  "Initial guess for the spatiotemporal extrinsics of the second sensor w.r.t. to the first.");

DEFINE_string(
  output_file,
  "output.json",
  "Output for the spatiotemporal extrinsics of the second sensor w.r.t. to the first.");

DEFINE_bool(
  use_jpl,
  false,
  "Use JPL quaternion convention.");

Eigen::Vector4d toInternalQuaternionConvention(const Eigen::Vector4d & q) {
  return FLAGS_use_jpl ? q : sm::kinematics::quatInv(q);
}
Eigen::Vector4d fromInternalQuaternionConvention(const Eigen::Vector4d & q) {
  return FLAGS_use_jpl ? q : sm::kinematics::quatInv(q);
}

using namespace aslam::calibration;
class SimpleModelFrame : public Frame, public NamedMinimal {
  using NamedMinimal::NamedMinimal;
};
SimpleModelFrame world("world");
SimpleModelFrame body("body");

ValueStoreRef valueStoreFromFile(std::string file_path, sm::BoostPropertyTree * bpt_ptr = nullptr) {
  sm::BoostPropertyTree bpt;
  if (bpt_ptr) {
    *bpt_ptr = bpt;
  }
  bpt.load(file_path);
  return ValueStoreRef(bpt);
}

void readPosesFromCsv(const std::string & path, PoseSensor & pose_sensor, aslam::calibration::CalibratorI & c) {
  std::ifstream indata(path);
  std::vector<double> values;
  if (indata.good()) {
    std::string line;
    while (std::getline(indata, line)) {
      std::stringstream line_stream(line);
      std::string cell;
      values.clear();
      while (std::getline(line_stream, cell, ',')) {
        values.push_back(std::stod(cell));
      }
      if (values.size() == 1) {  // outlier
        pose_sensor.addMeasurement(values[0], PoseSensor::Outlier, c.getCurrentStorage());
      } else {
        CHECK_EQ(8, values.size());
        Eigen::Vector3d t(values[1], values[2], values[3]);
        Eigen::Vector4d q(values[4], values[5], values[6], values[7]);
        q = toInternalQuaternionConvention(q);
        CHECK_NEAR(1, q.norm(), 1e-8);
        pose_sensor.addMeasurement(Timestamp(values[0]), q, t, c.getCurrentStorage());
      }
    }
  } else {
    LOG(FATAL)<<"Could not open " << path;
  }
}

void writePosesToCsv(const std::string & path, aslam::calibration::CalibratorI & c, Sensor & sensor, double sample_delta) {
  std::ofstream out(path);
  out.precision(kCsvOutputFixedPrecision);
  out << std::fixed;

  Eigen::IOFormat comma_fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
  if (out.good()) {
    for (Timestamp t = c.getCurrentEffectiveBatchInterval().start; t <= c.getCurrentEffectiveBatchInterval().end; t += sample_delta) {
      auto model_at = c.getModelAt(t, 0, { });
      auto T = sensor.getTransformationTo(model_at, world);
      auto q = fromInternalQuaternionConvention(T.q());
      out << double(t) << ", " << T.t().format(comma_fmt) << ", " << q.format(comma_fmt) << std::endl;
    }
  } else {
    LOG(FATAL)<<"Could not open " << path;
  }
}

ValueStoreRef loadConfigFile() {
  //TODO find a solution for configuration files in installed packages. Maybe this works already?
  std::string config_file = FLAGS_config_file;
  const auto marker_pos = config_file.find(PACKAGE_PATH_MARKER);
  if (marker_pos != std::string::npos) {

    const std::string package_path = ros::package::getPath("hand_eye_calibration_batch_estimation");
    config_file = config_file.replace(marker_pos, strlen(PACKAGE_PATH_MARKER), package_path);
  }
  LOG(INFO) << "Loading " << config_file << " as base configuration file.";
  return valueStoreFromFile(config_file);
}


// From https://stackoverflow.com/a/27511119
std::vector<std::string> split(const std::string &s, char delim) {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}
sm::BoostPropertyTree loadExtraConfigStrings() {
  sm::BoostPropertyTree bpt_extra;
  if (!FLAGS_model_config.empty()) {
    bpt_extra.loadStrings(split(FLAGS_model_config, ','));
    LOG(INFO)
      << "Loaded extra configuration\n*******************\n"
      << bpt_extra.asInfoString() <<"\n*******************\n";
  }
  return bpt_extra;
}

int main(int argc, char ** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(FLAGS_v > 0 ? google::INFO : google::WARNING);
  google::InstallFailureSignalHandler();

  auto vs_config = loadConfigFile();
  sm::BoostPropertyTree bpt_extra = loadExtraConfigStrings();
  sm::BoostPropertyTree init_guess_file_bpt;
  auto vs_init_guess = valueStoreFromFile(FLAGS_init_guess_file, &init_guess_file_bpt);
  auto vs_model = ValueStoreRef(
      new sm::LayeredValueStore(ValueStoreRef(bpt_extra),
      ValueStoreRef(new sm::PrefixedValueStore(vs_init_guess, PrefixedValueStore::PrefixMode::REMOVE, "pose2")),
      vs_config.getChild("model"))
    );

  FrameGraphModel model(vs_model, nullptr, { &body, &world });
  PoseSensor pose1_sensor(model, "pose1", vs_model);
  PoseSensor pose2_sensor(model, "pose2", vs_model);
  PoseTrajectory traj(model, "traj", vs_model);
  model.addModulesAndInit(pose1_sensor, pose2_sensor, traj);

  auto c = createBatchCalibrator(vs_config.getChild("calibrator"), std::shared_ptr<Model>(&model, sm::null_deleter()));

  readPosesFromCsv(FLAGS_pose1_csv, pose1_sensor, *c);
  readPosesFromCsv(FLAGS_pose2_csv, pose2_sensor, *c);

  for (auto & m : pose1_sensor.getAllMeasurements(c->getCurrentStorage())) {
    c->addMeasurementTimestamp(m.first, pose1_sensor);  // add timestamps to determine the batch interval
  }

  c->calibrate();

  const double dt = 0.02;
  writePosesToCsv(FLAGS_pose1_csv + ".out", *c, pose1_sensor, dt);
  writePosesToCsv(FLAGS_pose2_csv + ".out", *c, pose2_sensor, dt);

  for (auto && c : model.getCalibrationVariables()) {
    c->updateStore();
  }
  if (!FLAGS_output_file.empty()) {
    LOG(INFO)<< "Writing output to " << FLAGS_output_file <<".";
    init_guess_file_bpt.save(FLAGS_output_file);
  }
}
