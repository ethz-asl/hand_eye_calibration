#include <fstream>
#include <limits>
#include <memory>

#include <glog/logging.h>
#include <Eigen/Core>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <sm/BoostPropertyTree.hpp>
#include <sm/value_store/LayeredValueStore.hpp>
#include <sm/value_store/PrefixedValueStore.hpp>
#include <sm/boost/null_deleter.hpp>

#include <aslam/calibration/model/FrameGraphModel.h>
#include <aslam/calibration/model/sensors/PoseSensor.hpp>
#include <aslam/calibration/model/sensors/PositionSensor.hpp>
#include <aslam/calibration/model/PoseTrajectory.h>
#include <aslam/calibration/data/PoseMeasurement.h>
#include <aslam/calibration/data/MeasurementsContainer.h>
#include <aslam/calibration/CalibratorI.hpp>

#include "aslam/calibration/algo/MotionCaptureSource.hpp"


DEFINE_string(
    pose1csv, "pose1.csv",
    "Pose1 input file. "
    "format: [frame_id, x[mm], y[mm], z[mm], x, y, z, w]");

DEFINE_string(
    pose2csv, "pose2.csv",
    "Pose2 input file. "
    "format: [frame_id, x[mm], y[mm], z[mm], x, y, z, w]");

DEFINE_string(config_file, "config.info", "Configuration file.");
DEFINE_string(init_guess_file, "initial_guess.json", "Initial guess for the spatiotemporal extrinsics of the second sensor w.r.t. to the first.");
DEFINE_string(output_file, "output.json", "Output for the spatiotemporal extrinsics of the second sensor w.r.t. to the first.");
DEFINE_bool(use_jpl, false, "Use JPL quaternion convention.");

Eigen::Vector4d toInternalQuaternionConvention(const Eigen::Vector4d & q){
  return FLAGS_use_jpl ? q : sm::kinematics::quatInv(q);
}
Eigen::Vector4d fromInternalQuaternionConvention(const Eigen::Vector4d & q){
  return FLAGS_use_jpl ? q : sm::kinematics::quatInv(q);
}

using namespace aslam::calibration;
class SimpleModelFrame : public Frame, public NamedMinimal {
  using NamedMinimal::NamedMinimal;
};
SimpleModelFrame world("world");
SimpleModelFrame body("body");

ValueStoreRef valueStoreFromFile(std::string filePath, sm::BoostPropertyTree * bptPtr = nullptr){
  sm::BoostPropertyTree bpt;
  if(bptPtr){
    *bptPtr = bpt;
  }
  bpt.load(filePath);
  return ValueStoreRef(bpt);
}

void readPosesFromCsv(const std::string & path, PoseSensor & poseSensor){
    std::ifstream indata(path);
    std::vector<double> values;
    if(indata.good()){
      std::string line;
      while (std::getline(indata, line)) {
          std::stringstream lineStream(line);
          std::string cell;
          values.clear();
          while (std::getline(lineStream, cell, ',')) {
              values.push_back(std::stod(cell));
          }
          if(values.size() == 1){ // outlier
            poseSensor.addMeasurement(PoseSensor::Outlier, values[0]);
          } else {
            CHECK_EQ(8, values.size());
            Eigen::Vector3d t_m_mf(values[1], values[2], values[3]);
            Eigen::Vector4d q_m_f(values[4], values[5], values[6], values[7]);
            q_m_f = toInternalQuaternionConvention(q_m_f);
            CHECK_NEAR(1,  q_m_f.norm(), 1e-8);
            poseSensor.addMeasurement(q_m_f, t_m_mf, Timestamp(values[0]));
          }
      }
    } else {
      LOG(FATAL) <<"Could not open " << path;
    }
}

void writePosesToCsv(const std::string & path, aslam::calibration::CalibratorI & c, Sensor & sensor, double sampleDelta){
    std::ofstream out(path);
    out.precision(30);

    Eigen::IOFormat commaFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    if(out.good()){
      for(Timestamp t = c.getCurrentEffectiveBatchInterval().start; t <= c.getCurrentEffectiveBatchInterval().end; t += sampleDelta){
        auto modelAt = c.getModelAt(t, 0, {});
        auto T = sensor.getTransformationTo(modelAt, world);
        auto q = fromInternalQuaternionConvention(T.q());
        out << double(t) << ", " << T.t().format(commaFmt) << ", " << q.format(commaFmt) << std::endl;
      }
    } else {
      LOG(FATAL) <<"Could not open " << path;
    }
}


int main(int argc, char ** argv){
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(FLAGS_v > 0 ? google::INFO : google::WARNING);
  google::InstallFailureSignalHandler();

  auto vsConfig = valueStoreFromFile(FLAGS_config_file);
  sm::BoostPropertyTree initGuessFileBpt;
  auto vsInitGuess = valueStoreFromFile(FLAGS_init_guess_file, &initGuessFileBpt);
  auto vsModel = ValueStoreRef(new sm::LayeredValueStore(
      ValueStoreRef(new sm::PrefixedValueStore(vsInitGuess, PrefixedValueStore::PrefixMode::REMOVE, "pose2")),
      vsConfig.getChild("model")
    ));

  FrameGraphModel model(vsModel, nullptr, {&body, &world});
  PoseSensor pose1Sensor(model, "pose1", vsModel);
  PoseSensor pose2Sensor(model, "pose2", vsModel);
  PoseTrajectory traj(model, "traj", vsModel);
  model.addModulesAndInit(pose1Sensor, pose2Sensor, traj);

  readPosesFromCsv(FLAGS_pose1csv, pose1Sensor);
  readPosesFromCsv(FLAGS_pose2csv, pose2Sensor);

  auto c = createBatchCalibrator(vsConfig.getChild("calibrator"), std::shared_ptr<Model>(&model, sm::null_deleter()));

  for(auto & m : pose1Sensor.getMeasurements()){
    c->addMeasurementTimestamp(m.first, pose1Sensor); // add timestamps to determine the batch interval
  }

  c->calibrate();

  const double dt = 0.02;
  writePosesToCsv(FLAGS_pose1csv + ".out", *c, pose1Sensor, dt);
  writePosesToCsv(FLAGS_pose2csv + ".out", *c, pose2Sensor, dt);

  for(auto && c : model.getCalibrationVariables()){
    c->updateStore();
  }
  if(!FLAGS_output_file.empty()){
    LOG(INFO) << "Writing output to " << FLAGS_output_file <<".";
    initGuessFileBpt.save(FLAGS_output_file);
  }
}
