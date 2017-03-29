#include <fstream>
#include <limits>

#include <aslam/calibration/target-aprilgrid.h>
#include <aslam/cameras/camera.h>
#include <aslam/common/pose-types.h>
#include <aslam/geometric-vision/pnp-pose-estimator.h>
#include <glog/logging.h>
#include <Eigen/Core>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cv_bridge/cv_bridge.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#pragma GCC diagnostic pop

DEFINE_string(bag, "", "Rosbag path and filename.");
DEFINE_string(topic, "/cam0/image_raw", "Image topic name.");
DEFINE_bool(draw_extraction, false, "Show extracted corners.");
DEFINE_string(eval_camera_yaml, "",
              "Camera model definition to evaluate reprojection errors.");

DEFINE_string(camera_pose_T_C_G_output_file, "camera_pose.csv",
              "Outputs the camera poses for all successful observations. "
              "format: [frame_id, x, y, z, x, y, z, w] (JPL quaternion "
              "convention)");

DEFINE_string(camera_pose_T_C_G_output_file_timestamped,
              "camera_poses_timestamped.csv",
              "Outputs the camera poses for all successful observations. "
              "format: [timestamp, x, y, z, x, y, z, w] (JPL quaternion "
              "convention)");

DEFINE_string(
    camera_pose_T_C_G_output_file_mm, "camera_pose_mm.csv",
    "Outputs the camera poses for all successful observations. "
    "format: [frame_id, x[mm], y[mm], z[mm], x, y, z, w] (JPL quaternion "
    "convention)");

DEFINE_string(
    camera_pose_T_C_G_output_file_mm_timestamped,
    "camera_poses_mm_timestamped.csv",
    "Outputs the camera poses for all successful observations. "
    "format: [timestamp, x[mm], y[mm], z[mm], x, y, z, w] (JPL quaternion "
    "convention)");

DEFINE_bool(output_position_in_mm, false,
            "Converts the camera position into millimeters when outputting to "
            "csv file.");

DEFINE_bool(write_csv_header, true, "Write a header into the csv files.");

DEFINE_double(inlier_ratio_for_good_camera_pose, 0.4,
              "Minimal inlier ratio required for exported camera pose.");

DEFINE_int32(april_tag_number_vertical, 6, "Vertical number of april tags");
DEFINE_int32(april_tag_number_horizontal, 6, "Horizontal number of april tags");
DEFINE_double(april_tag_size_m, 0.055, "Size of april tags in meters.");
DEFINE_double(april_tag_gap_size_m, 0.0165,
              "Size of gaps between april tags in meters.");
DEFINE_int32(april_tag_pixel_boarder, 2,
             "Size of april tag boarder in pixels.");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  std::ofstream reprojection_logger;
  std::ofstream camera_pose_logger;
  std::ofstream camera_pose_logger_timestamped;
  std::ofstream camera_pose_logger_mm;
  std::ofstream camera_pose_logger_mm_timestamped;

  // Create target and detector.
  aslam::calibration::TargetAprilGrid::TargetConfiguration aprilgrid_config;
  aprilgrid_config.num_tag_rows = FLAGS_april_tag_number_vertical;
  aprilgrid_config.num_tag_cols = FLAGS_april_tag_number_horizontal;
  aprilgrid_config.tag_size_meter = FLAGS_april_tag_size_m;
  aprilgrid_config.tag_inbetween_space_meter = FLAGS_april_tag_gap_size_m;
  aprilgrid_config.black_tag_border_bits = FLAGS_april_tag_pixel_boarder;

  LOG(INFO) << "April Tag Target:";
  LOG(INFO) << "Dimension:     " << aprilgrid_config.num_tag_rows << "x"
            << aprilgrid_config.num_tag_rows;
  LOG(INFO) << "Tag size:      " << aprilgrid_config.tag_size_meter;
  LOG(INFO) << "Gap size:      " << aprilgrid_config.tag_inbetween_space_meter;
  LOG(INFO) << "Border pixels: " << aprilgrid_config.black_tag_border_bits;

  aslam::calibration::TargetAprilGrid::Ptr aprilgrid(
      new aslam::calibration::TargetAprilGrid(aprilgrid_config));

  aslam::calibration::DetectorAprilGrid::DetectorConfiguration
      aprilgrid_detector_config;
  aslam::calibration::DetectorAprilGrid aprilgird_detector(
      aprilgrid, aprilgrid_detector_config);

  // Load rosbag and connect to image topic.
  rosbag::Bag bag;
  bag.open(FLAGS_bag, rosbag::bagmode::Read);
  rosbag::View bag_view(bag, rosbag::TopicQuery(FLAGS_topic));

  // Iterate over all images of the topic and extract the calibration targets.
  std::vector<aslam::calibration::TargetObservation::Ptr> target_observations;
  target_observations.reserve(bag_view.size());

  std::vector<double> timestamps;

  // common::ProgressBar progress(bag_view.size());
  for (const rosbag::MessageInstance& message : bag_view) {
    sensor_msgs::ImageConstPtr image_message =
        message.instantiate<sensor_msgs::Image>();
    LOG_IF(FATAL, !image_message) << "Can only process image messages.";

    timestamps.push_back(image_message->header.stamp.toSec());

    // Convert image to cv::Mat.
    cv::Mat image;
    if (image_message->encoding == "16UC1") {
      sensor_msgs::Image img;
      img.header = image_message->header;
      img.height = image_message->height;
      img.width = image_message->width;
      img.is_bigendian = image_message->is_bigendian;
      img.step = image_message->step;
      img.data = image_message->data;
      img.encoding = "mono16";

      cv_bridge::CvImageConstPtr cv_ptr;
      try {
        cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
      } catch (const cv_bridge::Exception& e) {
        LOG(FATAL) << "cv_bridge exception: " << e.what();
      }
      CHECK(cv_ptr);
      image = cv_ptr->image;
    } else {
      cv_bridge::CvImageConstPtr cv_ptr;
      try {
        cv_ptr = cv_bridge::toCvShare(image_message,
                                      sensor_msgs::image_encodings::MONO8);
      } catch (const cv_bridge::Exception& e) {
        LOG(FATAL) << "cv_bridge exception: " << e.what();
      }
      CHECK(cv_ptr);
      image = cv_ptr->image;
    }

    // Extract calibration targets.
    aslam::calibration::TargetObservation::Ptr observation =
        aprilgird_detector.detectTargetInImage(image);

    // Only keep the observation if it was successful.
    target_observations.emplace_back(observation);

    // Optionally draw the extracted corners.
    if (FLAGS_draw_extraction) {
      cv::Mat image_w_corners;
      cv::cvtColor(image, image_w_corners, CV_GRAY2BGR);
      cv::namedWindow("extracted corners");

      if (observation.get() != nullptr) {
        // Detection successful.
        observation->drawCornersIntoImage(&image_w_corners);
      } else {
        // Detection failed.
        cv::putText(image_w_corners, "failed",
                    cv::Point(10, image_w_corners.rows / 2), 5, 5,
                    cv::Scalar(0, 0, 255), 5, 5);
      }
      cv::imshow("extracted corners", image_w_corners);
      cv::waitKey(1);
    }
  }
  LOG(INFO) << "Number of observations in total: " << target_observations.size()
            << " from " << bag_view.size() << " images.";

  // Optionally evaluate all the reprojection errors of all target
  // observations.
  if (!FLAGS_eval_camera_yaml.empty()) {
    const std::string kLogFilename = "corner_reprojections.csv";
    reprojection_logger.open(kLogFilename);
    if (FLAGS_write_csv_header) {
      reprojection_logger
          << "frame_id, corner_id, pnp_success, pnp_inlier, reproj_success, "
             "meas_u, meas_v, reproj_u, reproj_v"
          << std::endl;
      ;
    }

    camera_pose_logger.open(FLAGS_camera_pose_T_C_G_output_file);
    if (FLAGS_write_csv_header) {
      camera_pose_logger << "sequence_number, x [m], y[m], z[m], x, y, z, w"
                         << std::endl;
    }

    camera_pose_logger_timestamped.open(
        FLAGS_camera_pose_T_C_G_output_file_timestamped);
    if (FLAGS_write_csv_header) {
      camera_pose_logger_timestamped
          << "timestamp, x [m], y[m], z[m], x, y, z, w" << std::endl;
    }

    if (FLAGS_output_position_in_mm) {
      CHECK(!FLAGS_camera_pose_T_C_G_output_file_mm.empty());
      camera_pose_logger_mm.open(FLAGS_camera_pose_T_C_G_output_file_mm);
      if (FLAGS_write_csv_header) {
        camera_pose_logger_mm
            << "sequence_number, x [mm], y[mm], z[mm], x, y, z, w" << std::endl;
      }

      camera_pose_logger_mm_timestamped.open(
          FLAGS_camera_pose_T_C_G_output_file_mm_timestamped);
      if (FLAGS_write_csv_header) {
        camera_pose_logger_mm_timestamped
            << "timestamp, x [mm], y[mm], z[mm], x, y, z, w" << std::endl;
      }
    }

    aslam::Camera::ConstPtr camera =
        aslam::Camera::loadFromYaml(FLAGS_eval_camera_yaml);
    CHECK(camera);

    constexpr bool kRunNonlinearRefinement = true;
    const double kPixelSigma = 0.5;
    const int kMaxRansacIters = 1000;
    aslam::geometric_vision::PnpPoseEstimator pnp(kRunNonlinearRefinement);

    size_t number_observation_success = 0u;
    size_t number_pnp_success = 0u;
    size_t number_of_good_camera_poses = 0u;
    size_t frame_id = 0;
    CHECK_EQ(target_observations.size(), timestamps.size());
    for (const aslam::calibration::TargetObservation::Ptr& obs :
         target_observations) {
      if (obs) {
        number_observation_success++;
        const Eigen::Matrix2Xd& keypoints_measured = obs->getObservedCorners();
        const Eigen::Matrix3Xd G_corner_positions =
            obs->getCorrespondingTargetPoints();

        aslam::Transformation T_G_C;
        std::vector<int> inliers;
        int num_iters = 0;
        bool pnp_success = pnp.absolutePoseRansacPinholeCam(
            keypoints_measured, G_corner_positions, kPixelSigma,
            kMaxRansacIters, camera, &T_G_C, &inliers, &num_iters);

        const double number_inliers = inliers.size();
        const double number_keypoints = keypoints_measured.cols();
        const double inlier_ratio = number_inliers / number_keypoints;
        const bool sufficient_inliers =
            inlier_ratio > FLAGS_inlier_ratio_for_good_camera_pose;

        // Reproject the corners.
        Eigen::Matrix3Xd C_corner_positions =
            T_G_C.inverse().transformVectorized(G_corner_positions);

        // Only output the succesful poses.
        if (pnp_success) {
          ++number_pnp_success;
          if (sufficient_inliers) {
            ++number_of_good_camera_poses;

            Eigen::Matrix3d rotation_matrix = T_G_C.getRotationMatrix();
            Eigen::Quaterniond quaternion(rotation_matrix);
            if (quaternion.w() < 0.0) {
              quaternion.x() = -quaternion.x();
              quaternion.y() = -quaternion.y();
              quaternion.z() = -quaternion.z();
              quaternion.w() = -quaternion.w();
            }
            Eigen::Vector3d position = T_G_C.getPosition();
            CHECK_LT(frame_id, timestamps.size());
            constexpr double kMetersToMillimeters = 1000.0;

            camera_pose_logger
                << std::setprecision(std::numeric_limits<double>::max_digits10)
                << frame_id << ", " << position.x() << ", " << position.y()
                << ", " << position.z() << ", " << quaternion.x() << ", "
                << quaternion.y() << ", " << quaternion.z() << ", "
                << quaternion.w() << std::endl;
            camera_pose_logger_timestamped
                << std::setprecision(std::numeric_limits<double>::max_digits10)
                << timestamps[frame_id] << ", " << position.x() << ", "
                << position.y() << ", " << position.z() << ", "
                << quaternion.x() << ", " << quaternion.y() << ", "
                << quaternion.z() << ", " << quaternion.w() << std::endl;

            if (FLAGS_output_position_in_mm) {
              camera_pose_logger_mm
                  << std::setprecision(
                         std::numeric_limits<double>::max_digits10)
                  << frame_id << ", " << kMetersToMillimeters * position.x()
                  << ", " << kMetersToMillimeters * position.y() << ", "
                  << kMetersToMillimeters * position.z() << ", "
                  << quaternion.x() << ", " << quaternion.y() << ", "
                  << quaternion.z() << ", " << quaternion.w() << std::endl;
              camera_pose_logger_mm_timestamped
                  << std::setprecision(
                         std::numeric_limits<double>::max_digits10)
                  << timestamps[frame_id] << ", "
                  << kMetersToMillimeters * position.x() << ", "
                  << kMetersToMillimeters * position.y() << ", "
                  << kMetersToMillimeters * position.z() << ", "
                  << quaternion.x() << ", " << quaternion.y() << ", "
                  << quaternion.z() << ", " << quaternion.w() << std::endl;
            }
          }
        }

        Eigen::Matrix2Xd keypoints_reprojected;
        std::vector<aslam::ProjectionResult> projection_result;
        camera->project3Vectorized(C_corner_positions, &keypoints_reprojected,
                                   &projection_result);

        CHECK_EQ(obs->numObservedCorners(),
                 static_cast<size_t>(keypoints_measured.cols()));
        for (size_t idx = 0u; idx < obs->numObservedCorners(); ++idx) {
          const bool pnp_inlier =
              (std::find(inliers.begin(), inliers.end(),
                         static_cast<int>(idx)) != inliers.end());
          reprojection_logger
              << frame_id << ", " << obs->getObservedCornerIds()(idx) << ", "
              << pnp_success << ", " << pnp_inlier << ", "
              << projection_result[idx].isKeypointVisible() << ", "
              << keypoints_measured(0, idx) << ", "
              << keypoints_measured(1, idx) << ", "
              << keypoints_reprojected(0, idx) << ", "
              << keypoints_reprojected(1, idx) << std::endl;
        }
      }
      ++frame_id;
    }

    LOG(INFO) << "#########################################";
    LOG(INFO) << "Total # observations:     " << frame_id;
    LOG(INFO) << "      # successful:       " << number_observation_success;
    LOG(INFO) << "      # pnp successful:   " << number_pnp_success;
    LOG(INFO) << "      # inlier count ok:  " << number_of_good_camera_poses;
    LOG(INFO) << "Written reprojected corners to:    " << kLogFilename;
    LOG(INFO) << "Written extracted camera poses to: "
              << FLAGS_camera_pose_T_C_G_output_file_timestamped;
    LOG(INFO) << "#########################################";
  }
  reprojection_logger.close();
  camera_pose_logger.close();
  camera_pose_logger_timestamped.close();
  camera_pose_logger_mm.close();
  camera_pose_logger_mm_timestamped.close();

  return 0;
}
