#ifndef GPS_IMU_FUSION_ESKF_FLOW_H
#define GPS_IMU_FUSION_ESKF_FLOW_H

#include <deque>
#include <iostream>
#include <memory>

#include "eskf.hpp"
#include "gps_flow.h"
#include "imu_flow.h"
#include "observability_analysis.h"

class ESKFFlow {
 public:
  ESKFFlow() = default;
  ESKFFlow(const std::string &work_space_path);

  /*!
   * 从本地文件中读取IMU和GPS的数据
   * @return
   */
  bool ReadData();

  /*!
   * 对IMU和GPS数据进行时间戳对齐，该函数只在ESKF初始化时使用
   * @return
   */
  bool ValidGPSAndIMUData();

  bool ValidIMUData();

  bool ValidGPSData();

  bool Run();

  bool TestRun();

  /*!
   * 保存位姿，为kitti格式
   * @param ofs
   * @param pose
   */
  void SavePose(std::ofstream &ofs, const Eigen::Matrix4d &pose);

 private:
  std::shared_ptr<ESKF> eskf_ptr_;
  std::shared_ptr<IMUFlow> imu_flow_ptr_;
  std::shared_ptr<GPSFlow> gps_flow_ptr_;

  ObservabilityAnalysis observability_analysis;  // 可观测度分析工具

  std::deque<IMUData> imu_data_buff_;
  std::deque<GPSData> gps_data_buff_;

  IMUData curr_imu_data_;
  GPSData curr_gps_data_;

  bool use_observability_analysis_ = false;  // 是否进行可观测度分析

  const std::string work_space_path_;
};

int count = 0;
double total_cost = 0.0;

Eigen::Matrix4d Vector2Matrix(const Eigen::Vector3d &vec) {
  Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
  matrix.block<3, 1>(0, 3) = vec;

  return matrix;
}

ESKFFlow::ESKFFlow(const std::string &work_space_path)
    : work_space_path_(work_space_path) {
  std::string config_file_path = work_space_path_ + "/config/config.yaml";
  YAML::Node config_node = YAML::LoadFile(config_file_path);
  eskf_ptr_ = std::make_shared<ESKF>(config_node);
}

bool ESKFFlow::ValidGPSAndIMUData() {
  curr_imu_data_ = imu_data_buff_.front();
  curr_gps_data_ = gps_data_buff_.front();

  double delta_time = curr_imu_data_.time - curr_gps_data_.time;

  if (delta_time > 0.01) {
    gps_data_buff_.pop_front();
    return false;
  }

  if (delta_time < -0.01) {
    imu_data_buff_.pop_front();
    return false;
  }

  imu_data_buff_.pop_front();
  gps_data_buff_.pop_front();

  return true;
}

bool ESKFFlow::ValidIMUData() {
  curr_imu_data_ = imu_data_buff_.front();
  imu_data_buff_.front();

  return true;
}

bool ESKFFlow::ValidGPSData() {
  curr_gps_data_ = gps_data_buff_.front();
  gps_data_buff_.pop_front();

  return true;
}

bool ESKFFlow::Run() {
  while (!imu_data_buff_.empty() && !gps_data_buff_.empty()) {
    if (!ValidGPSAndIMUData()) {
      continue;
    } else {
      eskf_ptr_->Init(curr_gps_data_, curr_imu_data_);
      break;
    }
  }

  while (!imu_data_buff_.empty() && !gps_data_buff_.empty()) {
    std::chrono::time_point<std::chrono::system_clock> start =
        std::chrono::system_clock::now();

    curr_imu_data_ = imu_data_buff_.front();
    curr_gps_data_ = gps_data_buff_.front();
    if (curr_imu_data_.time < curr_gps_data_.time) {
      eskf_ptr_->Predict(curr_imu_data_);
      imu_data_buff_.pop_front();
    } else {
      eskf_ptr_->Predict(curr_imu_data_);
      imu_data_buff_.pop_front();

      eskf_ptr_->Correct(curr_gps_data_);

      gps_data_buff_.pop_front();

      std::chrono::time_point<std::chrono::system_clock> end =
          std::chrono::system_clock::now();
      std::chrono::duration<double> duration_ = end - start;
      count++;
      total_cost += duration_.count() * 1000;
      double average_cost = total_cost / count;
      // std::cout << count << " iter cost: " << double(duration_.count() *
      // 1000)
      //           << "ms    average cost: " << average_cost
      //           << "     total cost: " << total_cost << std::endl;
    }

    if (use_observability_analysis_) {
      Eigen::Matrix<double, 15, 15> F;
      Eigen::Matrix<double, 3, 15> G;
      Eigen::Matrix<double, 3, 1> Y;
      eskf_ptr_->GetFGY(F, G, Y);
      observability_analysis.SaveFG(F, G, Y, curr_gps_data_.time);
    }
  }

  if (use_observability_analysis_) {
    observability_analysis.ComputeSOM();
    observability_analysis.ComputeObservability();
  }
}

bool ESKFFlow::TestRun() {
  ReadData();

  while (!imu_data_buff_.empty() && !gps_data_buff_.empty()) {
    if (!ValidGPSAndIMUData()) {
      continue;
    } else {
      eskf_ptr_->Init(curr_gps_data_, curr_imu_data_);
      std::cout << "\ntime: " << curr_gps_data_.time << std::endl;
      std::cout << "vel: " << eskf_ptr_->GetVelocity().transpose() << std::endl;
      std::cout << "measure vel: " << curr_gps_data_.velocity.transpose()
                << std::endl;
      std::cout << "true vel: " << curr_gps_data_.true_velocity.transpose()
                << std::endl;
      std::cout << "time: " << curr_gps_data_.time << std::endl;
      break;
    }
  }

  std::ofstream gt_file(work_space_path_ + "/data/gt.txt", std::ios::trunc);
  std::ofstream fused_file(work_space_path_ + "/data/fused.txt",
                           std::ios::trunc);
  std::ofstream measured_file(work_space_path_ + "/data/measured.txt",
                              std::ios::trunc);

  while (!imu_data_buff_.empty() && !gps_data_buff_.empty()) {
    curr_imu_data_ = imu_data_buff_.front();
    curr_gps_data_ = gps_data_buff_.front();
    eskf_ptr_->Predict(curr_imu_data_);
    imu_data_buff_.pop_front();
    SavePose(fused_file, eskf_ptr_->GetPose());
  }
}

void ESKFFlow::SavePose(std::ofstream &ofs, const Eigen::Matrix4d &pose) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      ofs << pose(i, j);

      if (i == 2 && j == 3) {
        ofs << std::endl;
      } else {
        ofs << " ";
      }
    }
  }
}

#endif  // GPS_IMU_FUSION_ESKF_FLOW_H
