#include <Python.h>
#include <geometry_msgs/Vector3.h>
#include <ikd-Tree/ikd_Tree.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <omp.h>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <so3_math.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <unistd.h>
#include <visualization_msgs/Marker.h>

#include <Eigen/Core>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#include "GPS_Process.hpp"
#include "eskf.hpp"

#define INIT_TIME (0.1)
#define GPS_COV (0.1)

// DEBUG
std::string time_str;
int meas_num = 0;

std::mutex mtx_buffer;
condition_variable sig_buffer;

ofstream fout_pre, fout_out, fout_dbg;
double solve_time = 0, solve_const_H_time = 0;
double res_mean_last = 0.05, total_residual = 0.0;  // 设置残差平均值，残差总和
double gyr_cov = 1.0e-5, acc_cov = 1.0e-4, b_gyr_cov = 0.0001,
       b_acc_cov = 0.0001;
// double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;

// double prior_cov_pos = 1.0e-4, prior_cov_vel = 1.0e-4, prior_cov_ori
// = 1.0e-6,
//        prior_cov_epsilon = 1.0e-6, prior_cov_delta = 1.0e-6;
// double meas_cov_pos = 10;
// bool flg_eskf_init = false;

int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0,
    laserCloudValidNum = 0, count_ = 0;
double last_timestamp_imu = -1.0, last_timestamp_gps = 0, first_gps_time = 0.0,
       gps_curr_time = 0.0;
deque<double> time_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<V4D> gps_buffer;
bool flg_first_gps = true, path_en = true, flg_EKF_inited;
std::string imu_topic, gps_topic;
bool TRANSAXIS = true;

vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);

// GPS with respect to IMU
V3D GPS_T_wrt_IMU(Zero3d);
M3D GPS_R_wrt_IMU(Eye3d);
V3D euler_cur;
V3D res_pos;
Eigen::Quaterniond res_quat(0, 0, 0, 1);

// shared_ptr<ImuProcess> imu_proc(new ImuProcess());
shared_ptr<GPSProcess> gps_proc(new GPSProcess());
// shared_ptr<ESKF> eskf_proc(new ESKF());
ESKF eskf_proc;
// pointcloud & imu msg
MeasureGroup Measures;
// 状态，噪声维度，输入
// esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
// state_ikfom state_point;  // 状态
// world系下lidar坐标
vect3 pos_lid;

// OUTPUT
nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

void NED2ENU(const sensor_msgs::Imu::ConstPtr &imu_in,
             sensor_msgs::Imu::Ptr &imu_out) {
  double kDegree2Radian = M_PI / 180.0;

  Eigen::Quaterniond Q_b_w =
      Eigen::AngleAxisd(90 * kDegree2Radian, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(0 * kDegree2Radian, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(180 * kDegree2Radian, Eigen::Vector3d::UnitX());
  Eigen::Vector3d vec_acc;
  Eigen::Vector3d vec_gyro;
  vec_acc.x() = imu_in->linear_acceleration.x;
  vec_acc.y() = imu_in->linear_acceleration.y;
  vec_acc.z() = imu_in->linear_acceleration.z;
  vec_gyro.x() = imu_in->angular_velocity.x;
  vec_gyro.y() = imu_in->angular_velocity.y;
  vec_gyro.z() = imu_in->angular_velocity.z;

  // std::string write_path1 =
  //     "/home/xng/catkin_ws/src/inno_lio/data/res/ang_vel_origin.txt";
  // std::ofstream outfile1;
  // outfile1.open(write_path1, std::ofstream::app);
  // outfile1 << meas_num << " " << vec_gyro.x() << " " << vec_gyro.y() << " "
  //          << vec_gyro.z() << " " << 0 << " " << 0 << " " << 0 << " " << 1
  //          << std::endl;
  // outfile1.close();

  Eigen::Vector3d vec_acc_out = Q_b_w.inverse() * vec_acc;
  Eigen::Vector3d vec_gyro_out = Q_b_w.inverse() * vec_gyro;

  // Eigen::Vector3d vec_acc_out = vec_acc;
  // Eigen::Vector3d vec_gyro_out = vec_gyro;

  imu_out->header.stamp = imu_in->header.stamp;
  imu_out->header.frame_id = imu_in->header.frame_id;
  imu_out->linear_acceleration.x = vec_acc_out.x();
  imu_out->linear_acceleration.y = vec_acc_out.y();
  imu_out->linear_acceleration.z = vec_acc_out.z();
  // deg -->rad
  imu_out->angular_velocity.x = vec_gyro_out.x() * 0.017453292519943295;
  imu_out->angular_velocity.y = vec_gyro_out.y() * 0.017453292519943295;
  imu_out->angular_velocity.z = vec_gyro_out.z() * 0.017453292519943295;

  //* nearly right
  // imu_out->angular_velocity.x = vec_gyro.y() * 0.017453292519943295;
  // imu_out->angular_velocity.y = vec_gyro.x() * 0.017453292519943295;
  // imu_out->angular_velocity.z = -vec_gyro.z() * 0.017453292519943295;

  // std::string write_path2 =
  //     "/home/xng/catkin_ws/src/inno_lio/data/res/ang_vel_test.txt";
  // std::ofstream outfile2;
  // outfile2.open(write_path2, std::ofstream::app);
  // outfile2 << meas_num << " " << imu_out->angular_velocity.x << " "
  //          << imu_out->angular_velocity.y << " " <<
  //          imu_out->angular_velocity.z
  //          << " " << 0 << " " << 0 << " " << 0 << " " << 1 << std::endl;
  // outfile2.close();
  // meas_num++;
}

void gps_cbk(const sensor_msgs::NavSatFix::ConstPtr &gps_msg) {
  sensor_msgs::NavSatFix::Ptr msg(new sensor_msgs::NavSatFix(*gps_msg));
  double timestamp = gps_msg->header.stamp.toSec();
  mtx_buffer.lock();

  if (timestamp < last_timestamp_gps) {
    ROS_WARN("gps loop back, clear buffer");
    gps_buffer.clear();
  }
  V4D temp_utm;
  // to utm
  if (!gps_proc->proj_init) {
    gps_proc->Initialize(gps_msg->longitude, gps_msg->latitude);
  }
  gps_proc->Process(gps_msg, temp_utm);

  last_timestamp_gps = timestamp;
  time_buffer.push_back(timestamp);
  gps_buffer.push_back(temp_utm);
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  double timestamp = msg->header.stamp.toSec();
  mtx_buffer.lock();

  if (timestamp < last_timestamp_imu) {
    ROS_WARN("imu loop back, clear buffer");
    imu_buffer.clear();
  }
  last_timestamp_imu = timestamp;
  sensor_msgs::Imu::Ptr temp_imu(new sensor_msgs::Imu(*msg_in));
  if (1) {
    NED2ENU(msg_in, temp_imu);
  } else {
    temp_imu = msg;
  }

  imu_buffer.push_back(temp_imu);
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

// 离GPS时间点最近的IMU数据从缓存队列中取出，进行时间对齐，并保存到meas中
bool sync_packages(MeasureGroup &meas) {
  if (imu_buffer.empty() || gps_buffer.empty()) {
    return false;
  }
  double gps_time = 0.0;
  /*** push gps data, and pop from imu buffer ***/
  meas.gps.clear();

  while ((!gps_buffer.empty())) {
    gps_time = gps_buffer.front()[0];
    meas.gps.push_back(gps_buffer.front());
    gps_buffer.pop_front();
  }
  gps_curr_time = gps_time;
  /*** push imu data, and pop from imu buffer ***/
  double imu_time = imu_buffer.front()->header.stamp.toSec();
  meas.imu.clear();
  while ((!imu_buffer.empty()) && (imu_time < gps_time)) {
    imu_time = imu_buffer.front()->header.stamp.toSec();
    if (imu_time > gps_time) break;

    sensor_msgs::Imu::Ptr temp_imu(new sensor_msgs::Imu(*imu_buffer.front()));
    // NED2ENU(imu_buffer.front(), temp_imu);

    meas.imu.push_back(temp_imu);
    imu_buffer.pop_front();
    // }
  }
  // std::cout << "-----------" << std::endl;
  std::cout << " meas.imu.size: " << meas.imu.size()
            << "    meas.gps.size: " << meas.gps.size() << std::endl;
  // std::string write_path = "/home/xng/catkin_ws/src/inno_lio/data/res/meas_"
  // +
  //                          std::to_string(meas_num) + ".txt";
  // std::ofstream outfile;
  // outfile.open(write_path, std::ofstream::app);
  // outfile << "GPS: " << meas.gps.size() << " " << meas.gps.back()[0] << " "
  //         << meas.gps.back()[1] << " " << meas.gps.back()[2] << " "
  //         << meas.gps.back()[3] << "  IMU: " << meas.imu.size() << " "
  //         << meas.imu.back()->header.stamp << " "
  //         << meas.imu.back()->linear_acceleration.x << " "
  //         << meas.imu.back()->angular_velocity.x << std::endl;
  // outfile.close();
  // meas_num++;

  time_buffer.pop_front();
  return true;
}

void h_shared_model_GIO(state_ikfom &s,
                        esekfom::dyn_share_datastruct<double> &ekfom_data) {
  ekfom_data.h_x = MatrixXd::Zero(3, 12);
  ekfom_data.h.resize(3);
  // ekfom_data.h_x.block<3, 3>(0, 9) = Eigen::Matrix3d::Identity();
  ekfom_data.h_x.block<1, 12>(0, 0) << 1, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0;
  ekfom_data.h_x.block<1, 12>(1, 0) << 0, 1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0;
  ekfom_data.h_x.block<1, 12>(2, 0) << 0, 0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0;
  // TODO======CHECK======
  for (int i = 0; i < 3; i++) {
    //* predict pos - utm pos
    ekfom_data.h(i) = Measures.gps.front()[i + 1] - s.pos(i);
  }
  // std::cout << "s.pos(i):  " << s.pos(0) << " " << s.pos(1) << " " <<
  // s.pos(2)
  //           << std::endl;
  // std::cout << "Measures.gps: " << Measures.gps.front()[1] << " "
  //           << Measures.gps.front()[2] << " " << Measures.gps.front()[3]
  //           << std::endl;
  // std::cout << "ekfom_data.h(i): " << ekfom_data.h(0) << " " <<
  // ekfom_data.h(1)
  //           << " " << ekfom_data.h(2) << std::endl;
  // std::cout << "ekfom_data.h_x: " << ekfom_data.h_x << std::endl;
}

// pub pose
template <typename T>
void set_posestamp(T &out) {
  out.pose.position.x = res_pos[0];
  out.pose.position.y = res_pos[1];
  out.pose.position.z = res_pos[2];
  out.pose.orientation.x = res_quat.x();
  out.pose.orientation.y = res_quat.y();
  out.pose.orientation.z = res_quat.z();
  out.pose.orientation.w = res_quat.w();
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped) {
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "body";
  odomAftMapped.header.stamp = ros::Time().fromSec(
      gps_curr_time);  // ros::Time().fromSec(lidar_end_time);
  set_posestamp(odomAftMapped.pose);
  pubOdomAftMapped.publish(odomAftMapped);
  // auto P = kf.get_P();
  // for (int i = 0; i < 6; i++) {
  //   int k = i < 3 ? i + 3 : i - 3;
  //   odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
  //   odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
  //   odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
  //   odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
  //   odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
  //   odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
  // }
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                  odomAftMapped.pose.pose.position.y,
                                  odomAftMapped.pose.pose.position.z));
  q.setW(odomAftMapped.pose.pose.orientation.w);
  q.setX(odomAftMapped.pose.pose.orientation.x);
  q.setY(odomAftMapped.pose.pose.orientation.y);
  q.setZ(odomAftMapped.pose.pose.orientation.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp,
                                        "camera_init", "body"));
}

void publish_path(const ros::Publisher pubPath) {
  set_posestamp(msg_body_pose);
  msg_body_pose.header.stamp = ros::Time().fromSec(gps_curr_time);
  msg_body_pose.header.frame_id = "camera_init";

  /*** if path is too large, the rvis will crash ***/
  static int jjj = 0;
  jjj++;
  if (jjj % 10 == 0) {
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;

  time_t rawtime = time(NULL);
  struct tm *timeinfo = localtime(&rawtime);
  char str_time[100];
  sprintf(str_time, "%04d%02d%02d%02d%02d", timeinfo->tm_year + 1900,
          timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour,
          timeinfo->tm_min);
  std::string temp_str(str_time);
  time_str = temp_str;
  nh.param<bool>("publish/path_en", path_en, true);
  nh.param<int>("max_iteration", NUM_MAX_ITERATIONS,
                4);  // 卡尔曼滤波的最大迭代次数
  nh.param<std::string>("common/imu_topic", imu_topic, "/imu");
  nh.param<std::string>("common/gps_topic", gps_topic, "/gps");
  nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";

  //* 1. 设置雷达到IMU的外参，状态变量的协方差
  GPS_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
  GPS_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
  //* 2. 传入函数初始化
  // kf.init_dyn_share_gio(get_f_gio, df_dx, df_dw, h_shared_model_GIO,
  //                       NUM_MAX_ITERATIONS, epsi);

  ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
  ros::Subscriber sub_gps = nh.subscribe(gps_topic, 200000, gps_cbk);
  ros::Publisher pubOdomAftMapped =
      nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
  ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

  ros::Rate rate(5000);
  bool status = ros::ok();
  size_t point_num = 0;
  int count_ = 0;
  int pred_num = 0;
  int correct_num = 0;
  int debug_num = 0;

  while (status) {
    auto clock1 = std::chrono::steady_clock::now();
    ros::spinOnce();
    //* 3. 对齐传感器输入，存入Measures，
    // if (sync_packages(Measures)) {
    // std::chrono::time_point<std::chrono::system_clock> start =
    //     std::chrono::system_clock::now();
    // while (!Measures.imu.empty() && !Measures.gps.empty()) {
    while (!imu_buffer.empty() && !gps_buffer.empty()) {
      // sensor_msgs::Imu::Ptr curr_imu_data(
      //     new sensor_msgs::Imu(*Measures.imu.front()));
      // V4D curr_gps_data = Measures.gps.front();
      sensor_msgs::Imu::Ptr curr_imu_data(
          new sensor_msgs::Imu(*imu_buffer.front()));
      V4D curr_gps_data = gps_buffer.front();

      // ===============DEBUG===============
      // std::string write_path_1 =
      //     "/home/xng/catkin_ws/src/inno_ligo/data/res/acc_in.txt";
      // std::ofstream outfile_1;
      // outfile_1.open(write_path_1, std::ofstream::app);
      // outfile_1 << debug_num << " " << curr_imu_data->linear_acceleration.x
      //           << " " << curr_imu_data->linear_acceleration.y << " "
      //           << curr_imu_data->linear_acceleration.z << " " << 0 << " " <<
      //           0
      //           << " " << 0 << " " << 1 << std::endl;
      // outfile_1.close();

      // std::string write_path_2 =
      //     "/home/xng/catkin_ws/src/inno_ligo/data/res/gps_in.txt";
      // std::ofstream outfile_2;
      // outfile_2.open(write_path_2, std::ofstream::app);
      // outfile_2 << debug_num << " " << curr_gps_data[1] << " "
      //           << curr_gps_data[2] << " " << curr_gps_data[3] << " " << 0
      //           << " " << 0 << " " << 0 << " " << 1 << std::endl;
      // outfile_2.close();

      debug_num++;
      // ====================================

      if (!eskf_proc.flg_eskf_init) {
        eskf_proc.Init(curr_imu_data);
      }
      // imu_buffer.empty() || gps_buffer

      if (curr_imu_data->header.stamp.toSec() < curr_gps_data[0]) {
        std::cout << "---gps stamp: " << curr_gps_data[0]
                  << "  imu_stamp: " << curr_imu_data->header.stamp.toSec()
                  << std::endl;
        eskf_proc.Predict(curr_imu_data);
        // eskf_proc->Predict(curr_imu_data);
        imu_buffer.pop_front();
        pred_num++;
      } else {
        std::cout << "===gps stamp: " << curr_gps_data[0]
                  << "  imu_stamp: " << curr_imu_data->header.stamp.toSec()
                  << std::endl;
        correct_num++;
        eskf_proc.Predict(curr_imu_data);
        // eskf_proc->Predict(curr_imu_data);
        imu_buffer.pop_front();
        eskf_proc.Correct(curr_gps_data);
        // eskf_proc->Correct(curr_gps_data);
        gps_buffer.pop_front();

        eskf_proc.GetPose(res_pos, res_quat);
        // eskf_proc->GetPose(res_pos, res_quat);
        /******* Publish odometry *******/
        publish_odometry(pubOdomAftMapped);
        /******* Publish path *******/
        if (path_en) publish_path(pubPath);

        std::string write_path = "/home/xng/catkin_ws/src/inno_ligo/data/res/" +
                                 time_str + "_path.txt";
        std::ofstream outfile;
        outfile.open(write_path, std::ofstream::app);
        outfile << count_ << " " << res_pos[0] << " " << res_pos[1] << " "
                << res_pos[2] << " " << res_quat.x() << " " << res_quat.y()
                << " " << res_quat.z() << " " << res_quat.w() << std::endl;
        outfile.close();
        count_++;

        std::cout << " pred_num: " << pred_num
                  << "  correct_num: " << correct_num << std::endl;
      }

      // if (Measures.imu.size() > 1) {
      //   std::cout << "---gps stamp: " << curr_gps_data[0]
      //             << "  imu_stamp: " << curr_imu_data->header.stamp.toSec()
      //             << std::endl;
      //   eskf_proc.Predict(curr_imu_data);
      //   // eskf_proc->Predict(curr_imu_data);
      //   Measures.imu.pop_front();
      //   pred_num++;
      // } else if (Measures.imu.size() == 1) {
      //   std::cout << "===gps stamp: " << curr_gps_data[0]
      //             << "  imu_stamp: " << curr_imu_data->header.stamp.toSec()
      //             << std::endl;
      //   correct_num++;
      //   eskf_proc.Predict(curr_imu_data);
      //   // eskf_proc->Predict(curr_imu_data);
      //   Measures.imu.pop_front();
      //   eskf_proc.Correct(curr_gps_data);
      //   // eskf_proc->Correct(curr_gps_data);
      //   Measures.gps.pop_front();

      //   eskf_proc.GetPose(res_pos, res_quat);
      //   // eskf_proc->GetPose(res_pos, res_quat);
      //   /******* Publish odometry *******/
      //   publish_odometry(pubOdomAftMapped);
      //   /******* Publish path *******/
      //   if (path_en) publish_path(pubPath);

      //   std::string write_path =
      //       "/home/xng/catkin_ws/src/inno_ligo/data/res/" + time_str +
      //       "_path.txt";
      //   std::ofstream outfile;
      //   outfile.open(write_path, std::ofstream::app);
      //   outfile << count_ << " " << res_pos[0] << " " << res_pos[1] << " "
      //           << res_pos[2] << " " << res_quat.x() << " " << res_quat.y()
      //           << " " << res_quat.z() << " " << res_quat.w() << std::endl;
      //   outfile.close();
      //   count_++;

      //   std::cout << " pred_num: " << pred_num
      //             << "  correct_num: " << correct_num << std::endl;
      // }
      // }

      // auto clock2 = std::chrono::steady_clock::now();
      // double ct =
      //     std::chrono::duration_cast<std::chrono::microseconds>(clock2 -
      //     clock1)
      //         .count() /
      //     1000.0;
      // std::cout << "Time cost(ms): " << ct << std::endl;
    }
    status = ros::ok();
    rate.sleep();
  }
  return 0;
}
