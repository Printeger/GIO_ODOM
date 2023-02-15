#include <common_lib.h>
#include <proj_api.h>
#include <sensor_msgs/NavSatFix.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "use-ikfom.hpp"
class PJTransformer {
 public:
  explicit PJTransformer(double lon_0, double lat_0) {
    // init projPJ
    std::stringstream stream;
    // stream << "+proj=utm +zone=" << zone_id << " +ellps=WGS84" << std::endl;
    // stream << "+proj=tmerc +k=0.9996 +lon_0=" << lon_0 << "e +ellps=WGS84"
    //        << std::endl;
    stream << "+proj=tmerc +k=0.9996 +lon_0=" << lon_0 << " +lat_0= " << lat_0
           << "e +ellps=WGS84" << std::endl;
    pj_utm_ = pj_init_plus(stream.str().c_str());
    // pj_utm_ = pj_init_plus_ctx();
    if (pj_utm_ == nullptr) {
      std::cout << "proj4 init failed!" << stream.str() << std::endl;
      return;
    }
    pj_latlong_ = pj_init_plus("+proj=latlong +ellps=WGS84");
    if (pj_latlong_ == nullptr) {
      std::cout << "proj4 pj_latlong init failed!";
      return;
    }
    std::cout << "proj4 init success" << std::endl;
  }

  ~PJTransformer() {
    if (pj_latlong_) {
      pj_free(pj_latlong_);
      pj_latlong_ = nullptr;
    }
    if (pj_utm_) {
      pj_free(pj_utm_);
      pj_utm_ = nullptr;
    }
  }

  int LatlonToUtm(int64_t point_count, int point_offset, double *x, double *y,
                  double *z) {
    if (!pj_latlong_ || !pj_utm_) {
      std::cout << "pj_latlong_:" << pj_latlong_ << "pj_utm_:" << pj_utm_
                << std::endl;
      return -1;
    }
    return pj_transform(pj_latlong_, pj_utm_, point_count, point_offset, x, y,
                        z);
  }

 private:
  projPJ pj_latlong_;
  projPJ pj_utm_;
};

class GPSProcess {
 public:
  GPSProcess();
  ~GPSProcess();

  // gps 2 utm
  // set_gps_cov()
  void Initialize(double lon_init, double lat_init);
  void Reset();
  void set_gps_cov(const V3D &scaler);
  void set_extrinsic(const MD(4, 4) & T);
  void set_extrinsic(const V3D &trans);
  void set_extrinsic(const V3D &trans, const M3D &rot);
  void LLA2UTM(const V3D &lla_, V3D &utm_);
  void Process(const sensor_msgs::NavSatFix::ConstPtr &msg, V4D &gps_out);
  sensor_msgs::NavSatFixPtr last_gps_;
  std::vector<V3D> GPS_position;
  double timestamp;
  V3D LLA;  // lon, lat, alt
  V3D UTM;
  V3D cov_gps;
  M3D GPS_R_wrt_IMU;  // GPS to IMU extrinsic
  V3D GPS_T_wrt_IMU;
  std::shared_ptr<PJTransformer> proj_ = nullptr;
  // PJTransformer proj_;
  bool proj_init = false;
  double lon_0 = 120.0;
  double lat_0 = 30.0;
};

GPSProcess::GPSProcess() {
  LLA = V3D(0, 0, 0);
  UTM = V3D(0, 0, 0);
  cov_gps = V3D(0.1, 0.1, 0.1);
  GPS_R_wrt_IMU = Eye3d;
  GPS_T_wrt_IMU = Zero3d;
  last_gps_.reset(new sensor_msgs::NavSatFix());
}

GPSProcess::~GPSProcess() {}

void GPSProcess::Reset() {
  LLA = V3D(0, 0, 0);
  UTM = V3D(0, 0, 0);

  GPS_position.clear();
  last_gps_.reset(new sensor_msgs::NavSatFix());
}

void GPSProcess::set_extrinsic(const MD(4, 4) & T) {
  GPS_T_wrt_IMU = T.block<3, 1>(0, 3);
  GPS_R_wrt_IMU = T.block<3, 3>(0, 0);
}

void GPSProcess::set_extrinsic(const V3D &trans) {
  GPS_T_wrt_IMU = trans;
  GPS_R_wrt_IMU.setIdentity();
}

void GPSProcess::set_extrinsic(const V3D &trans, const M3D &rot) {
  GPS_T_wrt_IMU = trans;
  GPS_R_wrt_IMU = rot;
}

void GPSProcess::set_gps_cov(const V3D &scaler) { cov_gps = scaler; }

void GPSProcess::LLA2UTM(const V3D &lla_, V3D &utm_) {
  double lon_, lat_, alt_;
  lat_ = lla_[0] / 180 * M_PI;
  lon_ = lla_[1] / 180 * M_PI;
  alt_ = lla_[2];
  proj_->LatlonToUtm(1, 1, &lon_, &lat_, &alt_);
  // proj_->LatlonToUtm(1, 1, &lat_, &lon_, &alt_);

  utm_[0] = lon_;
  utm_[1] = lat_;
  utm_[2] = alt_;
}
//  initialize proj zero point as the first coming GPS position
void GPSProcess::Initialize(double lon_init, double lat_init) {
  if (!proj_init) {
    lon_0 = lon_init;
    lat_0 = lat_init;
    proj_init = true;
  }
  // 初始化UTM原点坐标，这里 lat/lon 顺序不影响
  proj_ = std::make_shared<PJTransformer>(lon_0, lat_0);
  std::cout << "=====UTM initialize DONE=====" << std::endl;
}
void GPSProcess::Process(const sensor_msgs::NavSatFix::ConstPtr &msg,
                         V4D &gps_out) {
  timestamp = msg->header.stamp.toSec();
  LLA[0] = msg->latitude;
  LLA[1] = msg->longitude;
  LLA[2] = msg->altitude;
  LLA2UTM(LLA, UTM);
  gps_out = V4D(timestamp, UTM[0], UTM[1], UTM[2]);
}