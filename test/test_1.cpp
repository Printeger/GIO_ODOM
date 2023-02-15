#include <gtest/gtest.h>
#include <ros/ros.h>

// #include <memory>

#include "../src/GPS_Process.hpp"

TEST(TestSuite, testCase1) {
  V4D temp_utm;
  std::shared_ptr<GPSProcess> gps_proc(new GPSProcess());
  sensor_msgs::NavSatFix::Ptr origin_msg(new sensor_msgs::NavSatFix());
  sensor_msgs::NavSatFix::Ptr msg(new sensor_msgs::NavSatFix());

  //   //   msg->header.stamp.now();
  origin_msg->longitude = 125;
  origin_msg->latitude = 35;
  msg->longitude = 125;
  msg->latitude = 35;
  if (!gps_proc->proj_init) {
    gps_proc->Initialize(origin_msg->longitude, origin_msg->latitude);
  }
  gps_proc->Process(msg, temp_utm);
  std::cout << "UTM: " << temp_utm << std::endl;

  EXPECT_TRUE(true);
}

int main(int argc, char **argv) {
  //   testing::GTEST_FLAG(output) = "xml:/home/jd/";
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "test_node");
  ros::NodeHandle nh;

  return RUN_ALL_TESTS();
}