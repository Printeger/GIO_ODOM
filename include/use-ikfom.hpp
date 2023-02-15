#ifndef USE_IKFOM_H
#define USE_IKFOM_H

#include <IKFoM_toolkit/esekfom/esekfom.hpp>

// typedef MTK::vect<3, double> vect3;
// typedef MTK::SO3<double> SO3;
// typedef MTK::S2<double, 98090, 10000, 1> S2;
// typedef MTK::vect<1, double> vect1;
// typedef MTK::vect<2, double> vect2;

/*
        xk = A * xk-1 + B * u + w
*/

MTK_BUILD_MANIFOLD(state_ikfom,             // state vector
                   ((vect3, pos))           // position_imu2global
                   ((SO3, rot))             // rotation_imu2global
                   ((SO3, offset_R_L_I))    // rotation_lidar2imu
                   ((vect3, offset_T_L_I))  // translation_lidar2imu
                   ((vect3, vel))           // velocity
                   ((vect3, bg))            // bias_gyro
                   ((vect3, ba))            // bias_acc
                   ((S2, grav))             // gravity
);

MTK_BUILD_MANIFOLD(input_ikfom,     // u
                   ((vect3, acc))   // acc
                   ((vect3, gyro))  // gyro
);

MTK_BUILD_MANIFOLD(process_noise_ikfom,  // noise
                   ((vect3, ng))         // gyro noise
                   ((vect3, na))         // acc noise
                   ((vect3, nbg))        // gyro bias's noise
                   ((vect3, nba))        // acc bias's noise
);

MTK::get_cov<process_noise_ikfom>::type process_noise_cov() {
  MTK::get_cov<process_noise_ikfom>::type cov =
      MTK::get_cov<process_noise_ikfom>::type::Zero();
  MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng,
                                                  0.0001);  // 0.03
  MTK::setDiagonal<process_noise_ikfom, vect3, 3>(
      cov, &process_noise_ikfom::na, 0.0001);  // *dt 0.01 0.01 * dt * dt 0.05
  MTK::setDiagonal<process_noise_ikfom, vect3, 6>(
      cov, &process_noise_ikfom::nbg,
      0.00001);  // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
  MTK::setDiagonal<process_noise_ikfom, vect3, 9>(
      cov, &process_noise_ikfom::nba, 0.00001);  // 0.001 0.05 0.0001/out 0.01
  return cov;
}

//*名义状态：f(x, u, w) = [v[3] w[3] 0 0 0 0 0 0 a[3] 0 0 0 0 0 0 0 0 0]T (24*1)
// get flatted_state res (state need propogation)
Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in) {
  // 将imu积分方程矩阵初始化为0,这里的24个对应了速度(3)，角速度(3),外参偏置T(3),外参偏置R(3)，加速度(3),角速度偏置(3),加速度偏置(3),位置(3)
  Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
  // imu角速度 gyro_gt = in.gyro - s.gyro_bias
  vect3 omega;
  in.gyro.boxminus(omega, s.bg);
  // 世界坐标系加速度: acc_gt = rot_imu2global * (acc - acc_bias)
  vect3 a_inertial = s.rot * (in.acc - s.ba);
  // vect3 a_inertial = s.rot * in.acc;
  // 更新速度/角速度/加速度
  for (int i = 0; i < 3; i++) {
    res(i) = s.vel[i];
    res(i + 3) = omega[i];
    res(i + 12) = a_inertial[i] + s.grav[i];
  }
  return res;
}

Eigen::Matrix<double, 24, 1> get_f_gio(state_ikfom &s, const input_ikfom &in,
                                       vect3 &bg_gyro, vect3 &bg_acc) {
  // 将imu积分方程矩阵初始化为0,这里的24个对应了速度(3)，角速度(3),外参偏置T(3),外参偏置R(3)，加速度(3),角速度偏置(3),加速度偏置(3),位置(3)
  Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
  // imu角速度 gyro_gt = in.gyro - s.gyro_bias
  // omega = in.gyro - average_gyro = rot_in_dt
  vect3 omega;
  vect3 a_inertial;
  if (1) {
    in.gyro.boxminus(omega, s.bg);
    // omega = -bg_gyro;
    // 世界坐标系加速度: acc_gt = rot_imu2global * (acc - acc_bias)
    // a_inertial = s.rot * (in.acc - bg_acc);
    a_inertial = s.rot * (in.acc - s.ba);
    // a_inertial = in.acc - bg_acc;
  } else {
    in.gyro.boxminus(omega, s.bg);
    a_inertial = s.rot * (in.acc - s.ba);
  }
  // 更新速度/角速度/加速度
  for (int i = 0; i < 3; i++) {
    res(i) = s.vel[i];
    res(i + 3) = omega[i];
    res(i + 12) = a_inertial[i] + s.grav[i];
  }
  Eigen::Quaterniond quat_(s.rot.w(), s.rot.x(), s.rot.y(), s.rot.z());
  Eigen::Vector3d eua = quat_.matrix().eulerAngles(2, 1, 0);
  // std::cout << "===============================================" <<
  // std::endl; std::cout << "in.gyro: " << in.gyro << std::endl; std::cout <<
  // "bg_gyro: " << bg_gyro << std::endl; std::cout << "s.bg: " << s.bg <<
  // std::endl; std::cout << "omega: " << omega << std::endl; std::cout <<
  // "s.rot: " << eua.transpose() * 57.29578049 << std::endl; std::cout <<
  // "------------------------------------------------" << std::endl; std::cout
  // << "in.acc: " << in.acc << std::endl; std::cout << "bg_acc: " << bg_acc <<
  // std::endl; std::cout << "a_inertial: " << a_inertial << std::endl;
  // std::cout << "s.grav: " << s.grav << std::endl;
  // std::cout << "===============================================" <<
  // std::endl;

  return res;
}

//*更新矩阵df_dx   [ delta_x(i) = df_dx * delta_x(i-1) + df_dw * bias_(i-1) ]
Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in) {
  Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();
  cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
  vect3 acc_;
  in.acc.boxminus(acc_, s.ba);
  vect3 omega;
  in.gyro.boxminus(omega, s.bg);
  cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix() * MTK::hat(acc_);
  cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
  Eigen::Matrix<state_ikfom::scalar, 2, 1> vec =
      Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
  Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
  s.S2_Mx(grav_matrix, vec, 21);
  cov.template block<3, 2>(12, 21) = grav_matrix;
  cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
  return cov;
}

//*更新矩阵df_dw   [ delta_x(i) = df_dx * delta_x(i-1) + df_dw * bias_(i-1) ]
Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in) {
  Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
  cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();
  cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
  cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
  cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
  return cov;
}

vect3 SO3ToEuler(const SO3 &orient) {
  Eigen::Matrix<double, 3, 1> _ang;
  Eigen::Vector4d q_data = orient.coeffs().transpose();
  // scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1],
  // z=orient.coeffs[2];
  double sqw = q_data[3] * q_data[3];
  double sqx = q_data[0] * q_data[0];
  double sqy = q_data[1] * q_data[1];
  double sqz = q_data[2] * q_data[2];
  double unit = sqx + sqy + sqz +
                sqw;  // if normalized is one, otherwise is correction factor
  double test = q_data[3] * q_data[1] - q_data[2] * q_data[0];

  if (test > 0.49999 * unit) {  // singularity at north pole

    _ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI / 2, 0;
    double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
    vect3 euler_ang(temp, 3);
    return euler_ang;
  }
  if (test < -0.49999 * unit) {  // singularity at south pole
    _ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI / 2, 0;
    double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
    vect3 euler_ang(temp, 3);
    return euler_ang;
  }

  _ang << std::atan2(2 * q_data[0] * q_data[3] + 2 * q_data[1] * q_data[2],
                     -sqx - sqy + sqz + sqw),
      std::asin(2 * test / unit),
      std::atan2(2 * q_data[2] * q_data[3] + 2 * q_data[1] * q_data[0],
                 sqx - sqy - sqz + sqw);
  double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
  vect3 euler_ang(temp, 3);
  // euler_ang[0] = roll, euler_ang[1] = pitch, euler_ang[2] = yaw
  return euler_ang;
}

#endif