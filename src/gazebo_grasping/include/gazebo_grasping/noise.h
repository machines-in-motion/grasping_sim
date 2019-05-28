#ifndef __GAZEBO_GRASPING__NOISE_H__
#define __GAZEBO_GRASPING__NOISE_H__

#include <ignition/math.hh>
#include <random>

class Noise {
 public:
  Noise();
  Noise(double noise_pos, double noise_orient);

  bool is_initialized();
  ignition::math::Pose3d random_pose();

 private:
  bool initialized_;
  std::mt19937 rng_;
  std::normal_distribution<double> normal_;
  std::uniform_real_distribution<double> uniform01_;
  double noise_pos_;
  double noise_orient_;
};

#endif  // __GAZEBO_GRASPING__NOISE_H__
