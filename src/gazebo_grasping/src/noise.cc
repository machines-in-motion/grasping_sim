#include <cmath>

#include "gazebo_grasping/noise.h"

Noise::Noise() : initialized_(false) {}

Noise::Noise(double noise_pos, double noise_orient)
    : initialized_(true), uniform01_(0, 1.0), noise_pos_(noise_pos), noise_orient_(noise_orient) {
  rng_.seed(std::random_device()());
}

bool Noise::is_initialized() { return initialized_; }

ignition::math::Pose3d Noise::random_pose() {
  ignition::math::Pose3d pose;

  // Not to be confused because the samples are from a normal distribution, this actually
  // generates noise uniformly within a sphere.
  pose.Pos().X() = normal_(rng_);
  pose.Pos().Y() = normal_(rng_);
  pose.Pos().Z() = normal_(rng_);
  pose.Pos() = pose.Pos().Normalize() * uniform01_(rng_) * noise_pos_;

  double x = normal_(rng_), y = normal_(rng_), z = normal_(rng_);
  double length = sqrt(x * x + y * y + z * z);
  pose.Rot().Axis(x / length, y / length, z / length, noise_orient_ * normal_(rng_));
  return pose;
}
