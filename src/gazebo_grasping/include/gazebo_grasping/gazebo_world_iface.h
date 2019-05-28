#ifndef __GAZEBO_GRASPING__GAZEBO_WORLD_IFACE_H__
#define __GAZEBO_GRASPING__GAZEBO_WORLD_IFACE_H__

#include <boost/python.hpp>
#include <ignition/transport.hh>

using namespace boost::python;

class GazeboWorldIface {
 public:
  GazeboWorldIface() {}

  bool set_config(const std::string& contact_type, double sensor_freq,
                  const dict& points_of_interest, double pos_noise, double orient_noise);
  list step(const list& action);
  void apply_action(const list& action);
  void reset();
  double get_sim_time();
  list get_object_rel_pose();
  void set_logging(bool state);
  void set_paused(bool state);
  void set_control(const list& control);
  void set_object_rel_pose(const list& pose);
  void set_gravity(const list& gravity);
  void spawn_object(const std::string& name);

 private:
  static ignition::transport::Node node_;
};

#endif  // __GAZEBO_GRASPING__GAZEBO_WORLD_IFACE_H__
