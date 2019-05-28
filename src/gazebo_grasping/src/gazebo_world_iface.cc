#include <memory>
#include <string>

#include <boost/python.hpp>
#include <gazebo/msgs/msgs.hh>
#include <ignition/msgs.hh>
#include <ignition/transport.hh>

#include "barrett_control.pb.h"
#include "observation.pb.h"
#include "world_config.pb.h"

#include "gazebo_grasping/gazebo_world_iface.h"

using namespace boost::python;

ignition::transport::Node GazeboWorldIface::node_;

bool GazeboWorldIface::set_config(const std::string& contact_type, double sensor_freq,
                                  const dict& points_of_interest, double pos_noise,
                                  double orient_noise) {
  gazebo_grasping::msgs::WorldConfig req;
  ignition::msgs::Boolean rep;

  bool result;
  unsigned int timeout = 10000;

  const list& links = points_of_interest.keys();
  for (int i = 0; i < len(links); ++i) {
    auto poi = req.add_points_of_interest();
    poi->set_link_name(extract<std::string>(links[i]));

    const list& positions = extract<list>(points_of_interest[links[i]]);
    for (int k = 0; k < len(positions); ++k) {
      auto point = poi->add_points();
      point->set_x(extract<double>(positions[k][0]));
      point->set_y(extract<double>(positions[k][1]));
      point->set_z(extract<double>(positions[k][2]));
    }
  }

  req.set_contact_type(contact_type);
  req.set_sensor_freq(sensor_freq);
  req.set_pos_noise(pos_noise);
  req.set_orient_noise(orient_noise);

  if (!node_.Request("/set_config", req, timeout, rep, result)) throw "Setting config failed!";
  return rep.data();
}

list GazeboWorldIface::step(const list& action) {
  gazebo_grasping::msgs::BarrettControl req;
  req.set_spread(extract<double>(action[0]));
  req.set_joint1(extract<double>(action[1]));
  req.set_joint2(extract<double>(action[2]));
  req.set_joint3(extract<double>(action[3]));

  // Take into account the twist input.
  if (len(action) == 10)
    for (int i = 4; i < 10; ++i) req.add_twist(extract<double>(action[i]));

  gazebo_grasping::msgs::Observation rep;

  bool result;
  unsigned int timeout = 20000;

  if (!node_.Request("/step", req, timeout, rep, result)) throw "Step failed!";

  list obs, contacts, hand_dof, obj_rel_pose, obj_rel_twist, sdf_points;
  for (int i = 0; i < rep.contacts_size(); ++i) contacts.append(rep.contacts(i));
  for (int i = 0; i < rep.hand_dof_size(); ++i) hand_dof.append(rep.hand_dof(i));
  for (int i = 0; i < rep.obj_rel_pose_size(); ++i) obj_rel_pose.append(rep.obj_rel_pose(i));
  for (int i = 0; i < rep.obj_rel_twist_size(); ++i) obj_rel_twist.append(rep.obj_rel_twist(i));
  for (int i = 0; i < rep.sdf_points_size(); ++i) sdf_points.append(rep.sdf_points(i));

  obs.append(contacts);
  obs.append(hand_dof);
  obs.append(obj_rel_pose);
  obs.append(obj_rel_twist);
  obs.append(rep.obj_speed());
  obs.append(sdf_points);
  return obs;
}

void GazeboWorldIface::apply_action(const list& action) {
  gazebo_grasping::msgs::BarrettControl req;
  req.set_spread(extract<double>(action[0]));
  req.set_joint1(extract<double>(action[1]));
  req.set_joint2(extract<double>(action[2]));
  req.set_joint3(extract<double>(action[3]));

  // Take into account the twist input.
  if (len(action) == 10)
    for (int i = 4; i < 10; ++i) req.add_twist(extract<double>(action[i]));

  ignition::msgs::Boolean rep;
  bool result;
  unsigned int timeout = 10000;

  if (!node_.Request("/apply_action", req, timeout, rep, result)) throw "Apply action failed!";
}

void GazeboWorldIface::reset() {
  ignition::msgs::Boolean rep;

  bool result;
  unsigned int timeout = 20000;

  if (!node_.Request("/reset", timeout, rep, result)) {
    std::cerr << "Reset failed! Retrying..." << std::endl;
    timeout = 30000;
    if (!node_.Request("/reset", timeout, rep, result)) throw "Reset failed two times!";
  }
}

double GazeboWorldIface::get_sim_time() {
  ignition::msgs::Double rep;

  bool result;
  unsigned int timeout = 5000;

  if (!node_.Request("/get_sim_time", timeout, rep, result))
    throw "Getting simulation time failed!";

  return rep.data();
}

list GazeboWorldIface::get_object_rel_pose() {
  gazebo::msgs::Pose rep;

  bool result;
  unsigned int timeout = 5000;

  if (!node_.Request("/get_object_rel_pose", timeout, rep, result))
    throw "Getting simulation time failed!";

  list rel_pose;
  rel_pose.append(rep.position().x());
  rel_pose.append(rep.position().y());
  rel_pose.append(rep.position().z());
  rel_pose.append(rep.orientation().w());
  rel_pose.append(rep.orientation().x());
  rel_pose.append(rep.orientation().y());
  rel_pose.append(rep.orientation().z());
  return rel_pose;
}

void GazeboWorldIface::set_logging(bool state) {
  ignition::msgs::Boolean req;
  req.set_data(state);

  if (!node_.Request("/set_logging", req)) throw "Setting logging failed!";
}

void GazeboWorldIface::set_paused(bool state) {
  ignition::msgs::Boolean req;
  req.set_data(state);

  if (!node_.Request("/set_paused", req)) throw "Setting pause failed!";
}

void GazeboWorldIface::set_control(const list& control) {
  gazebo_grasping::msgs::BarrettControl req;
  req.set_spread(extract<double>(control[0]));
  req.set_joint1(extract<double>(control[1]));
  req.set_joint2(extract<double>(control[2]));
  req.set_joint3(extract<double>(control[3]));

  if (len(control) == 10)
    for (int i = 4; i < 10; ++i) req.add_twist(extract<double>(control[i]));

  if (!node_.Request("/set_control", req)) throw "Setting control failed!";
}

void GazeboWorldIface::set_object_rel_pose(const list& pose) {
  gazebo::msgs::Pose req;
  req.mutable_position()->set_x(extract<double>(pose[0]));
  req.mutable_position()->set_y(extract<double>(pose[1]));
  req.mutable_position()->set_z(extract<double>(pose[2]));
  req.mutable_orientation()->set_w(extract<double>(pose[3]));
  req.mutable_orientation()->set_x(extract<double>(pose[4]));
  req.mutable_orientation()->set_y(extract<double>(pose[5]));
  req.mutable_orientation()->set_z(extract<double>(pose[6]));

  if (!node_.Request("/set_object_rel_pose", req)) throw "Setting object relative pose failed!";
}

void GazeboWorldIface::set_gravity(const list& gravity) {
  gazebo::msgs::Vector3d req;
  req.set_x(extract<double>(gravity[0]));
  req.set_y(extract<double>(gravity[1]));
  req.set_z(extract<double>(gravity[2]));

  if (!node_.Request("/set_gravity", req)) throw "Setting gravity failed!";
}

void GazeboWorldIface::spawn_object(const std::string& name) {
  gazebo::msgs::GzString req;
  req.set_data(name);

  if (!node_.Request("/spawn_object", req)) throw "Spawning object failed!";
}

// Name has to match the name in CMakeLists' `add_library`.
BOOST_PYTHON_MODULE(gazebo_grasping) {
  class_<GazeboWorldIface>("GazeboWorldIface")
      .def("set_config", &GazeboWorldIface::set_config)
      .def("step", &GazeboWorldIface::step)
      .def("apply_action", &GazeboWorldIface::apply_action)
      .def("reset", &GazeboWorldIface::reset)
      .def("get_sim_time", &GazeboWorldIface::get_sim_time)
      .def("get_object_rel_pose", &GazeboWorldIface::get_object_rel_pose)
      .def("set_logging", &GazeboWorldIface::set_logging)
      .def("set_paused", &GazeboWorldIface::set_paused)
      .def("set_control", &GazeboWorldIface::set_control)
      .def("set_object_rel_pose", &GazeboWorldIface::set_object_rel_pose)
      .def("set_gravity", &GazeboWorldIface::set_gravity)
      .def("spawn_object", &GazeboWorldIface::spawn_object);
}
