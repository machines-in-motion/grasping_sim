#include <unistd.h>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/bind.hpp>
#include <gazebo/common/Time.hh>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/Contact.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/util/LogRecord.hh>
#include <ignition/msgs.hh>
#include <ignition/transport.hh>

#include "barrett_control.pb.h"
#include "observation.pb.h"
#include "world_config.pb.h"

#include "gazebo_grasping/noise.h"

namespace gazebo {

void dummy_cb(ConstContactsPtr&) {}

class GraspingWorldPlugin : public WorldPlugin {
 public:
  ~GraspingWorldPlugin() {
    joints_update_->~Connection();
    shutdown();
  }

  void Load(physics::WorldPtr parent, sdf::ElementPtr /*sdf*/) {
    configured_ = false;

    collision_map_["palm_link_collision"] = 0u;
    collision_map_["finger_1_prox_link_collision"] = 1u;
    collision_map_["finger_1_med_link_collision"] = 2u;
    collision_map_["finger_1_dist_link_collision"] = 3u;
    collision_map_["finger_2_prox_link_collision"] = 4u;
    collision_map_["finger_2_med_link_collision"] = 5u;
    collision_map_["finger_2_dist_link_collision"] = 6u;
    collision_map_["finger_3_med_link_collision"] = 7u;
    collision_map_["finger_3_dist_link_collision"] = 8u;

    // Physics.
    world_ = parent;
    world_->SetPaused(true);
    physics::PhysicsEnginePtr pe = world_->Physics();
    pe->SetMaxStepSize(1e-3);
    pe->SetRealTimeUpdateRate(0.0);
    pe->SetGravity(ignition::math::Vector3d(0, 0, 0));

    // This is because in Gazebo 7 contacts are not evaluated if there are no subscribers.
    gz_node_ = transport::NodePtr(new transport::Node());
    gz_node_->Init(world_->Name());
    dummy_sub_ = gz_node_->Subscribe("~/physics/contacts", dummy_cb);

    barrett_ = world_->ModelByName("barrett");
    GZ_ASSERT(barrett_, "Barrett hand is not spawned!");

    // Since torque is applied for only one timestep.
    joints_update_ = event::Events::ConnectBeforePhysicsUpdate(
        boost::bind(&GraspingWorldPlugin::OnUpdate, this, _1));

    reset();

    // Advertise services.
    if (!node_.Advertise("/set_config", &GraspingWorldPlugin::set_config_srv, this))
      gzthrow("Error advertising service [ /set_config ]!");
    if (!node_.Advertise("/step", &GraspingWorldPlugin::step_srv, this))
      gzthrow("Error advertising service [ /step ]!");
    if (!node_.Advertise("/apply_action", &GraspingWorldPlugin::apply_action_srv, this))
      gzthrow("Error advertising service [ /apply_action ]!");
    if (!node_.Advertise("/reset", &GraspingWorldPlugin::reset_srv, this))
      gzthrow("Error advertising service [ /reset ]!");
    if (!node_.Advertise("/get_sim_time", &GraspingWorldPlugin::get_sim_time_srv, this))
      gzthrow("Error advertising service [ /get_sim_time ]!");
    if (!node_.Advertise("/get_object_rel_pose", &GraspingWorldPlugin::get_obj_rel_pose_srv, this))
      gzthrow("Error advertising service [ /get_object_rel_pose ]!");
    if (!node_.Advertise("/set_logging", &GraspingWorldPlugin::set_logging_srv, this))
      gzthrow("Error advertising service [ /set_logging ]!");
    if (!node_.Advertise("/set_paused", &GraspingWorldPlugin::set_paused_srv, this))
      gzthrow("Error advertising service [ /set_paused ]!");
    if (!node_.Advertise("/set_control", &GraspingWorldPlugin::set_control_srv, this))
      gzthrow("Error advertising service [ /set_control ]!");
    if (!node_.Advertise("/set_object_rel_pose", &GraspingWorldPlugin::set_obj_rel_pose_srv, this))
      gzthrow("Error advertising service [ /set_object_rel_pose ]!");
    if (!node_.Advertise("/spawn_object", &GraspingWorldPlugin::spawn_object_srv, this))
      gzthrow("Error advertising service [ /spawn_object ]!");
    if (!node_.Advertise("/set_gravity", &GraspingWorldPlugin::set_gravity_srv, this))
      gzthrow("Error advertising service [ /set_gravity ]!");

    gzmsg << "World plugin successfully loaded." << std::endl;
  }

  void set_mimic_joints() {
    static const std::vector<std::string> joints{"finger_1_med_joint", "finger_2_med_joint",
                                                 "finger_3_med_joint"};
    static const std::vector<std::string> dep_joints{"finger_1_dist_joint", "finger_2_dist_joint",
                                                     "finger_3_dist_joint"};
    static const std::vector<double> gear_ratios{0.344262295, 0.344262295, 0.344262295};

    if (joint_torques_[0] >= 0 &&
            barrett_->GetJoint("finger_1_prox_joint")->Position() >
                barrett_->GetJoint("finger_2_prox_joint")->Position() ||
        joint_torques_[0] < 0 &&
            barrett_->GetJoint("finger_1_prox_joint")->Position() <
                barrett_->GetJoint("finger_2_prox_joint")->Position()) {
      barrett_->GetJoint("finger_2_prox_joint")
          ->SetPosition(0, barrett_->GetJoint("finger_1_prox_joint")->Position());
    } else {
      barrett_->GetJoint("finger_1_prox_joint")
          ->SetPosition(0, barrett_->GetJoint("finger_2_prox_joint")->Position());
    }

    for (size_t i = 0u; i < dep_joints.size(); ++i)
      barrett_->GetJoint(dep_joints[i])
          ->SetPosition(0, barrett_->GetJoint(joints[i])->Position() * gear_ratios[i]);
  }

  void max_contacts(gazebo_grasping::msgs::Observation& obs,
                    const gazebo_grasping::msgs::Observation& obs_temp) {
    GZ_ASSERT(obs.contacts_size() == obs_temp.contacts_size(), "Contacts have different size!");
    // Take the max of the contact vector.
    for (int i = 0; i < obs_temp.contacts_size() / 3; ++i) {
      const double sq_norm1 = obs.contacts(3 * i + 0) * obs.contacts(3 * i + 0) +
                              obs.contacts(3 * i + 1) * obs.contacts(3 * i + 1) +
                              obs.contacts(3 * i + 2) * obs.contacts(3 * i + 2);
      const double sq_norm2 = obs_temp.contacts(3 * i + 0) * obs_temp.contacts(3 * i + 0) +
                              obs_temp.contacts(3 * i + 1) * obs_temp.contacts(3 * i + 1) +
                              obs_temp.contacts(3 * i + 2) * obs_temp.contacts(3 * i + 2);
      if (sq_norm2 > sq_norm1) {
        obs.set_contacts(3 * i + 0, obs_temp.contacts(3 * i + 0));
        obs.set_contacts(3 * i + 1, obs_temp.contacts(3 * i + 1));
        obs.set_contacts(3 * i + 2, obs_temp.contacts(3 * i + 2));
      }
    }
  }

  void get_obs_contacts(gazebo_grasping::msgs::Observation& obs) {
    // The resulting contact forces are current contact force measurements of the obs plus the new
    // measured contact forces. In case of standard/max contacts make sure that the obs contacts are
    // cleared beforehand. In case of mean contacts this addition is desirable.
    GZ_ASSERT(obs.contacts_size() >= 27, "Make sure observations are resized.");
    // ========== Get contacts. [27] // 9 links * 3 for x, y, z
    const std::vector<physics::Contact*>& contacts =
        world_->Physics()->GetContactManager()->GetContacts();
    unsigned idx;  // Index of the link in contact.
    for (const auto& contact : contacts) {
      bool barrett_first_collision =
          collision_map_.find(contact->collision1->GetName()) != collision_map_.end();
      bool barrett_second_collision =
          collision_map_.find(contact->collision2->GetName()) != collision_map_.end();
      if (barrett_first_collision && barrett_second_collision)
        continue;
      else if (barrett_first_collision)
        idx = collision_map_[contact->collision1->GetName()];
      else if (barrett_second_collision)
        idx = collision_map_[contact->collision2->GetName()];
      else if (contact->collision1->GetName() == "c_gp" ||
               contact->collision2->GetName() == "c_gp")  // Collision ground plane.
        continue;
      else
        gzthrow("Unknown collisions:\n: " + contact->collision1->GetName() + "\n" +
                contact->collision2->GetName());

      if (barrett_first_collision) {
        for (int i = 0; i < contact->count; ++i) {
          obs.set_contacts(3 * idx + 0,
                           obs.contacts(3 * idx + 0) + contact->wrench[i].body1Force.X());
          obs.set_contacts(3 * idx + 1,
                           obs.contacts(3 * idx + 1) + contact->wrench[i].body1Force.Y());
          obs.set_contacts(3 * idx + 2,
                           obs.contacts(3 * idx + 2) + contact->wrench[i].body1Force.Z());
        }
      } else {
        for (int i = 0; i < contact->count; ++i) {
          obs.set_contacts(3 * idx + 0,
                           obs.contacts(3 * idx + 0) + contact->wrench[i].body2Force.X());
          obs.set_contacts(3 * idx + 1,
                           obs.contacts(3 * idx + 1) + contact->wrench[i].body2Force.Y());
          obs.set_contacts(3 * idx + 2,
                           obs.contacts(3 * idx + 2) + contact->wrench[i].body2Force.Z());
        }
      }
    }
    world_->Physics()->GetContactManager()->Clear();
  }

  void get_obs_contact_torques(gazebo_grasping::msgs::Observation& obs) {
    GZ_ASSERT(obs.contacts_size() >= 3, "Make sure observations are resized.");
    // ========== Get contact torques. [3] // 3 dist links
    const std::vector<physics::Contact*>& contacts =
        world_->Physics()->GetContactManager()->GetContacts();
    unsigned idx;  // Index of the link in contact.
    for (const auto& contact : contacts) {
      bool barrett_first_collision =
          collision_map_.find(contact->collision1->GetName()) != collision_map_.end();
      bool barrett_second_collision =
          collision_map_.find(contact->collision2->GetName()) != collision_map_.end();
      if (barrett_first_collision && barrett_second_collision)
        continue;
      else if (barrett_first_collision)
        idx = collision_map_[contact->collision1->GetName()];
      else if (barrett_second_collision)
        idx = collision_map_[contact->collision2->GetName()];
      else if (contact->collision1->GetName() == "c_gp" ||
               contact->collision2->GetName() == "c_gp")  // Collision ground plane.
        continue;
      else
        gzthrow("Unknown collisions:\n: " + contact->collision1->GetName() + "\n" +
                contact->collision2->GetName());

      // We only consider distal links, and these are the corresponding indices.
      if (idx != 3u && idx != 6u && idx != 8u) continue;
      if (barrett_first_collision) {
        if (idx == 3u) {
          for (int i = 0; i < contact->count; ++i) {
            const auto rel_pos =
                (ignition::math::Pose3d(contact->positions[i], ignition::math::Quaterniond()) -
                 barrett_->GetLink("finger_1_dist_link")->WorldPose())
                    .Pos();
            obs.set_contacts(0, obs.contacts(0) + rel_pos.Cross(contact->wrench[i].body1Force).Z());
          }
        } else if (idx == 6u) {
          for (int i = 0; i < contact->count; ++i) {
            const auto rel_pos =
                (ignition::math::Pose3d(contact->positions[i], ignition::math::Quaterniond()) -
                 barrett_->GetLink("finger_2_dist_link")->WorldPose())
                    .Pos();
            obs.set_contacts(1, obs.contacts(1) + rel_pos.Cross(contact->wrench[i].body1Force).Z());
          }
        } else if (idx == 8u) {
          for (int i = 0; i < contact->count; ++i) {
            const auto rel_pos =
                (ignition::math::Pose3d(contact->positions[i], ignition::math::Quaterniond()) -
                 barrett_->GetLink("finger_3_dist_link")->WorldPose())
                    .Pos();
            obs.set_contacts(2, obs.contacts(2) + rel_pos.Cross(contact->wrench[i].body1Force).Z());
          }
        }
      } else {
        if (idx == 3u) {
          for (int i = 0; i < contact->count; ++i) {
            const auto rel_pos =
                (ignition::math::Pose3d(contact->positions[i], ignition::math::Quaterniond()) -
                 barrett_->GetLink("finger_1_dist_link")->WorldPose())
                    .Pos();
            obs.set_contacts(0, obs.contacts(0) + rel_pos.Cross(contact->wrench[i].body2Force).Z());
          }
        } else if (idx == 6u) {
          for (int i = 0; i < contact->count; ++i) {
            const auto rel_pos =
                (ignition::math::Pose3d(contact->positions[i], ignition::math::Quaterniond()) -
                 barrett_->GetLink("finger_2_dist_link")->WorldPose())
                    .Pos();
            obs.set_contacts(1, obs.contacts(1) + rel_pos.Cross(contact->wrench[i].body2Force).Z());
          }
        } else if (idx == 8u) {
          for (int i = 0; i < contact->count; ++i) {
            const auto rel_pos =
                (ignition::math::Pose3d(contact->positions[i], ignition::math::Quaterniond()) -
                 barrett_->GetLink("finger_3_dist_link")->WorldPose())
                    .Pos();
            obs.set_contacts(2, obs.contacts(2) + rel_pos.Cross(contact->wrench[i].body2Force).Z());
          }
        }
      }
    }
    world_->Physics()->GetContactManager()->Clear();
  }

  void get_obs_rest(gazebo_grasping::msgs::Observation& obs) {
    physics::ModelPtr obj_model = world_->ModelByName("object");
    const ignition::math::Pose3d& obj_w_pose =
        pose_noise_.is_initialized() ? pose_noise_.random_pose() + obj_model->WorldPose()
                                     : obj_model->WorldPose();
    // ========== Get hand DOF. [4]
    obs.add_hand_dof(barrett_->GetJoint("finger_1_prox_joint")->Position());
    obs.add_hand_dof(barrett_->GetJoint("finger_1_med_joint")->Position());
    obs.add_hand_dof(barrett_->GetJoint("finger_2_med_joint")->Position());
    obs.add_hand_dof(barrett_->GetJoint("finger_3_med_joint")->Position());

    // ========== Get object relative pose. [7]
    const ignition::math::Pose3d& hand_pose = barrett_->WorldPose();
    const ignition::math::Pose3d obj_rel_pose = obj_w_pose - hand_pose;

    obs.add_obj_rel_pose(obj_rel_pose.Pos().X());
    obs.add_obj_rel_pose(obj_rel_pose.Pos().Y());
    obs.add_obj_rel_pose(obj_rel_pose.Pos().Z());
    obs.add_obj_rel_pose(obj_rel_pose.Rot().W());
    obs.add_obj_rel_pose(obj_rel_pose.Rot().X());
    obs.add_obj_rel_pose(obj_rel_pose.Rot().Y());
    obs.add_obj_rel_pose(obj_rel_pose.Rot().Z());

    // ========== Get object relative twist. [6]
    const ignition::math::Vector3d model_rel_linear_vel = hand_pose.Rot().RotateVectorReverse(
        obj_model->WorldLinearVel() - barrett_->WorldLinearVel());
    const ignition::math::Vector3d model_rel_angular_vel = hand_pose.Rot().RotateVectorReverse(
        obj_model->WorldAngularVel() - barrett_->WorldAngularVel());
    obs.add_obj_rel_twist(model_rel_linear_vel.X());
    obs.add_obj_rel_twist(model_rel_linear_vel.Y());
    obs.add_obj_rel_twist(model_rel_linear_vel.Z());
    obs.add_obj_rel_twist(model_rel_angular_vel.X());
    obs.add_obj_rel_twist(model_rel_angular_vel.Y());
    obs.add_obj_rel_twist(model_rel_angular_vel.Z());

    // ========== Get object absolute speed. [1]
    obs.set_obj_speed(obj_model->WorldLinearVel().Length());

#ifndef NDEBUG
    int counter = 0;
#endif
    // ========== Get fingertip positions.
    // The order of iteration is uniquely determined, since map items are sorted by key (link) name.
    for (const auto& poi : points_of_interest_) {
      const ignition::math::Pose3d& link_pose = barrett_->GetLink(poi.first)->WorldPose();
      for (const auto& point3d : poi.second) {
        ignition::math::Vector3d pos = link_pose.Pos() + link_pose.Rot().RotateVector(point3d);
#ifndef NDEBUG
        world_->ModelByName("position_probe" + std::to_string(counter++))
            ->SetWorldPose(ignition::math::Pose3d(pos, ignition::math::Quaterniond()));
#endif
        // Relative frame of the object.
        pos = obj_w_pose.Rot().RotateVectorReverse(pos - obj_w_pose.Pos());

        obs.add_sdf_points(pos.X());
        obs.add_sdf_points(pos.Y());
        obs.add_sdf_points(pos.Z());
      }
    }
  }

  void set_config_srv(const gazebo_grasping::msgs::WorldConfig& req, ignition::msgs::Boolean& rep,
                      bool& result) {
    if (configured_) {
      gzerr << "World already configured!" << std::endl;
      rep.set_data(false);
      result = false;
      return;
    }
    static const std::unordered_map<std::string, ContactType> contact_types_map{
        {"standard", ContactType::Standard},
        {"mean", ContactType::Mean},
        {"max", ContactType::Max},
        {"torque", ContactType::Torque},
        {"none", ContactType::None}};
    if (contact_types_map.find(req.contact_type()) == contact_types_map.end()) {
      // Contact type not recognized.
      rep.set_data(false);
      result = false;
      return;
    }
    contact_type_ = contact_types_map.at(req.contact_type());

    update_period_ = 1.0 / req.sensor_freq();
    if (req.pos_noise() > 0 || req.orient_noise() > 0)
      pose_noise_ = Noise(req.pos_noise(), req.orient_noise());

#ifndef NDEBUG
    int counter = 0;
#endif
    for (int i = 0; i < req.points_of_interest_size(); ++i) {
      std::vector<ignition::math::Vector3d> points;

      const auto& poi = req.points_of_interest(i);
      for (int k = 0; k < poi.points_size(); ++k) {
#ifndef NDEBUG
        spawn_position_probe(counter++);
#endif
        points.push_back(
            ignition::math::Vector3d(poi.points(k).x(), poi.points(k).y(), poi.points(k).z()));
      }

      points_of_interest_[req.points_of_interest(i).link_name()] = std::move(points);
    }

    configured_ = true;
    rep.set_data(true);
    result = true;
  }

  void step_srv(const gazebo_grasping::msgs::BarrettControl& req,
                gazebo_grasping::msgs::Observation& rep, bool& result) {
    if (!configured_) throw "World not configured!";
    joint_torques_[0] = req.spread();
    joint_torques_[1] = req.joint1();
    joint_torques_[2] = req.joint2();
    joint_torques_[3] = req.joint3();

    if (req.twist_size() == 6)
      // The velocity is set in the hand frame, because only object's relative pose is known, so
      // setting world twist makes no sense.
      barrett_->SetWorldTwist(barrett_->WorldPose().Rot().RotateVector(ignition::math::Vector3d(
                                  req.twist(0), req.twist(1), req.twist(2))),
                              barrett_->WorldPose().Rot().RotateVector(ignition::math::Vector3d(
                                  req.twist(3), req.twist(4), req.twist(5))));

    if (contact_type_ != ContactType::None && contact_type_ != ContactType::Torque)
      rep.mutable_contacts()->Resize(27, 0.0);

    if (contact_type_ == ContactType::Standard) {
      while (world_->SimTime() - last_observation_time_ < update_period_) world_->Step(1);
      // Standard contacts -> just get the last measurement.
      get_obs_contacts(rep);
    } else if (contact_type_ == ContactType::Mean) {
      // Get the mean of contacts.
      int counter = 0;
      do {
        world_->Step(1);
        // Sums up all the contacts.
        get_obs_contacts(rep);
        ++counter;
      } while (world_->SimTime() - last_observation_time_ < update_period_);
      // Rescale to mean.
      for (int i = 0; i < rep.contacts_size(); ++i) rep.set_contacts(i, rep.contacts(i) / counter);
    } else if (contact_type_ == ContactType::Max) {
      // Get the max of contacts.
      gazebo_grasping::msgs::Observation rep_temp;

      world_->Step(1);
      get_obs_contacts(rep);
      while (world_->SimTime() - last_observation_time_ < update_period_) {
        world_->Step(1);
        // Need to clear every time since get_obs_contacts is adding contact forces.
        rep_temp.mutable_contacts()->Resize(27, 0.0);
        get_obs_contacts(rep_temp);
        max_contacts(rep, rep_temp);
      }
    } else if (contact_type_ == ContactType::Torque) {
      rep.mutable_contacts()->Resize(3, 0.0);
      // Get the mean of torques.
      int counter = 0;
      do {
        world_->Step(1);
        // Sums up all the contacts.
        get_obs_contact_torques(rep);
        ++counter;
      } while (world_->SimTime() - last_observation_time_ < update_period_);
      // Rescale to mean.
      for (int i = 0; i < rep.contacts_size(); ++i) rep.set_contacts(i, rep.contacts(i) / counter);
    } else {
      // Contact type is None so just simulate the world.
      while (world_->SimTime() - last_observation_time_ < update_period_) world_->Step(1);
    }
    // Get the rest of the observation.
    get_obs_rest(rep);
    last_observation_time_ += update_period_;
    result = true;
  }

  void apply_action_srv(const gazebo_grasping::msgs::BarrettControl& req,
                        ignition::msgs::Boolean& rep, bool& result) {
    if (!configured_) throw "World not configured!";
    joint_torques_[0] = req.spread();
    joint_torques_[1] = req.joint1();
    joint_torques_[2] = req.joint2();
    joint_torques_[3] = req.joint3();

    if (req.twist_size() == 6)
      // The velocity is set in the hand frame, because only object's relative pose is known, so
      // setting world twist makes no sense.
      barrett_->SetWorldTwist(barrett_->WorldPose().Rot().RotateVector(ignition::math::Vector3d(
                                  req.twist(0), req.twist(1), req.twist(2))),
                              barrett_->WorldPose().Rot().RotateVector(ignition::math::Vector3d(
                                  req.twist(3), req.twist(4), req.twist(5))));

    while (world_->SimTime() - last_observation_time_ < update_period_) world_->Step(1);

    last_observation_time_ += update_period_;
    result = true;
  }

  void reset() {
    joint_torques_ = std::vector<double>(4, 0.0);
    world_->Reset();
    last_observation_time_ = world_->SimTime();
    for (auto& joint : barrett_->GetJoints()) {
      joint->SetPosition(0, 0.0);
      usleep(2000);
    }
  }

  void reset_srv(ignition::msgs::Boolean& rep, bool& result) {
    if (!configured_) throw "World not configured!";
    reset();
    result = true;
  }

  void get_sim_time_srv(ignition::msgs::Double& rep, bool& result) {
    rep.set_data(world_->SimTime().Double());
    result = true;
  }

  void get_obj_rel_pose_srv(msgs::Pose& rep, bool& result) {
    physics::ModelPtr obj_model = world_->ModelByName("object");
    const ignition::math::Pose3d& obj_w_pose =
        pose_noise_.is_initialized() ? pose_noise_.random_pose() + obj_model->WorldPose()
                                     : obj_model->WorldPose();

    const ignition::math::Pose3d obj_rel_pose = obj_w_pose - barrett_->WorldPose();

    rep.mutable_position()->set_x(obj_rel_pose.Pos().X());
    rep.mutable_position()->set_y(obj_rel_pose.Pos().Y());
    rep.mutable_position()->set_z(obj_rel_pose.Pos().Z());
    rep.mutable_orientation()->set_w(obj_rel_pose.Rot().W());
    rep.mutable_orientation()->set_x(obj_rel_pose.Rot().X());
    rep.mutable_orientation()->set_y(obj_rel_pose.Rot().Y());
    rep.mutable_orientation()->set_z(obj_rel_pose.Rot().Z());
    result = true;
  }

  void set_logging_srv(const ignition::msgs::Boolean& req) {
    if (req.data()) {
      util::LogRecord::Instance()->SetBasePath(std::getenv("GRASPING_LOG_DIR"));
      static const std::string kMsgEncoding = "zlib";
      util::LogRecord::Instance()->Start(kMsgEncoding);
    } else {
      util::LogRecord::Instance()->Stop();
    }
  }

  void set_paused_srv(const ignition::msgs::Boolean& req) { world_->SetPaused(req.data()); }

  void set_control_srv(const gazebo_grasping::msgs::BarrettControl& req) {
    joint_torques_[0] = req.spread();
    joint_torques_[1] = req.joint1();
    joint_torques_[2] = req.joint2();
    joint_torques_[3] = req.joint3();
  }

  void set_obj_rel_pose_srv(const msgs::Pose& req) {
    ignition::math::Vector3d pos(req.position().x(), req.position().y(), req.position().z());
    ignition::math::Quaterniond orien(req.orientation().w(), req.orientation().x(),
                                      req.orientation().y(), req.orientation().z());
    ignition::math::Pose3d obj_w_pose = ignition::math::Pose3d(pos, orien) + barrett_->WorldPose();
    while (!world_->ModelByName("object")) usleep(2000);
    world_->ModelByName("object")->SetWorldPose(obj_w_pose);
  }

  void spawn_object_srv(const msgs::GzString& req) {
    // TODO: Make this work properly.
    if (world_->ModelByName("object")) {
      world_->RemoveModel("object");
      usleep(500000);
    }
    world_->InsertModelFile("model://" + req.data());
  }

  void set_gravity_srv(const gazebo::msgs::Vector3d& req) {
    world_->Physics()->SetGravity(ignition::math::Vector3d(req.x(), req.y(), req.z()));
  }

  void OnUpdate(const common::UpdateInfo& /*_info*/) {
    barrett_->GetJoint("finger_1_prox_joint")->SetForce(0, joint_torques_[0] / 2);
    barrett_->GetJoint("finger_2_prox_joint")->SetForce(0, joint_torques_[0] / 2);
    barrett_->GetJoint("finger_1_med_joint")->SetForce(0, joint_torques_[1]);
    barrett_->GetJoint("finger_2_med_joint")->SetForce(0, joint_torques_[2]);
    barrett_->GetJoint("finger_3_med_joint")->SetForce(0, joint_torques_[3]);
    set_mimic_joints();
  }

#ifndef NDEBUG
  void spawn_position_probe(int index) {
    // This function is used for debugging purposes to spawn red spheres. These red spheres are used
    // to debug positions of points of interest.
    std::ostringstream new_model;

    new_model << "<sdf version='1.6'>"
              << "  <model name ='position_probe" << index << "'>"
              << "    <pose>0.0 0.0 2.0 0.0 0.0 0.0</pose>"
              << "    <link name='link'>"
              << "      <visual name='visual'>"
              << "        <geometry>"
              << "          <sphere>"
              << "            <radius>0.001</radius>"
              << "          </sphere>"
              << "        </geometry>"
              << "        <material>"
              << "          <script>"
              << "            <name>Gazebo/Red</name>"
              << "            <uri>file://media/materials/scripts/gazebo.material</uri>"
              << "          </script>"
              << "        </material>"
              << "      </visual>"
              << "    </link>"
              << "  </model>"
              << "</sdf>";

    world_->InsertModelString(new_model.str());
    while (!world_->ModelByName("position_probe" + std::to_string(index))) usleep(2000);
  }
#endif

 private:
  enum class ContactType { None = -1, Standard = 0, Mean = 1, Max = 2, Torque = 3 };

  ignition::transport::Node node_;
  transport::NodePtr gz_node_;
  transport::SubscriberPtr dummy_sub_;
  physics::WorldPtr world_;
  physics::ModelPtr barrett_;
  event::ConnectionPtr joints_update_;

  bool configured_;
  ContactType contact_type_;
  std::map<std::string, std::vector<ignition::math::Vector3d>> points_of_interest_;
  common::Time update_period_;
  common::Time last_observation_time_;
  Noise pose_noise_;
  std::vector<double> joint_torques_;
  std::unordered_map<std::string, unsigned> collision_map_;
};

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(GraspingWorldPlugin)

}  // namespace gazebo
