# Copyright (c) 2018 Max Planck Gesellschaft
# Author : Hamza Meric
# License : License BSD-3-Clause


import os, os.path as osp, sys, signal, subprocess, time
from collections import defaultdict
import pickle
import numpy as np

from gym import spaces
from gym.utils import closer

from gazebo_grasping.gazebo_grasping import GazeboWorldIface

env_closer = closer.Closer()


class GraspingEnv(object):

    # pose: pos.x, pos.y, pos.z, rot.w, rot.x, rot.y, rot.z
    def __init__(self,
                 world="empty",
                 log_every=None,
                 log_dir=None,
                 episode_len=1000,
                 sensor_freq=100.0,      # In Hz.
                 warm_init_eps=0,        # Number of simple policy initialization episodes.
                 contact_type="mean",    # Possible: "standard", "mean", "max", "torque" and "none".
                 binary_cont_rew=True,   # Reward from contacts is either binary or continuous.
                 k_cont=0.4,             # Weight of the reward from contacts.
                 k_obj_dist=2.0,         # Weight penalizing distance to object.
                 k_obj_vel=2.0,          # Weight penalizing object velocity.
                 k_mean_poi=1.0,         # Weight penalizing the mean of SDF distances of pois.
                 k_torque_reg=1.0e-5,    # Regularization factor for the input torque.
                 k_twist_reg=1.0e-4,     # Regularization factor for the input twist.
                 points_of_interest=None,
                 cont_in_state=True,
                 dof_in_state=True,
                 obj_pose_in_state=True,
                 obj_twist_in_state=True,
                 obj_speed_in_state=False,
                 pois_in_state=True,
                 time_in_state=False,
                 max_hand_lin_vel=0.0,   # Limit controlling hand's linear velocity.
                 max_hand_ang_vel=0.0,   # Limit controlling hand's angular velocity.
                 obj_pos_noise=0.0,
                 obj_orient_noise=0.0,
                 *args, **kwargs):
        assert os.environ['GRASPING_WS'], "GRASPING_WS environment variable is not defined!"
        os.environ["GRASPING_LOG_DIR"] = log_dir if log_dir is not None else ""

        # Launch the simulation with the given launch file.
        self.gazebo_grasping_path = osp.join(os.environ['GRASPING_WS'], "src/gazebo_grasping")
        world_path = self.gazebo_grasping_path + "/worlds/" + world + ".world"
        args = ["gzserver", "--verbose", world_path]
        self.gazebo_pid = subprocess.Popen(
            args, shell=False, stderr=sys.stderr, stdout=sys.stdout, preexec_fn=os.setsid).pid
        self.gzclient_pid = None

        # It is important that the world plugin gets loaded before services get called!
        # A better approach would be to send a signal from world_plugin to iface when it is ready.
        time.sleep(5.0)

        # Launch wrapped [C++] world interface.
        self.world = GazeboWorldIface()

        # OpenAI compatibility.
        self._env_closer_id = env_closer.register(self)
        self._closed = False
        self.spec = EnvSpec(id=0, max_episode_steps=episode_len, nondeterministic=True)
        self.metadata = {'render.modes': []}

        # Logging.
        self.log_every = log_every
        self.logging = False
        if log_every is not None and log_dir is not None:
            self.log_dir = log_dir
            self.log = defaultdict(list)

        # Stepping.
        self.sensor_freq = sensor_freq
        self.warm_init_eps = warm_init_eps
        self.n_steps = None
        self.n_episodes = None

        # Reward parameters.
        self.binary_cont_rew = binary_cont_rew
        self.k_cont = k_cont
        self.k_obj_dist = k_obj_dist
        self.k_obj_vel = k_obj_vel
        self.k_mean_poi = k_mean_poi
        self.k_torque_reg = k_torque_reg
        self.k_twist_reg = k_twist_reg
        if points_of_interest is None:
            points_of_interest = {}
        points_of_interest = dict(points_of_interest)  # Convert from OrderedDict type.

        # State space.
        self.cont_in_state = cont_in_state
        self.dof_in_state = dof_in_state
        self.obj_pose_in_state = obj_pose_in_state
        self.obj_twist_in_state = obj_twist_in_state
        self.obj_speed_in_state = obj_speed_in_state
        self.pois_in_state = pois_in_state
        self.time_in_state = time_in_state
        n_dim = 0
        self.contact_type = contact_type
        if contact_type in ["standard", "mean", "max"] and cont_in_state:
            n_dim = 27
        elif contact_type == "torque" and cont_in_state:
            n_dim = 3
        if dof_in_state:
            n_dim += 4
        if obj_pose_in_state:
            n_dim += 7
        if obj_twist_in_state:
            n_dim += 6
        if obj_speed_in_state:
            n_dim += 1
        if pois_in_state:
            n_dim += sum([len(x) for x in points_of_interest.values()])
        if time_in_state:
            n_dim += 1
        self.observation_space = spaces.Box(-np.inf * np.ones(n_dim), np.inf * np.ones(n_dim))
        self.obs = None
        self.sdfs = {}

        # Action space.
        action_limit = 2 * np.ones(4)
        # If twist control is also enabled.
        if max_hand_lin_vel > 0 or max_hand_ang_vel > 0:
            action_limit = np.hstack((action_limit, max_hand_lin_vel * np.ones(3),
                                      max_hand_ang_vel * np.ones(3)))
        self.action_space = spaces.Box(-action_limit, action_limit)

        # Rewards.
        self.reward_range = (-np.inf, np.inf)

        assert self.world.set_config(
            contact_type, sensor_freq, points_of_interest,
            obj_pos_noise, obj_orient_noise), "Setting world config failed!"

        # Initialize cached variables.
        self.contacts_principal = self.obj_rel_pose = self.obj_speed = self.poi_dists = None

    def unpause(self):
        self.world.set_paused(True)

    def pause(self):
        self.world.set_paused(False)

    def set_gravity(self, gravity):
        self.world.set_gravity(gravity)

    def start_logging(self):
        print("Starting logging.")
        self.logging = True
        self.world.set_logging(True)

    def stop_logging(self):
        print("Stopping logging.")
        self.logging = False
        self.world.set_logging(False)
        dump_file = self.log_dir + "/" + str(self.n_episodes) + ".p"
        with open(dump_file, 'wb') as file:
            pickle.dump(self.log, file, protocol=2)
        self.log.clear()

    def get_obj_dist_from_hand(self):
        obj_pos = self.obj_rel_pose[0:3]
        return np.linalg.norm(obj_pos)

    def get_poi_dists(self, pois):
        if not self.sdfs:
            return 0
        pos = np.reshape(pois, (-1, 3))
        return self.sdfs[self.object].get_distance(pos)

    def process_raw_obs(self, raw_obs):
        # As defined in world_plugin.cc:get_obs_contacts and get_obs_rest.
        # Cache previous values.
        self.prev_contacts_principal = self.contacts_principal
        self.prev_obj_rel_pose = self.obj_rel_pose
        self.prev_obj_speed = self.obj_speed
        self.prev_poi_dists = self.poi_dists

        self.contacts, self.hand_dof, self.obj_rel_pose, self.obj_rel_twist, \
            self.obj_speed, self.sdf_pts = raw_obs

        self.contacts = np.array(self.contacts)
        self.hand_dof = np.array(self.hand_dof)
        self.obj_rel_pose = np.array(self.obj_rel_pose)
        self.obj_rel_twist = np.array(self.obj_rel_twist)
        self.poi_dists = self.get_poi_dists(np.array(self.sdf_pts))
        contacts = self.contacts if self.cont_in_state else []
        hand_dof = self.hand_dof if self.dof_in_state else []
        obj_rel_pose = self.obj_rel_pose if self.obj_pose_in_state else []
        obj_rel_twist = self.obj_rel_twist if self.obj_twist_in_state else []
        obj_speed = self.obj_speed if self.obj_speed_in_state else []
        poi_dists = self.poi_dists if self.pois_in_state else []
        time = self.n_steps if self.time_in_state else []

        # Update contacts principal value.
        self.contacts_principal = 0.0
        if self.contact_type != "none":
            if self.binary_cont_rew:
                # Number of links that are in contact with the object.
                if self.contact_type == "torque":
                    self.contacts_principal = np.sum(np.abs(self.contacts) > 1e-2)
                else:
                    for i in range(9):
                        # It is in contact in case any of the three force coordinates are non-zero.
                        if np.any(np.abs(self.contacts[3 * i:3 * i + 3]) > 1e-2):
                            self.contacts_principal += 1
            else:
                # Sum of contact force norms.
                # For every link sum up the norms of the contact vector.
                for i in range(9):
                    self.contacts_principal += np.linalg.norm(self.contacts[3 * i:3 * i + 3])

        self.obs = np.hstack((contacts, hand_dof, obj_rel_pose,
                              obj_rel_twist, obj_speed, poi_dists, time))

    def get_base_reward(self):
        # Clip object's distance and speed.
        contact_principal_diff = self.contacts_principal - self.prev_contacts_principal
        obj_dist_diff = np.linalg.norm(self.prev_obj_rel_pose[:3]) - self.get_obj_dist_from_hand()
        obj_speed_diff = self.prev_obj_speed - self.obj_speed
        poi_dists_diff = np.mean(self.prev_poi_dists) - np.mean(self.poi_dists)

        if self.logging:
            self.log['r_contacts'].append(self.k_cont * contact_principal_diff)
            self.log['r_obj_dist'].append(self.k_obj_dist * obj_dist_diff)
            self.log['r_obj_speed'].append(self.k_obj_vel * obj_speed_diff)
            self.log['r_fing_dist'].append(self.k_mean_poi * poi_dists_diff)
            self.log['r_torque_reg'].append(-self.k_torque_reg * self.last_torque_sq_sum -
                                            self.k_twist_reg * self.last_twist_sq_sum)
        return (self.k_cont * contact_principal_diff +
                self.k_obj_dist * obj_dist_diff +
                self.k_obj_vel * obj_speed_diff +
                self.k_mean_poi * poi_dists_diff -
                self.k_torque_reg * self.last_torque_sq_sum -
                self.k_twist_reg * self.last_twist_sq_sum)

    def world_step(self, action):
        self.last_torque_sq_sum = action[:4].dot(action[:4])
        self.last_twist_sq_sum = action[4:10].dot(action[4:10]) if len(action) >= 10 else 0
        return self.world.step(action.tolist())

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        if self.gzclient_pid is None:
            self.gzclient_pid = subprocess.Popen(["gzclient"], shell=False, stderr=sys.stderr,
                                                 stdout=sys.stdout, preexec_fn=os.setsid).pid

    def close(self):
        """Perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        if not hasattr(self, '_closed') or self._closed:
            return
        env_closer.unregister(self._env_closer_id)
        if self.logging:
            self.stop_logging()
            time.sleep(0.5)
        if self.gzclient_pid is not None:
            os.killpg(os.getpgid(self.gzclient_pid), signal.SIGTERM)
        os.killpg(os.getpgid(self.gazebo_pid), signal.SIGTERM)
        self._closed = True

    def __del__(self):
        # Not completely Pythonic, but that is what OpenAI's gym does as well.
        # https://github.com/openai/gym/blob/master/gym/core.py#L201
        self.close()

    def __str__(self):
        return "GraspingEnv"

    @property
    def unwrapped(self):
        """Completely unwrap this env.
        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EnvSpec(object):
    """A specification for a particular instance of the environment.
    Args:
        id (str): The official environment ID
        nondeterministic (bool): Whether this environment is non-deterministic even after seeding
        tags (dict[str:any]): A set of arbitrary key-value tags on this environment, including
                              simple property=True tags
    Attributes:
        id (str): The official environment ID
    """

    def __init__(self,
                 id,
                 nondeterministic=False,
                 tags=None,
                 max_episode_steps=None,
                 max_episode_seconds=None):
        self.id = id
        self.nondeterministic = nondeterministic
        self.max_episode_steps = max_episode_steps
        self.max_episode_seconds = max_episode_seconds

    def __repr__(self):
        return "EnvSpec({})".format(self.id)

    @property
    def timestep_limit(self):
        return self.max_episode_steps

    @timestep_limit.setter
    def timestep_limit(self, value):
        self.max_episode_steps = value
