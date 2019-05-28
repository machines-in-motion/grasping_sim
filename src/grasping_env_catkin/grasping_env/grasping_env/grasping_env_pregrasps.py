# Copyright (c) 2018 Max Planck Gesellschaft
# Author : Hamza Meric
# License : License BSD-3-Clause


import os, os.path as osp, random
import numpy as np

from .core import GraspingEnv
from .sdf_tools import SignedDensityField
from .utils import load_pregrasps


class GraspingEnvPregrasps(GraspingEnv):

    def __init__(self,
                 world="pregrasps",
                 objects=None,            # Set to ["bottle_poisson_009"] by default.
                 max_obj_dist=0.5,
                 leave_bound_rew=-1,
                 drop_test_rew=10.0,
                 min_phys_score=0,        # Use grasps above certain score. 0 >= score => all.
                 max_phys_score=1,        # Use grasps below certain score. score >= 1 => all.
                 same_grasp_prob=-1,      # Probability to sample the same grasp again.
                 *args,
                 **kwargs):
        super().__init__(world=world, *args, **kwargs)
        self.max_obj_dist = max_obj_dist
        self.leave_bound_rew = leave_bound_rew
        self.drop_test_rew = drop_test_rew

        # Populate object data.
        self.same_grasp_prob = same_grasp_prob
        self.object = None
        self.objects = objects if objects is not None else ["bottle_poisson_009"]
        self.custom_pregrasp = None  # Used to allow manually setting a pre-grasp.
        self.obj_pregrasps = {}
        self.obj_phys_scores = {}
        for obj in self.objects:
            pregrasps_file = self.gazebo_grasping_path + "/models/" + obj + "/pregrasps.pkl"
            sdf_file = self.gazebo_grasping_path + "/models/" + obj + "/sdf.pkl"
            assert osp.isfile(pregrasps_file), "Pregrasps file not found!"
            assert osp.isfile(sdf_file), "SDF file not found!"
            self.obj_pregrasps[obj], self.obj_phys_scores[obj] = \
                load_pregrasps(pregrasps_file, min_phys_score)
            self.sdfs[obj] = SignedDensityField.from_pkl(sdf_file)

        # Actor defines the behavior during drop test. If actor is None, last applied torques are
        # repeated during the whole drop test.
        self.actor = None
        self.reset()

    def set_actor(self, actor):
        # Make sure wherever actor is used that the action is clipped.
        self.actor = actor

    def spawn_random_object(self):
        if self.custom_pregrasp is not None:
            self.object_initial_pose = self.custom_pregrasp
        else:
            # We keep the same pre-grasp with probability 0.8, which means that the object/pre-grasp
            # is expected to be repeated 5 times. This is done in order to feed more training data
            # of the same distribution.
            if self.object and self.same_grasp_prob > 0 and random.random() < self.same_grasp_prob:
                obj = self.object
            else:
                obj = random.choice(self.objects)
                self.object_initial_pose = random.choice(self.obj_pregrasps[obj])
            if obj != self.object:
                self.world.spawn_object(obj)
                self.object = obj
        self.world.set_object_rel_pose(self.object_initial_pose.tolist())

    def drop_test(self, duration):
        assert self.actor, "Actor must be defined!"
        # We execute the policy during drop test.
        passed = True
        gravity_vectors = [[0, 0, -12], [0, 0, 12]]
        initial_pos = np.copy(self.obj_rel_pose[0:3])
        for gravity in gravity_vectors:
            self.set_gravity(gravity)
            for i in range(int(duration * self.sensor_freq)):
                action = np.clip(self.actor(self.obs), self.action_space.low,
                                 self.action_space.high)
                raw_obs = self.world_step(action)
                self.process_raw_obs(raw_obs)

                # Fail in case of 5 cm movement.
                if np.linalg.norm(initial_pos - self.obj_rel_pose[0:3]) >= 0.05:
                    self.set_gravity([0.0, 0.0, 0.0])
                    if self.logging:
                        self.log['drop_test'] = False
                    return False
        self.set_gravity([0.0, 0.0, 0.0])

        if self.logging:
            self.log['drop_test'] = passed
        return passed

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        try:
            reward = 0
            done = False
            if self.n_steps >= self.spec.max_episode_steps:
                print("Max steps already reached!")
                return [], 0, True, {}

            raw_obs = self.world_step(action)
            self.process_raw_obs(raw_obs)

            reward = self.get_base_reward()
            if self.n_steps + 1 == self.spec.max_episode_steps:
                if self.drop_test_rew > 0:
                    # Need to cache since drop_test calls process_raw_obs which changes self.obs.
                    cached_obs = self.obs
                    if self.drop_test(0.5):
                        reward += self.drop_test_rew
                        self.success = True
                    self.obs = cached_obs
                done = True
            else:
                # Fail if the object moves more than 15 cm from the initial pose.
                if self.get_obj_dist_from_hand() > self.max_obj_dist:
                    done = True
                    reward += self.leave_bound_rew

            self.n_steps += 1
            return self.obs, reward, done, {}
        except Exception as err:
            raise RuntimeError(err)

    def reset(self):
        self.success = False
        if self.logging:
            self.stop_logging()

        self.world.reset()
        self.spawn_random_object()
        # To make sure object is spawned.
        self.world_step(np.array([0] * self.action_space.shape[0]))

        # Logging.
        self.n_episodes = self.n_episodes + 1 if self.n_episodes is not None else 0
        if self.log_every is not None and (self.n_episodes + 1) % self.log_every == 0:
            self.start_logging()

        raw_obs = self.world_step(np.array([0] * self.action_space.shape[0]))
        self.process_raw_obs(raw_obs)

        self.n_steps = 0
        return self.obs

    def __str__(self):
        return "GraspingEnvPregrasps"
