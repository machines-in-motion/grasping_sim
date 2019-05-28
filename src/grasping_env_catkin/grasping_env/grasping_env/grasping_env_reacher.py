# Copyright (c) 2018 Max Planck Gesellschaft
# Author : Hamza Meric
# License : License BSD-3-Clause


import numpy as np

from gym import spaces

from .core import GraspingEnv


class GraspingEnvReacher(GraspingEnv):

    def __init__(self,
                 world="reacher",
                 only_z=False,
                 *args,
                 **kwargs):
        super().__init__(world=world, *args, **kwargs)
        if self.pois_in_state or self.sdfs:
            print("Points of interest are not supported!")
            self.close()

        max_hand_lin_vel = kwargs.get('max_hand_lin_vel', 0.0)
        max_hand_ang_vel = kwargs.get('max_hand_ang_vel', 0.0)
        if only_z:
            action_limit = np.hstack((max_hand_lin_vel * np.ones(1), max_hand_ang_vel * np.ones(3)))
            self.preaction = np.array([-0.5, -0.5, -0.5, -0.5, 0, 0])
        else:
            action_limit = np.hstack((max_hand_lin_vel * np.ones(3), max_hand_ang_vel * np.ones(3)))
            self.preaction = np.array([-0.5, -0.5, -0.5, -0.5])
        self.action_space = spaces.Box(-action_limit, action_limit)
        self.reset()

    def get_reward(self):
        return self.k_obj_dist * (np.linalg.norm(self.prev_obs[:3]) - np.linalg.norm(self.obs[:3]))

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        try:
            reward = 0
            done = False
            if self.n_steps >= self.spec.max_episode_steps:
                print("Max steps already reached!")
                return [], 0, True, {}

            raw_obs = self.world_step(np.hstack((self.preaction, action)))
            self.process_raw_obs(raw_obs)

            reward = self.get_reward()
            if self.n_steps + 1 == self.spec.max_episode_steps:
                done = True

            self.prev_obs = self.obs
            self.n_steps += 1
            return self.obs, reward, done, {}
        except Exception as err:
            raise RuntimeError(err)

    def reset(self):
        if self.logging:
            self.stop_logging()

        self.world.reset()
        # Generate a random point within a sphere.
        new_pos = np.random.randn(3)
        new_pos = new_pos / (1e-7 + np.linalg.norm(new_pos)) * np.random.rand() * 2
        self.world.set_object_rel_pose(new_pos.tolist() + [1, 0, 0, 0])

        # Logging.
        self.n_episodes = self.n_episodes + 1 if self.n_episodes is not None else 0
        if self.log_every is not None and (self.n_episodes + 1) % self.log_every == 0:
            self.start_logging()

        raw_obs = self.world_step(np.array([0] * self.action_space.shape[0]))
        self.process_raw_obs(raw_obs)

        self.prev_obs = self.obs
        self.n_steps = 0
        return self.obs

    def __str__(self):
        return "GraspingEnvReacher"
