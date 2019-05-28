# Copyright (c) 2018 Max Planck Gesellschaft
# Author : Hamza Meric
# License : License BSD-3-Clause


import numpy as np

from .core import GraspingEnv


class GraspingEnvCylinder(GraspingEnv):

    def __init__(self,
                 world="cylinder",
                 *args,
                 **kwargs):
        super().__init__(world=world, *args, **kwargs)
        if self.pois_in_state or self.sdfs:
            print("Points of interest are not supported!")
            self.close()
        self.set_gravity([0, 0, -9.81])

        self.reset()

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
                done = True

            self.n_steps += 1
            return self.obs, reward, done, {}
        except Exception as err:
            raise RuntimeError(err)

    def reset(self):
        if self.logging:
            self.stop_logging()

        self.world.reset()

        # Logging.
        self.n_episodes = self.n_episodes + 1 if self.n_episodes is not None else 0
        if self.log_every is not None and (self.n_episodes + 1) % self.log_every == 0:
            self.start_logging()

        raw_obs = self.world_step(np.array([0] * self.action_space.shape[0]))
        self.process_raw_obs(raw_obs)

        self.n_steps = 0
        return self.obs

    def __str__(self):
        return "GraspingEnvCylinder"
