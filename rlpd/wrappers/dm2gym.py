# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""bsuite adapter for OpenAI gym run-loops."""

from typing import Any, Dict, Optional, Tuple, Union

import dm_env
from dm_env import specs
import gym
from gym import spaces
import numpy as np

import copy
from typing import OrderedDict

# OpenAI gym step format = obs, reward, is_finished, other_info
_GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]


def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict) or isinstance(spec, dict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        low = np.broadcast_to(spec.minimum, spec.shape)
        high = np.broadcast_to(spec.maximum, spec.shape)
        return spaces.Box(low=low,
                          high=high,
                          shape=spec.shape,
                          dtype=spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.integer):
            low = np.iinfo(spec.dtype).min
            high = np.iinfo(spec.dtype).max
        elif np.issubdtype(spec.dtype, np.inexact):
            low = float('-inf')
            high = float('inf')
        else:
            raise ValueError()

        return spaces.Box(low=low,
                          high=high,
                          shape=spec.shape,
                          dtype=spec.dtype)
    else:
        raise NotImplementedError


def dmc_obs2gym_obs(obs):
    if isinstance(obs, OrderedDict) or isinstance(obs, dict):
        obs = copy.copy(obs)
        for k, v in obs.items():
            obs[k] = dmc_obs2gym_obs(v)
        return obs
    else:
        return np.asarray(obs)


class DMCGYM(gym.core.Env):

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env: dm_env.Environment):
        self._env = env

        self.action_space = dmc_spec2gym_space(self._env.action_spec())

        self.observation_space = dmc_spec2gym_space(
            self._env.observation_spec())

        self.viewer = None
        self._last_observation = None

    def _get_viewer(self):
        if self.viewer is None:
            from gym.envs.mujoco.mujoco_rendering import Viewer
            self.viewer = Viewer(self._env.physics.model.ptr,
                                 self._env.physics.data.ptr)
        return self.viewer

    def __getattr__(self, name):
        return getattr(self._env, name)

    # def seed(self, seed: int):
    #     if hasattr(self._env, 'random_state'):
    #         self._env.random_state.seed(seed)
    #     else:
    #         self._env.task.random.seed(seed)

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        self._last_observation = time_step.observation
        reward = time_step.reward or 0
        done = time_step.last()
        obs = time_step.observation

        info = {}
        if done and time_step.discount == 1.0:
            info['TimeLimit.truncated'] = True

        return dmc_obs2gym_obs(obs), reward, done, info

    def reset(self):
        time_step = self._env.reset()
        self._last_observation = time_step.observation
        obs = time_step.observation
        return dmc_obs2gym_obs(obs)

    def render(self,
               mode='rgb_array',
               height: int = 84,
               width: int = 84,
               camera_id: int = 0,):
        assert mode in [
            'human', 'rgb_array'
        ], 'only support rgb_array and human mode, given %s' % mode
        # if mode == 'rgb_array':
        #     return self._env.physics.render(height=height,
        #                                     width=width,
        #                                     camera_id=camera_id)
        # elif mode == "human":
            # self._get_viewer().render()
        return self._last_observation["image"]


class GymFromDMEnv(gym.Env):
  """A wrapper that converts a dm_env.Environment to an OpenAI gym.Env."""

  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self, env: dm_env.Environment):
    self._env = env  # type: dm_env.Environment
    self._last_observation = None  # type: Optional[np.ndarray]
    self.viewer = None
    self.game_over = False  # Needed for Dopamine agents.

  def step(self, action: int) -> _GymTimestep:
    timestep = self._env.step(action)
    self._last_observation = timestep.observation
    reward = timestep.reward or 0.
    if timestep.last():
      self.game_over = True
    return timestep.observation, reward, timestep.last(), {}

  def reset(self) -> np.ndarray:
    self.game_over = False
    timestep = self._env.reset()
    self._last_observation = timestep.observation
    return timestep.observation

  def render(self, mode: str = 'rgb_array') -> Union[np.ndarray, bool]:
    if self._last_observation is None:
      raise ValueError('Environment not ready to render. Call reset() first.')

    if mode == 'rgb_array':
      return self._last_observation

    if mode == 'human':
      if self.viewer is None:
        # pylint: disable=import-outside-toplevel
        # pylint: disable=g-import-not-at-top
        from gym.envs.classic_control import rendering
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(self._last_observation)
      return self.viewer.isopen

  @property
  def action_space(self) -> spaces.Discrete:
    action_spec = self._env.action_spec()  # type: specs.DiscreteArray
    return spaces.Box(low=-1.0, high=1.0, shape=action_spec.shape, dtype=np.float32)

  @property
  def observation_space(self) -> spaces.Box:
    obs_spec = self._env.observation_spec()  # type: specs.Array
    # import ipdb; ipdb.set_trace();
    if isinstance(obs_spec, specs.BoundedArray):
      return spaces.Box(
          low=float(obs_spec.minimum),
          high=float(obs_spec.maximum),
          shape=obs_spec.shape,
          dtype=obs_spec.dtype)
    # return spaces.Box(
    #     low=-float('inf'),
    #     high=float('inf'),
    #     shape=obs_spec.shape,
    #     dtype=obs_spec.dtype)
    import ipdb; ipdb.set_trace()
    return {
      key:
      spaces.Box(
        low=obs_spec[key].minimum,
        high=obs_spec[key].maximum,
        shape=obs_spec[key].shape,
        dtype=obs_spec[key].dtype
      )
      for key in obs_spec.keys()
    }

  @property
  def reward_range(self) -> Tuple[float, float]:
    reward_spec = self._env.reward_spec()
    if isinstance(reward_spec, specs.BoundedArray):
      return reward_spec.minimum, reward_spec.maximum
    return -float('inf'), float('inf')

#   def __getattr__(self, attr):
#     """Delegate attribute access to underlying environment."""
#     return getattr(self._env, attr)


def space2spec(space: gym.Space, name: Optional[str] = None):
  """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.

  Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
  specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
  Dict spaces are recursively converted to tuples and dictionaries of specs.

  Args:
    space: The Gym space to convert.
    name: Optional name to apply to all return spec(s).

  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.
  """
  if isinstance(space, spaces.Discrete):
    return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                              minimum=space.low, maximum=space.high, name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype, minimum=0.0,
                              maximum=1.0, name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                              minimum=np.zeros(space.shape),
                              maximum=space.nvec, name=name)

  elif isinstance(space, spaces.Tuple):
    return tuple(space2spec(s, name) for s in space.spaces)

  elif isinstance(space, spaces.Dict):
    return {key: space2spec(value, name) for key, value in space.spaces.items()}

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))

