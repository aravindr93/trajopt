"""
Wrapper around a gym env that provides convenience functions
"""

import gym
import numpy as np
from typing import Union


class EnvSpec(object):
    def __init__(self, obs_dim, act_dim, horizon):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon


class GymEnv(object):
    def __init__(
        self,
        env: Union[str, gym.Env, callable],
        env_kwargs: Union[dict, None] = None,
        obs_mask: Union[np.ndarray, None] = None,
        act_repeat: int = 1,
        *args,
        **kwargs,
    ):
        """GymEnv class as a generic container for several types of low-level gym environments."""
        # get the correct env behavior
        if type(env) == str:
            env = gym.make(env)
        elif isinstance(env, gym.Env):
            env = env
        elif callable(env):
            env = env(**env_kwargs)
        else:
            print("Unsupported environment format")
            raise AttributeError

        self.env = env
        self.env_id = env.unwrapped.spec.id
        self.act_repeat = act_repeat

        try:
            self._horizon = env.spec.max_episode_steps
        except AttributeError:
            self._horizon = env.spec._horizon

        assert self._horizon % act_repeat == 0
        self._horizon = self._horizon // self.act_repeat

        try:
            self._action_dim = self.env.action_space.shape[0]
        except AttributeError:
            self._action_dim = self.env.unwrapped.action_dim

        try:
            self._observation_dim = self.env.observation_space.shape[0]
        except AttributeError:
            self._observation_dim = self.env.unwrapped.obs_dim

        # Specs
        self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon)

        # obs mask
        self.obs_mask = np.ones(self._observation_dim) if obs_mask is None else obs_mask

    def reset(self, *args, **kwargs):
        try:
            return self.env.reset()
        except:
            return self.env.unwrapped.reset()

    def reset_model(self, seed=None):
        # overloading for legacy code
        if seed is not None:
            self.set_seed(seed)
        return self.reset(seed)

    def step(self, action: np.ndarray):
        # If action space has bounds, enforce it here by clipping
        if hasattr(self.action_space, "low") and hasattr(self.action_space, "high"):
            action = action.clip(self.action_space.low, self.action_space.high)

        # Step low-level environment based on desired frameskip/action-repeat
        if self.act_repeat == 1:
            obs, cum_reward, done, ifo = self.env.step(action)
        else:
            cum_reward = 0.0
            for i in range(self.act_repeat):
                obs, reward, done, ifo = self.env.step(action)
                cum_reward += reward
                if done:
                    break
        return self.obs_mask * obs, cum_reward, done, ifo

    def render(self):
        if hasattr(self.env.unwrapped, "mj_render"):
            # Specific to some mujoco_py environments
            self.env.unwrapped.mujoco_render_frames = True
            return self.env.unwrapped.mj_render()
        else:
            self.env.render()

    def set_seed(self, seed=123):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        elif hasattr(self.env, "_seed"):
            return self.env._seed(seed)
        elif hasattr(self.env.unwrapped, "seed"):
            return self.env.unwrapped.seed(seed)
        elif hasattr(self.env.unwrapped, "_seed"):
            return self.env.unwrapped._seed(seed)
        else:
            print("No automatic way to set seed. Environment needs inspection.")
            raise NotImplementedError

    def get_obs(self):
        if hasattr(self.env, "get_obs"):
            return self.obs_mask * self.env.get_obs()
        elif hasattr(self.env, "_get_obs"):
            return self.obs_mask * self.env._get_obs()
        else:
            print("No automatic way to get observation.")
            raise NotImplementedError

    def get_env_infos(self):
        if hasattr(self.env.unwrapped, "get_env_infos"):
            return self.env.unwrapped.get_env_infos()
        else:
            return {}

    # ===========================================
    # Properties

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def horizon(self):
        return self._horizon

    # ===========================================
    # Trajectory optimization related
    # Envs should support these functions in case of trajopt

    def get_env_state(self):
        try:
            return self.env.unwrapped.get_env_state()
        except:
            raise NotImplementedError

    def set_env_state(self, state_dict):
        try:
            self.env.unwrapped.set_env_state(state_dict)
        except:
            raise NotImplementedError

    def real_env_step(self, bool_val):
        try:
            self.env.unwrapped.real_step = bool_val
        except:
            raise NotImplementedError
