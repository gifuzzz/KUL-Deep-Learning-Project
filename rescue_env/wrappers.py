"""Environment wrappers for FlameBox rescue environments."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PreviousActionWrapper(gym.Wrapper):
    """Tracks the previous action and includes it in the observation dict.

    - At episode start (reset): previous_action is initialized to 0.
    - At step t > 0: previous_action is the action executed at step t-1.
    - Space: Box(low=0, high=num_actions-1, shape=(), dtype=np.int64) or Discrete(num_actions).
    """

    def __init__(self, env: gym.Env, num_actions: int = 7):
        super().__init__(env)
        self.num_actions = num_actions

        obs_spaces = dict(self.env.observation_space.spaces)
        obs_spaces["previous_action"] = spaces.Box(
            low=0, high=self.num_actions - 1, shape=(), dtype=np.int64
        )
        self.observation_space = spaces.Dict(obs_spaces)
        self.previous_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_action = 0
        obs["previous_action"] = np.int64(0)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.previous_action = int(action)
        obs["previous_action"] = np.int64(action)
        if "terminal_observation" in info and isinstance(info["terminal_observation"], dict):
            info["terminal_observation"]["previous_action"] = np.int64(action)
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def action_masks(self) -> np.ndarray:
        if hasattr(self.env, "action_masks"):
            try:
                return self.env.action_masks()
            except RuntimeError:
                return np.ones(self.num_actions, dtype=bool)
        if hasattr(self.env.unwrapped, "action_masks"):
            try:
                return self.env.unwrapped.action_masks()
            except RuntimeError:
                return np.ones(self.num_actions, dtype=bool)
        return np.ones(self.num_actions, dtype=bool)
