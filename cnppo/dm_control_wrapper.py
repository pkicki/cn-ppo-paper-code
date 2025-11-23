import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite


class DMControlGymWrapper(gym.Env):
    """
    Minimal Gymnasium wrapper for DeepMind Control Suite environments.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, domain, task, seed=None):
        super().__init__()
        self._env = suite.load(domain, task, task_kwargs={"random": seed})

        # Observation: flatten dm_control dict
        obs_spec = self._env.observation_spec()
        obs_size = sum(np.prod(v.shape) for v in obs_spec.values())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action space: continuous [-1, 1]
        act_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=act_spec.minimum,
            high=act_spec.maximum,
            dtype=np.float32
        )

    def _flatten_obs(self, obs):
        return np.concatenate([v.ravel() for v in obs.values()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._env.task._random = np.random.RandomState(seed)

        timestep = self._env.reset()
        obs = self._flatten_obs(timestep.observation)
        return obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        timestep = self._env.step(action)

        obs = self._flatten_obs(timestep.observation)
        reward = timestep.reward or 0.0
        terminated = timestep.last()
        truncated = False  # dm_control has no truncation

        return obs, reward, terminated, truncated, {}

    def render(self):
        return self._env.physics.render(camera_id=0)

