import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


class UnityEnvAPI(gym.Env):
    def __init__(self, file_name, worker_id=0, no_graphics=False, max_steps=5000,additional_args=None):
        super(UnityEnvAPI, self).__init__()
        if additional_args is None:
            additional_args = []
        self.env = UnityEnvironment(
            file_name=file_name, worker_id=worker_id, no_graphics=no_graphics,additional_args=additional_args
        )
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.spec.action_spec.continuous_size,),
            dtype=np.float32,
        )

        obs_specs = self.spec.observation_specs
        total_obs_size = 0
        for obs_spec in obs_specs:
            obs_size = np.prod(obs_spec.shape)
            total_obs_size += obs_size

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )
        self.max_steps = max_steps
        self.current_epoch_steps = 0

    def reset(self, seed=None):
        self.env.reset()
        self.current_epoch_steps = 0
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        first_obs = self._get_obs(decision_steps)
        print(f"First Observation: {first_obs.shape}")
        return first_obs, {}

    def step(self, action):
        action_tuple = ActionTuple(continuous=np.array([action]).astype(np.float32))
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()
        self.current_epoch_steps += 1
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        if len(decision_steps) > 0 and self.current_epoch_steps < self.max_steps:
            obs = self._get_obs(decision_steps)
            reward = decision_steps.reward[0]
            done = False
        else:
            if len(terminal_steps) > 0:
                obs = self._get_obs(terminal_steps)
                reward = terminal_steps.reward[0]
            else:
                obs = self._get_obs(decision_steps)
                reward = decision_steps.reward[0]
            done = True
            if len(decision_steps) > 0:
                self.reset()

        truncated = done  # always truncated since there is no terminal condition other than time

        return obs, reward, done, truncated, {}

    def render(self, mode="human"):
        pass

    def close(self):
        self.env.close()

    def _get_obs(self, decision_steps):
        obs = []
        for obs_spec in decision_steps.obs:
            obs_flat = obs_spec.flatten()
            obs.append(obs_flat)
        return np.concatenate(obs)


def make_unity_env(**kwargs):
    if "no_graphics" not in kwargs:
        kwargs["no_graphics"] = False
    if "file_name" not in kwargs:
        kwargs["file_name"] = "Build/RL Environment"
    if "additional_args" not in kwargs:
        kwargs["additional_args"] = []
    return UnityEnvAPI(**kwargs)