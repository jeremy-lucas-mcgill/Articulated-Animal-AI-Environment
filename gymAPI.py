import gym
from gym import spaces
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

class UnityEnvAPI(gym.Env):
    def __init__(self, file_name, worker_id=0, no_graphics=False):
        super(UnityEnvAPI, self).__init__()
        self.env = UnityEnvironment(file_name=file_name, worker_id=worker_id, no_graphics=no_graphics)
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.spec.action_spec.continuous_size,), dtype=np.float32)
        
        obs_specs = self.spec.observation_specs
        total_obs_size = 0
        for obs_spec in obs_specs:
            obs_size = np.prod(obs_spec.shape)
            total_obs_size += obs_size
            
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32)
        
    def reset(self):
        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        first_obs = self._get_obs(decision_steps)
        return first_obs

    def step(self, action):
        action_tuple = ActionTuple(continuous=np.array([action]).astype(np.float32))
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        
        if len(decision_steps) > 0:
            obs = self._get_obs(decision_steps)
            reward = decision_steps.reward[0]
            done = False
        else:
            obs = self._get_obs(terminal_steps)
            reward = terminal_steps.reward[0]
            done = True
        
        return obs, reward, done, {}
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        self.env.close()

    def _get_obs(self, decision_steps):
        obs = []
        for obs_spec in decision_steps.obs:
            obs_flat = obs_spec.flatten()
            obs.append(obs_flat)
        return np.concatenate(obs)