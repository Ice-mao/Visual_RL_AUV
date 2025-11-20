import os
import gym
import numpy as np
from dm_env import StepType, specs
import dm_env
import numpy as np
from gym import spaces
from typing import Any, NamedTuple
from collections import deque

import sys
sys.path.append('/data/RL/DrM/RL_AUV_tracking')
import RL_AUV_tracking.auv_env as auv_env
from RL_AUV_tracking.config_loader import load_config

class AuvEnv:
    def __init__(
        self,
        name='AUVTracking_v1',
        config=None,
        seed=None,
        action_repeat=1,
        size = (64, 64)
        ):
        config_path = os.path.join('/data/RL/DrM/RL_AUV_tracking', 'configs','envs', f'{config}.yml')
        config = load_config(config_path)
        config['obs']['image'] = list(size)
        
        self._env = auv_env.make(name, 
                        config=config,
                        eval=False, t_steps=1000,
                        show_viewport=False,
                        )
        
        self._size = size
        self._action_repeat = action_repeat
        
        if seed is not None:
            np.random.seed(seed)

    @property
    def obs_space(self):
        # AUV environment returns dict with images and state
        spaces = {
            "image": self._env.observation_space['image'],
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space['state'],
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        return {"action": self._env.action_space}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        success = 0.0
        info = {}
        
        for _ in range(self._action_repeat):
            obs, rew, done, _, info = self._env.step(action["action"])
            reward += rew or 0.0
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,
            "is_terminal": False,
            "image": obs['image'],
            "state": obs['state'],
            "success": success,
        }
        return obs

    def reset(self):
        obs, info = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": obs['image'],
            "state": obs['state'],
            "success": False,
        }
        return obs

class NormalizeAction:
    def __init__(self, env, key="action"):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})

class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    success: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)
        
class auv_wrapper():
    def __init__(self, env, nstack=3):
        self._env = env
        self.nstack = nstack
        
        wos = env.obs_space['image']
        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)
        self.stackedobs = np.zeros(low.shape, low.dtype)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.uint8)

    def observation_spec(self):
        return specs.BoundedArray(self.observation_space.shape,
                                  np.uint8,
                                  0,
                                  255,
                                  name='observation')

    def action_spec(self):
        return specs.BoundedArray(self._env.act_space['action'].shape,
                                  np.float32,
                                  self._env.act_space['action'].low,
                                  self._env.act_space['action'].high,
                                  'action')

    def reset(self):
        time_step = self._env.reset()
        obs = time_step['image']  #(C, H, W)
        self.stackedobs[...] = 0
        self.stackedobs[-obs.shape[0]:, ...] = obs
        return ExtendedTimeStep(observation=self.stackedobs,
                                step_type=StepType.FIRST,
                                action=np.zeros(self.action_spec().shape, dtype=self.action_spec().dtype),
                                reward=0.0,
                                discount=1.0,
                                success=time_step['success'])

    def step(self, action):
        action = {'action':action}
        time_step = self._env.step(action)
        obs = time_step['image']  #(C, H, W)
        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[0], axis=0)
        self.stackedobs[-obs.shape[0]:, ...] = obs

        if time_step['is_first']:
            step_type = StepType.FIRST
        elif time_step['is_last']:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        return ExtendedTimeStep(observation=self.stackedobs,
                                 step_type=step_type,
                                 action=action['action'],
                                 reward=time_step['reward'],
                                 discount=1.0,
                                 success=time_step['success'])


def make(name, frame_stack, action_repeat, seed, config=None):
    """
    Create an AUV tracking environment compatible with DrM framework.
    
    Args:
        name: Environment name (e.g., 'AUVTracking_v1')
        frame_stack: Number of frames to stack
        action_repeat: Number of times to repeat each action
        seed: Random seed
        config: Optional environment configuration
    """
    # Create AUV environment
    env = AuvEnv(
        name=name, 
        seed=seed, 
        action_repeat=action_repeat, 
        size=(84, 84),
        config=config
    )
    
    # Apply wrappers
    env = NormalizeAction(env)
    env = TimeLimit(env, 200)  # Standard episode length for AUV tracking
    env = auv_wrapper(env, frame_stack)

    return env

if __name__ == "__main__":
    env = make('AUVTracking_v1', frame_stack=3, action_repeat=2, seed=42, config='v1_config')
    print("metaworld env created successfully!")