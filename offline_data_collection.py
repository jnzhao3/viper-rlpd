#! /usr/bin/env python
import dmcgym
import gym
import numpy as np
import tqdm
from absl import app, flags
from flax.core import FrozenDict
from ml_collections import config_flags

# import wandb
from rlpd.agents import DrQLearner
from rlpd.data import MemoryEfficientReplayBuffer, ReplayBuffer
from rlpd.data.vd4rl_datasets import VD4RLDataset
from rlpd.evaluation import evaluate
from rlpd.wrappers import WANDBVideo, wrap_pixels

from viperx_sim import env_reg
import pickle

env = env_reg.make_reach_task_env()

replaybuffer = MemoryEfficientReplayBuffer(env.observation_space, env.action_space, 1000000, pixel_keys=('camera_0',))

for i in tqdm.tqdm(range(50)):
    obs = env.reset()
    done = False
    while not done:
        
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replaybuffer.insert({ 'observations' : obs, 'actions': action, 'rewards': reward, 'masks' : mask, 'dones': done, 'next_observations': next_obs})
        # replaybuffer.add(obs, action, reward, next_obs, done)
        obs = next_obs

pickle.dump(replaybuffer, open('viperx_replaybuffer.pkl', 'wb'))
