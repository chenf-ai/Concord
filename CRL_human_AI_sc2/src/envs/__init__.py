from functools import partial
from envs.multiagentenv import MultiAgentEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
from smac.env import StarCraft2Env
os.environ.setdefault("SC2PATH","../Env/SMAC/3rdparty/StarCraftII")
REGISTRY["sc2"] = env = partial(env_fn, env=StarCraft2Env)
