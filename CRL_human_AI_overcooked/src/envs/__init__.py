from functools import partial
from envs.multiagentenv import MultiAgentEnv
import sys
import os
from .overcooked_old import Overcooked_Old, Overcooked_New

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["overcooked_old"] = partial(env_fn, env=Overcooked_Old)
REGISTRY["overcooked_new"] = partial(env_fn, env=Overcooked_New)
            
