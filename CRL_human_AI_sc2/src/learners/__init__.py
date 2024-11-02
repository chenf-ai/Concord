from .offpg_learner import OffPGLearner
from .offpg_learner_base import OffPGLearner_base

REGISTRY = {}
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["offpg_learner_base"] = OffPGLearner_base
