REGISTRY = {}

from .rnn_agent import RNNAgent, Concord, RNNAgent_EWC
REGISTRY["rnn"] = RNNAgent
REGISTRY["Concord"] = Concord
REGISTRY["rnn_ewc"] = RNNAgent_EWC

