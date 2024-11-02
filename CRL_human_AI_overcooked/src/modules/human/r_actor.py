import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from modules.human.util import init, check
from modules.human.mlp import MLPLayer
from modules.human.rnn import RNNLayer
from modules.human.act import ACTLayer
from modules.human.cnn import CNNBase

class R_Actor(nn.Module):
    def __init__(self,obs_space, action_dim, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = 64

        self._gain = 0.01
        self._use_orthogonal = True 
        self._activation_id = 1
        self._use_policy_active_masks = True 
        self._use_naive_recurrent_policy = False
        self._use_recurrent_policy = False
        self._use_influence_policy = False
        self._influence_layer_N = 1 
        self._use_policy_vhead = False
        self._use_popart = False 
        self._recurrent_N = 1
        self._cnn_layers_params = '32,3,1,1 64,3,1,1 32,3,1,1'
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        action_space = spaces.Discrete(action_dim)
        obs_shape = obs_space
        hyperparameters = {}
        hyperparameters['use_orthogonal'] = self._use_orthogonal
        hyperparameters['activation_id'] = self._activation_id
        hyperparameters['use_maxpool2d'] = False
        hyperparameters['hidden_size'] = self.hidden_size
        
        self._mixed_obs = False
        self.base = CNNBase(hyperparameters, obs_shape, cnn_layers_params=self._cnn_layers_params)
        
        
        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            input_size += self.hidden_size

        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        if self._use_policy_vhead:
            if self._use_popart:
                self.v_out = init_(PopArt(input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(input_size, 1))
        
        # in Overcooked, predict shaped info
        self._predict_other_shaped_info = False
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):        
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        #print('obs:',obs.shape) ([2, 9, 5, 20])
        #print('result:', np.array(obs[0].cpu().numpy())-np.array(obs[1].cpu().numpy()))
        actor_features = self.base(obs)
        #print('actor_features:',actor_features.shape) #[2, 64]
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        
        return actions, action_log_probs, rnn_states


