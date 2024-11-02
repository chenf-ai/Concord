import numpy as np
import torch
from modules.human.r_actor import R_Actor

class Human_Policy():
    def __init__(self, args):
        device = torch.device("cuda:0")
        if args.env_args['map_name'] == 'random1':
            self.actor = R_Actor((5, 5, 20), 6, device)
        elif args.env_args['map_name'] == 'unident_s8':
            self.actor = R_Actor((9, 6, 20), 6, device)
        elif args.env_args['map_name'] == 'many_orders':
            self.actor = R_Actor((5, 5, 26), 6, device)
        elif args.env_args['map_name'] == 'unident_open':
            self.actor = R_Actor((9, 5, 26), 6, device)
        self.device = device
        
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
        
    def load_checkpoint(self, ckpt_path):
        self.actor.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        
    def prep_rollout(self):
        self.actor.eval()
    
    def predict():
        pass