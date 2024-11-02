class R_Actor(nn.Module):
    def __init__(self, obs_space, action_space):
        super(R_Actor, self).__init__()

        self._gain = 0.01
        self._use_orthogonal = True

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = CNNBase(obs_shape, cnn_layers_params='32,3,1,1 64,3,1,1 32,3,1,1')
        
        input_size = self.base.output_size
        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)

        self.cuda()

    def forward(self, obs):
        actor_features = self.base(obs)
        actions, action_log_probs = self.act(actor_features)
        
        return actions, action_log_probs
        

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == 'Dict':
        obs_shape = obs_space.spaces
    else:
        raise NotImplementedError
    return obs_shape