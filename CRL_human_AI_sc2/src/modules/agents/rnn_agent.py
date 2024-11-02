import torch.nn as nn
import torch.nn.functional as F
import torch as th
import math
import copy


# Network for vanilla and ER (CLEAR)
class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, 64)
        self.rnn = nn.GRUCell(64, 64)
        self.fc2 = nn.Linear(64, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, 64).zero_()

    def forward(self, inputs, hidden_state):
        
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, 64)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

# Network for EWC
class RNNAgent_EWC(nn.Module):
    def __init__(self, input_shape, args, task_num=12):
        super(RNNAgent_EWC, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.EWC_gamma = 1.0
        self.initialize_fisher()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        
        return q, h

    def initialize_fisher(self):
        '''Initialize diagonal fisher matrix with the prior precision'''
        self.est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_EWC_prev_context'.format(n), p.detach().clone()*0)
                self.register_buffer( '{}_EWC_estimated_fisher'.format(n), th.zeros(p.shape))
                
    def estimate_fisher(self):
        if self.est_fisher_info == {}:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    self.est_fisher_info[n] = p.detach().clone().zero_().cuda()
        
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if p.grad is not None:
                    self.est_fisher_info[n] += (p.grad.detach() ** 2) / 320

    def keep_old_param(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_EWC_prev_context'.format(n), p.detach().clone())
                existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                self.est_fisher_info[n] += self.EWC_gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher'.format(n), self.est_fisher_info[n])

        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.est_fisher_info[n] = p.detach().clone().zero_().cuda()

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.est_fisher_info == {}:
            return th.tensor(0, dtype=th.float32).cuda()
        
        losses = []
        # If "offline EWC", loop over all previous contexts as each context has separate penalty term
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                target_p = getattr(self, '{}_EWC_prev_context'.format(n))
                fisher = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                losses.append((fisher * (p - target_p)**2).sum())
        
        return (1./2)*sum(losses)

# Network for recoginzer of Concord
class Recognizer(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim):
        super(Recognizer, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.drop1 = nn.Dropout(0.3)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, 32)
        self.drop2 = nn.Dropout(0.3)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, fuse=False):
        
        x = F.relu(self.fc1(inputs))
        x = self.drop1(x)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        if fuse:
           h_in = h_in.mean(0).repeat(h_in.shape[0], 1)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

# Network for hypernet of Concord
class Hypernet(nn.Module):
    def __init__(self, input_shape, args, task_num=12):
        super(Hypernet, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.act = F.elu
        self.act1 = th.nn.Identity()
        self.hyper_hdim = 20
        self.embed_hdim = args.embed_hdim
        
        self.embed = nn.ParameterList([nn.Parameter(data=th.Tensor(self.embed_hdim), requires_grad=True) for i in range(task_num)])
        
        # fc1
        self.embed_fc1_1_weight = nn.Parameter(data=th.Tensor(self.hyper_hdim, self.embed_hdim), requires_grad=True)
        self.embed_fc1_1_bias = nn.Parameter(data=th.Tensor(self.hyper_hdim), requires_grad=True)
        self.embed_fc1_2_weight = nn.Parameter(data=th.Tensor(self.hyper_hdim, self.hyper_hdim), requires_grad=True)
        self.embed_fc1_2_bias = nn.Parameter(data=th.Tensor(self.hyper_hdim), requires_grad=True)
        self.embed_head_fc1weight_weight = nn.Parameter(data=th.Tensor(input_shape*64, self.hyper_hdim), requires_grad=True)
        self.embed_head_fc1weight_bias = nn.Parameter(data=th.Tensor(input_shape*64), requires_grad=True)
        self.embed_head_fc1bias_weight = nn.Parameter(data=th.Tensor(64, self.hyper_hdim), requires_grad=True)
        self.embed_head_fc1bias_bias = nn.Parameter(data=th.Tensor(64), requires_grad=True)
            
        # fc2
        self.embed_fc2_1_weight = nn.Parameter(data=th.Tensor(self.hyper_hdim, self.embed_hdim), requires_grad=True)
        self.embed_fc2_1_bias = nn.Parameter(data=th.Tensor(self.hyper_hdim), requires_grad=True)
        self.embed_fc2_2_weight = nn.Parameter(data=th.Tensor(self.hyper_hdim, self.hyper_hdim), requires_grad=True)
        self.embed_fc2_2_bias = nn.Parameter(data=th.Tensor(self.hyper_hdim), requires_grad=True)
        self.embed_head_fc2weight_weight = nn.Parameter(data=th.Tensor(args.n_actions*64, self.hyper_hdim), requires_grad=True)
        self.embed_head_fc2weight_bias = nn.Parameter(data=th.Tensor(args.n_actions*64), requires_grad=True)
        self.embed_head_fc2bias_weight = nn.Parameter(data=th.Tensor(args.n_actions, self.hyper_hdim), requires_grad=True)
        self.embed_head_fc2bias_bias = nn.Parameter(data=th.Tensor(args.n_actions), requires_grad=True)
            
        # gru_ih_r gru_hh_r
        self.embed_gru_1_weight = nn.Parameter(data=th.Tensor(self.hyper_hdim, self.embed_hdim), requires_grad=True)
        self.embed_gru_1_bias = nn.Parameter(data=th.Tensor(self.hyper_hdim), requires_grad=True)
        self.embed_gru_2_weight = nn.Parameter(data=th.Tensor(self.hyper_hdim, self.hyper_hdim), requires_grad=True)
        self.embed_gru_2_bias = nn.Parameter(data=th.Tensor(self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_ihweight_r_weight = nn.Parameter(data=th.Tensor(4096, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_ihweight_r_bias = nn.Parameter(data=th.Tensor(4096), requires_grad=True)
        self.embed_head_gru_ihbias_r_weight = nn.Parameter(data=th.Tensor(64, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_ihbias_r_bias = nn.Parameter(data=th.Tensor(64), requires_grad=True)
        self.embed_head_gru_hhweight_r_weight = nn.Parameter(data=th.Tensor(4096, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_hhweight_r_bias = nn.Parameter(data=th.Tensor(4096), requires_grad=True)
        self.embed_head_gru_hhbias_r_weight = nn.Parameter(data=th.Tensor(64, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_hhbias_r_bias = nn.Parameter(data=th.Tensor(64), requires_grad=True)         

        # gru_ih_z gru_hh_z      
        self.embed_head_gru_ihweight_z_weight = nn.Parameter(data=th.Tensor(4096, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_ihweight_z_bias = nn.Parameter(data=th.Tensor(4096), requires_grad=True)
        self.embed_head_gru_ihbias_z_weight = nn.Parameter(data=th.Tensor(64, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_ihbias_z_bias = nn.Parameter(data=th.Tensor(64), requires_grad=True)
        self.embed_head_gru_hhweight_z_weight = nn.Parameter(data=th.Tensor(4096, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_hhweight_z_bias = nn.Parameter(data=th.Tensor(4096), requires_grad=True)
        self.embed_head_gru_hhbias_z_weight = nn.Parameter(data=th.Tensor(64, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_hhbias_z_bias = nn.Parameter(data=th.Tensor(64), requires_grad=True)
            
        # gru_ih_n gru_hh_n
        self.embed_head_gru_ihweight_n_weight = nn.Parameter(data=th.Tensor(4096, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_ihweight_n_bias = nn.Parameter(data=th.Tensor(4096), requires_grad=True)
        self.embed_head_gru_ihbias_n_weight = nn.Parameter(data=th.Tensor(64, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_ihbias_n_bias = nn.Parameter(data=th.Tensor(64), requires_grad=True)
        self.embed_head_gru_hhweight_n_weight = nn.Parameter(data=th.Tensor(4096, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_hhweight_n_bias = nn.Parameter(data=th.Tensor(4096), requires_grad=True)
        self.embed_head_gru_hhbias_n_weight = nn.Parameter(data=th.Tensor(64, self.hyper_hdim), requires_grad=True)
        self.embed_head_gru_hhbias_n_bias = nn.Parameter(data=th.Tensor(64), requires_grad=True)

        self.reset_parameters()
    
    def forward(self, task_id, given_embedding=None, require_grad=True):

        if given_embedding is None:
            if require_grad:
                embed_task = self.embed[task_id]
            else:
                embed_task = self.embed[task_id].clone().detach()
        else:
            embed_task = given_embedding

        a_fc1 = self.act(F.linear(embed_task, self.embed_fc1_1_weight, bias=self.embed_fc1_1_bias))
        a_fc1 = self.act(F.linear(a_fc1, self.embed_fc1_2_weight, bias=self.embed_fc1_2_bias))
        fc1_weight = self.act1(F.linear(a_fc1, self.embed_head_fc1weight_weight, bias=self.embed_head_fc1weight_bias)).reshape(64, self.input_shape)
        fc1_bias = self.act1(F.linear(a_fc1, self.embed_head_fc1bias_weight, bias=self.embed_head_fc1bias_bias))
        
        a_gru = self.act(F.linear(embed_task, self.embed_gru_1_weight, bias=self.embed_gru_1_bias))
        a_gru = self.act(F.linear(a_gru, self.embed_gru_2_weight, bias=self.embed_gru_2_bias))
        gru_weight_ih_r = self.act1(F.linear(a_gru, self.embed_head_gru_ihweight_r_weight, bias=self.embed_head_gru_ihweight_r_bias)).reshape(64, 64)
        gru_bias_ih_r = self.act1(F.linear(a_gru, self.embed_head_gru_ihbias_r_weight, bias=self.embed_head_gru_ihbias_r_bias))
        gru_weight_hh_r = self.act1(F.linear(a_gru, self.embed_head_gru_hhweight_r_weight, bias=self.embed_head_gru_hhweight_r_bias)).reshape(64, 64)
        gru_bias_hh_r = self.act1(F.linear(a_gru, self.embed_head_gru_hhbias_r_weight, bias=self.embed_head_gru_hhbias_r_bias))
        gru_weight_ih_z = self.act1(F.linear(a_gru, self.embed_head_gru_ihweight_z_weight, bias=self.embed_head_gru_ihweight_z_bias)).reshape(64, 64)
        gru_bias_ih_z = self.act1(F.linear(a_gru, self.embed_head_gru_ihbias_z_weight, bias=self.embed_head_gru_ihbias_z_bias))
        gru_weight_hh_z = self.act1(F.linear(a_gru, self.embed_head_gru_hhweight_z_weight, bias=self.embed_head_gru_hhweight_z_bias)).reshape(64, 64)
        gru_bias_hh_z = self.act1(F.linear(a_gru, self.embed_head_gru_hhbias_z_weight, bias=self.embed_head_gru_hhbias_z_bias))
        gru_weight_ih_n = self.act1(F.linear(a_gru, self.embed_head_gru_ihweight_n_weight, bias=self.embed_head_gru_ihweight_n_bias)).reshape(64, 64)
        gru_bias_ih_n = self.act1(F.linear(a_gru, self.embed_head_gru_ihbias_n_weight, bias=self.embed_head_gru_ihbias_n_bias))
        gru_weight_hh_n = self.act1(F.linear(a_gru, self.embed_head_gru_hhweight_n_weight, bias=self.embed_head_gru_hhweight_n_bias)).reshape(64, 64)
        gru_bias_hh_n = self.act1(F.linear(a_gru, self.embed_head_gru_hhbias_n_weight, bias=self.embed_head_gru_hhbias_n_bias))
        
        a_fc2 = self.act(F.linear(embed_task, self.embed_fc2_1_weight, bias=self.embed_fc2_1_bias))
        a_fc2 = self.act(F.linear(a_fc2, self.embed_fc2_2_weight, bias=self.embed_fc2_2_bias))
        fc2_weight = self.act1(F.linear(a_fc2, self.embed_head_fc2weight_weight, bias=self.embed_head_fc2weight_bias)).reshape(self.args.n_actions, 64)
        fc2_bias = self.act1(F.linear(a_fc2, self.embed_head_fc2bias_weight, bias=self.embed_head_fc2bias_bias))
        
        return fc1_weight, fc1_bias, gru_weight_ih_r, gru_bias_ih_r, gru_weight_hh_r, gru_bias_hh_r, gru_weight_ih_z, gru_bias_ih_z, \
            gru_weight_hh_z, gru_bias_hh_z, gru_weight_ih_n, gru_bias_ih_n, gru_weight_hh_n, gru_bias_hh_n, fc2_weight, fc2_bias

    def reset_parameters(self):
        for weight in self.parameters():
            limit = weight.data.shape[-1]
            stdv = 1.0 / math.sqrt(limit) if self.args.rnn_hidden_dim > 0 else 0
            th.nn.init.uniform_(weight, -stdv, stdv)

# Concord
class Concord(nn.Module):
    def __init__(self, input_shape, args, task_num=12):
        super(Concord, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.task_num = task_num
        self.hypernet = Hypernet(input_shape, args, task_num)
        self.hypernet_old = None
        self.reg_loss = nn.MSELoss()

    def init_hidden(self):

        return self.hypernet.embed_fc1_1_weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def keep_old_hypernet(self):
        
        self.hypernet_old = copy.deepcopy(self.hypernet)
        self.hypernet_old.eval()
        self.target_param = [None] * self.task_num
 
    def forward(self, task_id, inputs, hidden_state, given_embedding=None):
        
        assert task_id < self.task_num
        
        fc1_weight, fc1_bias, gru_weight_ih_r, gru_bias_ih_r, gru_weight_hh_r, gru_bias_hh_r, gru_weight_ih_z, gru_bias_ih_z, \
        gru_weight_hh_z, gru_bias_hh_z, gru_weight_ih_n, gru_bias_ih_n, gru_weight_hh_n, gru_bias_hh_n, fc2_weight, fc2_bias \
        = self.hypernet(task_id, given_embedding)
        
        # RNNAgent by handle
        x = F.relu(F.linear(inputs, fc1_weight, bias=fc1_bias))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim).cuda()  
        r = F.sigmoid(F.linear(x, gru_weight_ih_r, bias=gru_bias_ih_r) \
                      + F.linear(h_in, gru_weight_hh_r, bias=gru_bias_hh_r))            
        z = F.sigmoid(F.linear(x, gru_weight_ih_z, bias=gru_bias_ih_z) \
                      + F.linear(h_in, gru_weight_hh_z, bias=gru_bias_hh_z))
        n = F.tanh(F.linear(x, gru_weight_ih_n, bias=gru_bias_ih_n) \
                      + th.mul(r, F.linear(h_in, gru_weight_hh_n, bias=gru_bias_hh_n)))        
        h = th.mul((1 - z), n) + th.mul(z, h_in)
        q = F.linear(h, fc2_weight, bias=fc2_bias)
        
        return q, h
    
    def regularize_loss(self, task_ids):

        if self.hypernet_old is None or len(task_ids) == 0:
            res = th.tensor(0, dtype=th.float32).cuda()
            return res
        
        loss = None
        for task_id in task_ids:
            
            if self.target_param[task_id] is None:
                fc1_weight_old, fc1_bias_old, gru_weight_ih_r_old, gru_bias_ih_r_old, gru_weight_hh_r_old, gru_bias_hh_r_old, gru_weight_ih_z_old, gru_bias_ih_z_old, \
                gru_weight_hh_z_old, gru_bias_hh_z_old, gru_weight_ih_n_old, gru_bias_ih_n_old, gru_weight_hh_n_old, gru_bias_hh_n_old, fc2_weight_old, fc2_bias_old \
                = self.hypernet_old(task_id, require_grad=False)
                
                param_old = [fc1_weight_old, fc1_bias_old, gru_weight_ih_r_old, gru_bias_ih_r_old, gru_weight_hh_r_old, gru_bias_hh_r_old, gru_weight_ih_z_old, gru_bias_ih_z_old,
                gru_weight_hh_z_old, gru_bias_hh_z_old, gru_weight_ih_n_old, gru_bias_ih_n_old, gru_weight_hh_n_old, gru_bias_hh_n_old, fc2_weight_old, fc2_bias_old]
                
                param_old = [p.reshape(-1) for p in param_old]
                self.target_param[task_id] = th.cat(param_old, dim=0).detach().clone()

            fc1_weight, fc1_bias, gru_weight_ih_r, gru_bias_ih_r, gru_weight_hh_r, gru_bias_hh_r, gru_weight_ih_z, gru_bias_ih_z, \
            gru_weight_hh_z, gru_bias_hh_z, gru_weight_ih_n, gru_bias_ih_n, gru_weight_hh_n, gru_bias_hh_n, fc2_weight, fc2_bias \
            = self.hypernet(task_id, require_grad=True)
            
            param = [fc1_weight, fc1_bias, gru_weight_ih_r, gru_bias_ih_r, gru_weight_hh_r, gru_bias_hh_r, gru_weight_ih_z, gru_bias_ih_z,
            gru_weight_hh_z, gru_bias_hh_z, gru_weight_ih_n, gru_bias_ih_n, gru_weight_hh_n, gru_bias_hh_n, fc2_weight, fc2_bias]
            
            param = [p.reshape(-1) for p in param]
            predicted_param = th.cat(param, dim=0)
        
            if loss is None:
                loss = self.reg_loss(self.target_param[task_id], predicted_param) * predicted_param.numel()
            else:
                loss += self.reg_loss(self.target_param[task_id], predicted_param) * predicted_param.numel()
        
        return loss / len(task_ids)