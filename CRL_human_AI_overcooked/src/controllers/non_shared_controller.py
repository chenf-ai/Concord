from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.human.policy_prefrence import Human_Policy
import torch as th
import numpy as np
import os

class NonSharedMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)

        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        
        self.init_human_model()
            
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        assert self.n_agents == 2
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        # print(ep_batch["obs"].shape)
        agent_inputs, agent_Human_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        # human output
        if self.args.vecobs_dim != -1:
            agent_inputs = th.cat([agent_inputs[:, :self.args.vecobs_dim], agent_inputs[:, ep_batch['obs'].shape[-1]:]], dim=1)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        
        if self.args.env_args['map_name'] == 'unident_s8':
            map_height = 6
            map_channel = 20
        elif self.args.env_args['map_name'] == 'many_orders':
            map_height = 5
            map_channel = 26
        elif self.args.env_args['map_name'] == 'unident_open':
            map_height = 5
            map_channel = 26
        else: 
            map_height = 5
            map_channel = 20
            
        if self.args.vecobs_dim != -1:
            agent_Human_inputs = agent_Human_inputs[:, :ep_batch['obs'].shape[-1]][:, self.args.vecobs_dim:].reshape(agent_Human_inputs.shape[0], -1, map_height, map_channel)
        else:
            agent_Human_inputs = agent_Human_inputs[:, :ep_batch['obs'].shape[-1]].reshape(agent_Human_inputs.shape[0], -1, map_height, 20)
        agent_Human_outs_not_onehot, self.human_actor_rnn_state = self.policy_h.act(
                                                        agent_Human_inputs, 
                                                        self.human_actor_rnn_state, 
                                                        self.mask)
        agent_Human_outs = th.zeros((agent_Human_outs_not_onehot.shape[0], 6), dtype=th.float32).cuda()
        agent_Human_outs.scatter_(1, agent_Human_outs_not_onehot, 1).view(ep_batch.batch_size, 1, -1)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            agent_Human_outs = th.nn.functional.softmax(agent_Human_outs, dim=-1)
            
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)
        
        return th.cat([agent_outs.view(ep_batch.batch_size, 1, -1), \
            agent_Human_outs.view(ep_batch.batch_size, 1, -1)], dim=1)
    
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)  # bav
        self.human_actor_rnn_state = np.zeros((batch_size, 1, 64), dtype=np.float32)
        self.mask = np.ones((batch_size, 1), dtype=np.float32)

    def parameters(self):
        return list(self.agent.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        
    def init_human_model(self, ckpt_path=None):
        if ckpt_path==None:
            ckpt_path = self.args.human_model
        if ckpt_path == '':
            pass
        else:
            self.policy_h = Human_Policy(self.args)
            self.policy_h.load_checkpoint(ckpt_path)
            self.policy_h.prep_rollout()

    def cuda(self):
        self.agent.cuda()
        # self.policy_h.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)
        inputs = inputs.permute(1,0,2)
        return inputs[0], inputs[1]

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.vecobs_dim != -1:
            input_shape = self.args.vecobs_dim
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape