from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np
import os

class NonSharedMACHyper:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        
        self.policy_h = agent_REGISTRY['rnn'](input_shape, self.args)
        self.policy_h2 = agent_REGISTRY['rnn'](input_shape, self.args)
        self.init_human_model()
        
        self.task_id = 0
        self.recognizer = None
        self.given_embedding = None
    
    def set_task_id(self, task_id):
        
        self.task_id = task_id

    def set_given_embedding(self, given_embedding):
        
        self.given_embedding = given_embedding

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, use_old=False):
        
        assert self.n_agents <= 5
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):

        agent_inputs, agent_Human_inputs, agent_Human_inputs2 = self._build_inputs(ep_batch, t)  
        agent_outs, self.hidden_states = self.agent(self.task_id, agent_inputs, self.hidden_states, given_embedding=self.given_embedding)
        agent_Human_outs, self.human_actor_rnn_state = self.policy_h(agent_Human_inputs, self.human_actor_rnn_state)
        if self.args.human_num == 2:
            agent_Human_outs2, self.human_actor_rnn_state2 = self.policy_h2(agent_Human_inputs2, self.human_actor_rnn_state2)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            agent_Human_outs = th.nn.functional.softmax(agent_Human_outs, dim=-1)
            if self.args.human_num == 2:
                agent_Human_outs2 = th.nn.functional.softmax(agent_Human_outs2, dim=-1)
            
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

        if self.args.human_num == 1:
            return th.cat([agent_outs.view(ep_batch.batch_size, self.n_agents - 1, -1), \
                        agent_Human_outs.view(ep_batch.batch_size, 1, -1),], dim=1)
        elif self.args.human_num == 2:
            return th.cat([agent_outs.view(ep_batch.batch_size, self.n_agents - 2, -1), \
                        agent_Human_outs.view(ep_batch.batch_size, 1, -1),
                        agent_Human_outs2.view(ep_batch.batch_size, 1, -1),], dim=1)
    
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents - self.args.human_num, -1)
        self.human_actor_rnn_state = self.policy_h.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)
        self.human_actor_rnn_state2 = self.policy_h2.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)

    def parameters(self):
        return list(self.agent.hypernet.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        
    def init_human_model(self, ckpt_path=None):
        if ckpt_path == '' or ckpt_path is None or ckpt_path == []:
            return
        elif self.args.human_num == 1:
            self.policy_h.load_state_dict(th.load(ckpt_path))
        elif self.args.human_num == 2:
            self.policy_h.load_state_dict(th.load(ckpt_path[0]))
            self.policy_h2.load_state_dict(th.load(ckpt_path[1]))

    def cuda(self):
        self.agent.cuda()
        self.policy_h.cuda()
        self.policy_h2.cuda()

    def save_models(self, path):
        th.save(self.agent.hypernet.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.hypernet.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        assert self.args.human_num < self.n_agents
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
        if self.args.human_num == 1:
            return inputs[:, :-1].reshape(bs * (self.n_agents - 1), -1), \
                   inputs[:, -1:].reshape(bs, -1), None
        elif self.args.human_num == 2:
            return inputs[:, :-2].reshape(bs * (self.n_agents - 2), -1), \
                   inputs[:, -2:-1].reshape(bs, -1), \
                   inputs[:, -1:].reshape(bs, -1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape