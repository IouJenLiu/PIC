import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from ddpg_vec import Actor, soft_update, hard_update, Actor, Critic, adjust_lr




class DDPGH(object):
    def __init__(self, gamma, tau, hidden_size, obs_dim, n_action, n_agent, obs_dims, agent_id, actor_lr, critic_lr,
                 fixed_lr, critic_type, train_noise, num_episodes, num_steps,
                 critic_dec_cen, target_update_mode='soft', device='cpu', groups=None):
        self.group_dim_id = [obs_dims[g] for g in groups]
        self.group_cum_id = np.cumsum([0] + groups)
        self.n_group = len(groups)
        self.device = device
        self.obs_dim = obs_dim
        self.n_agent = n_agent
        self.n_action = n_action
        self.actors = [Actor(hidden_size, self.group_dim_id[i], n_action).to(self.device) for i in range(len(groups))]
        self.actor_targets = [Actor(hidden_size, self.group_dim_id[i], n_action).to(self.device) for i in range(len(groups))]
        self.actor_optims = [Adam(self.actors[i].parameters(),
                                lr=actor_lr, weight_decay=0) for i in range(len(groups))]

        self.critic = Critic(hidden_size, np.sum(obs_dims),
                                 n_action * n_agent, n_agent, critic_type, agent_id, groups).to(self.device)
        self.critic_target = Critic(hidden_size, np.sum(
                obs_dims), n_action * n_agent, n_agent, critic_type, agent_id, groups).to(self.device)
        critic_n_params = sum(p.numel() for p in self.critic.parameters())
        print('# of critic params', critic_n_params)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.fixed_lr = fixed_lr
        self.init_act_lr = actor_lr
        self.init_critic_lr = critic_lr
        self.num_episodes = num_episodes
        self.start_episode = 0
        self.num_steps = num_steps
        self.gamma = gamma
        self.tau = tau
        self.train_noise = train_noise
        self.obs_dims_cumsum = np.cumsum(obs_dims)
        self.critic_dec_cen = critic_dec_cen
        self.agent_id = agent_id
        self.debug = False
        self.target_update_mode = target_update_mode
        self.actors_params = [self.actors[i].parameters() for i in range(self.n_group)]
        self.critic_params = self.critic.parameters()

        # Make sure target is with the same weight
        for i in range(self.n_group):
            hard_update(self.actor_targets[i], self.actors[i])
        hard_update(self.critic_target, self.critic)

    def adjust_lr(self, i_episode):
        for i in range(self.n_group):
            adjust_lr(self.actor_optims[i], self.init_act_lr, i_episode, self.num_episodes, self.start_episode)
        adjust_lr(self.critic_optim, self.init_critic_lr, i_episode, self.num_episodes, self.start_episode)

    def lambda1(self, step):
        start_decrease_step = ((self.num_episodes / 2)
                               * self.num_steps) / 100
        max_step = (self.num_episodes * self.num_steps) / 100
        return 1 - ((step - start_decrease_step) / (
                max_step - start_decrease_step)) if step > start_decrease_step else 1

    def select_action(self, state, action_noise=None, param_noise=False, grad=False):
        actions_l = []
        mus_l = []
        scale = int(state.size()[0] / self.n_agent)
        for i in range(self.n_group):
            act, mu = self.select_action_single(self.actors[i], state[scale * self.group_cum_id[i]:scale * self.group_cum_id[i+1]],
                                      action_noise, param_noise, grad)
            actions_l.append(act)
            mus_l.append(mu)
        if grad:
            return torch.cat(actions_l, dim=0), torch.cat(mus_l, dim=0)
        else:
            return torch.cat(actions_l, dim=0)

    def select_action_single(self, actor, state, action_noise=None, param_noise=False, grad=False):
        actor.eval()
        mu = actor((Variable(state)))

        actor.train()
        if not grad:
            mu = mu.data

        if action_noise:
            noise = np.log(-np.log(np.random.uniform(0, 1, mu.size())))
            try:
                mu -= torch.Tensor(noise).to(self.device)
            except (AttributeError, AssertionError):
                mu -= torch.Tensor(noise)
        action = F.softmax(mu, dim=1)
        return action, mu

    def update_critic_parameters(self, batch, agent_id, shuffle=None, eval=False):
        state_batch = Variable(torch.cat(batch.state)).to(self.device)
        action_batch = Variable(torch.cat(batch.action)).to(self.device)
        reward_batch = Variable(torch.cat(batch.reward)).to(self.device)
        mask_batch = Variable(torch.cat(batch.mask)).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        if shuffle == 'shuffle':
            rand_idx = np.random.permutation(self.n_agent)
            new_state_batch = state_batch.view(-1, self.n_agent, self.obs_dim)
            state_batch = new_state_batch[:, rand_idx, :].view(-1, self.obs_dim * self.n_agent)
            new_next_state_batch = next_state_batch.view(-1, self.n_agent, self.obs_dim)
            next_state_batch = new_next_state_batch[:, rand_idx, :].view(-1, self.obs_dim * self.n_agent)
            new_action_batch = action_batch.view(-1, self.n_agent, self.n_action)
            action_batch = new_action_batch[:, rand_idx, :].view(-1, self.n_action * self.n_agent)

        next_action_batch = self.select_action(
            next_state_batch.view(-1, self.obs_dim), action_noise=self.train_noise)
        next_action_batch = next_action_batch.view(-1, self.n_action * self.n_agent)
        next_state_action_values = self.critic_target(
                next_state_batch, next_action_batch)

        reward_batch = reward_batch[:, agent_id].unsqueeze(1)
        mask_batch = mask_batch[:, agent_id].unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)
        self.critic_optim.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        perturb_out = 0
        value_loss = ((state_action_batch - expected_state_action_batch) ** 2).mean()
        if eval:
            return value_loss.item(), perturb_out
        value_loss.backward()
        unclipped_norm = clip_grad_norm_(self.critic_params, 0.5)

        self.critic_optim.step()
        if self.target_update_mode == 'soft':
            soft_update(self.critic_target, self.critic, self.tau)
        elif self.target_update_mode == 'hard':
            hard_update(self.critic_target, self.critic)
        return value_loss.item(), perturb_out, unclipped_norm

    def update_actor_parameters(self, batch, agent_id, shuffle=None):
        state_batch = Variable(torch.cat(batch.state)).to(self.device)
        if shuffle == 'shuffle':
            rand_idx = np.random.permutation(self.n_agent)
            new_state_batch = state_batch.view(-1, self.n_agent, self.obs_dim)
            state_batch = new_state_batch[:, rand_idx, :].view(-1, self.obs_dim * self.n_agent)

        for i in range(self.n_group):
            self.actor_optims[i].zero_grad()
        action_batch_n, logit = self.select_action(
            state_batch.view(-1, self.obs_dim), action_noise=self.train_noise, grad=True)
        action_batch_n = action_batch_n.view(-1, self.n_action * self.n_agent)

        policy_loss = -self.critic(state_batch, action_batch_n)
        policy_loss = policy_loss.mean() + 1e-3 * (logit ** 2).mean()
        policy_loss.backward()
        for i in range(self.n_group):
            clip_grad_norm_(self.actors_params[i], 0.5)
            self.actor_optims[i].step()
            soft_update(self.actor_targets[i], self.actors[i], self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_loss.item()


    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

    @property
    def actor_lr(self):
        return self.actor_optims[0].param_groups[0]['lr']
