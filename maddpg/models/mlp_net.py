"""Implements a simple two layer mlp network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MlpNet(nn.Module):
  """Implements a simple fully connected mlp network."""

  def __init__(self, sa_dim, n_agents, hidden_size,
               agent_id=0, agent_shuffle='none'):
    super(MlpNet, self).__init__()
    self.linear1 = nn.Linear(sa_dim * n_agents, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.V = nn.Linear(hidden_size, 1)
    self.V.weight.data.mul_(0.1)
    self.V.bias.data.mul_(0.1)

    self.n_agents = n_agents
    self.agent_id = agent_id
    self.agent_shuffle = agent_shuffle

  def forward(self, x):
    # Perform shuffling.
    bz = x.shape[0]
    if self.agent_shuffle == 'all':
      x_out = []
      for k in range(bz):
        rand_idx = np.random.permutation(self.n_agents)
        x_out.append(x[k, :, rand_idx].unsqueeze(0))
      x = torch.cat(x_out, 0)
    elif self.agent_shuffle == 'others':
      x_out = []
      for k in range(bz):
        rand_idx = np.random.permutation(self.n_agents-1)
        index_except = np.concatenate([np.arange(0, self.agent_id),
                                       np.arange(self.agent_id+1, self.n_agents) ])
        except_shuffle = index_except[rand_idx]
        x_tmp = x[k, :, :]
        x_tmp[:, index_except] = x_tmp[:, except_shuffle]
        x_out.append(x_tmp.unsqueeze(0))
      x = torch.cat(x_out, 0)
    elif self.agent_shuffle == 'none':
      pass
    else:
      raise NotImplemented(
          'Unsupported agent_shuffle opt: %s' % self.agent_shuffle)

    # Reshape to fit into mlp.
    x = x.view(bz, -1)

    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    V = self.V(x)
    return V


class MlpNetM(nn.Module):
  """Implements a simple fully connected mlp network."""

  def __init__(self, sa_dim, n_agents, hidden_size,
               agent_id=0, agent_shuffle='none'):
    super(MlpNetM, self).__init__()
    self.linear1 = nn.Linear(sa_dim, hidden_size)
    self.linear2 = nn.Linear(hidden_size * 3, hidden_size)
    self.linear3 = nn.Linear(hidden_size, hidden_size)
    self.V = nn.Linear(hidden_size, 1)
    self.V.weight.data.mul_(0.1)
    self.V.bias.data.mul_(0.1)

    self.n_agents = n_agents
    self.agent_id = agent_id
    self.agent_shuffle = agent_shuffle

  def forward(self, x):
    # Perform shuffling.
    bz = x.shape[0]
    if self.agent_shuffle == 'all':
      x_out = []
      for k in range(bz):
        rand_idx = np.random.permutation(self.n_agents)
        x_out.append(x[k, :, rand_idx].unsqueeze(0))
      x = torch.cat(x_out, 0)
    elif self.agent_shuffle == 'others':
      x_out = []
      for k in range(bz):
        rand_idx = np.random.permutation(self.n_agents-1)
        index_except = np.concatenate([np.arange(0, self.agent_id),
                                       np.arange(self.agent_id+1, self.n_agents) ])
        except_shuffle = index_except[rand_idx]
        x_tmp = x[k, :, :]
        x_tmp[:, index_except] = x_tmp[:, except_shuffle]
        x_out.append(x_tmp.unsqueeze(0))
      x = torch.cat(x_out, 0)
    elif self.agent_shuffle == 'none':
      pass
    else:
      raise NotImplemented(
          'Unsupported agent_shuffle opt: %s' % self.agent_shuffle)

    # Reshape to fit into mlp.
    x1, x2, x3 = self.linear1(x[:, :, 0]), self.linear1(x[:, :, 1]), self.linear1(x[:, :, 2])
    x = torch.cat((x1, x2, x3), 1)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    x = self.linear3(x)
    x = F.relu(x)
    V = self.V(x)
    return V
