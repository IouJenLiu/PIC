"""Implements graph layers."""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvLayer(Module):
    """Implements a GCN layer."""

    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.lin_layer = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, input_feature, input_adj):
        feat = self.lin_layer(input_feature)
        out = torch.matmul(input_adj, feat)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class MessageFunc(Module):
    """Implements a Message function"""

    def __init__(self, input_dim, hidden_size):
        super(MessageFunc, self).__init__()
        self.fe = nn.Linear(input_dim, hidden_size)
        self.input_dim = input_dim
        self.hidden_size = hidden_size

    def forward(self, input_feature):
        """
        :param x: [batch_size, n_agent, self.sa_dim] tensor
        :return msg: [batch_size, n_agent * n_agent, output_dim] tensor
        """
        n_agent = input_feature.size()[1]
        bz = input_feature.size()[0]
        x1 = input_feature.unsqueeze(2).repeat(1, 1, n_agent, 1)
        x1 = x1.view(bz, n_agent * n_agent, -1)
        x2 = input_feature.repeat(1, n_agent, 1)
        x = torch.cat((x1, x2), dim=2)
        msg = self.fe(x)
        return msg

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.hidden_size) + ')'


class UpdateFunc(Module):
    """Implements a Message function"""

    def __init__(self, sa_dim, n_agents, hidden_size):
        super(UpdateFunc, self).__init__()
        self.fv = nn.Linear(hidden_size + sa_dim, hidden_size)
        self.input_dim = hidden_size + sa_dim
        self.output_dim = hidden_size
        self.n_agents = n_agents

    def forward(self, input_feature, x, extended_adj):
        """
          :param input_feature: [batch_size, n_agent ** 2, self.sa_dim] tensor
          :param x: [batch_size, n_agent, self.sa_dim] tensor
          :param extended_adj: [n_agent, n_agent ** 2] tensor
          :return v: [batch_size, n_agent, hidden_size] tensor
        """

        agg = torch.matmul(extended_adj, input_feature)
        x = torch.cat((agg, x), dim=2)
        v = self.fv(x)
        return v

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'
