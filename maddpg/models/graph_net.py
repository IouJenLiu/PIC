import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_layers import GraphConvLayer, MessageFunc, UpdateFunc


class GraphNetHetro(nn.Module):

  # A graph net that supports different edge attributes.

    def __init__(self, sa_dim, n_agents, hidden_size, agent_groups, agent_id=0,
                 pool_type='avg', use_agent_id=False):
        """
        :param sa_dim: integer
        :param n_agents: integer
        :param hidden_size: integer
        :param agent_groups: list, represents number of agents in each group, agents in the same
        group are homogeneous. Agents in different groups are heterogeneous
        ex. agent_groups = [4] --> Group three has has agent 0, agent 1, agent 2, agent 3
            agent_groups =[2, 3] --> Group one has agent 0, agent 1.
                                     Group two has agent 2, agent 3, agent 4
            agent_groups =[2,3,4] --> Group one has agent 0, agent 1.
                                      Group two has agent 2, agent 3, agent 4.
                                      Group three has has agent 5, agent 6, agent 7
        """
        super(GraphNetHetro, self).__init__()
        assert n_agents == sum(agent_groups)

        self.sa_dim = sa_dim
        self.n_agents = n_agents
        self.pool_type = pool_type
        self.agent_groups = agent_groups

        group_emb_dim = 2  # Dimension for the group embedding.

        if use_agent_id:
            agent_id_attr_dim = 2
            self.gc1 = GraphConvLayer(sa_dim + agent_id_attr_dim, hidden_size)
            self.nn_gc1 = nn.Linear(sa_dim + agent_id_attr_dim, hidden_size)
        else:
            self.gc1 = GraphConvLayer(sa_dim + group_emb_dim, hidden_size)
            self.nn_gc1 = nn.Linear(sa_dim + group_emb_dim, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.nn_gc2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        # Create group embeddings.
        num_groups = len(agent_groups)

        self.group_emb = nn.ParameterList([nn.Parameter(torch.randn(1, 1, group_emb_dim), requires_grad=True) for k in range(num_groups)])

        # Assumes a fully connected graph.
        self.register_buffer('adj', (torch.ones(n_agents, n_agents) - torch.eye(n_agents)) / self.n_agents)

        self.use_agent_id = use_agent_id

        self.agent_id = agent_id

        if use_agent_id:
            self.curr_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim, 1), requires_grad=True)
            self.other_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim, 1), requires_grad=True)

            agent_att = []
            for k in range(self.n_agents):
                if k == self.agent_id:
                    agent_att.append(self.curr_agent_attr.unsqueeze(-1))
                else:
                    agent_att.append(self.other_agent_attr.unsqueeze(-1))
            agent_att = torch.cat(agent_att, -1)
            self.agent_att = agent_att.unsqueeze(0)

    def forward(self, x):
        """
        :param x: [batch_size, self.sa_dim, self.n_agent] tensor
        :return: [batch_size, self.output_dim] tensor
        """
        if self.use_agent_id:
            agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
            x = torch.cat([x, agent_att], 1)

        # Concat group embeddings, concat to input layer.
        group_emb_list = []
        for k_idx, k in enumerate(self.agent_groups):
          group_emb_list += [self.group_emb[k_idx]]*k
        group_emb = torch.cat(group_emb_list, 1)
        group_emb_batch = torch.cat([group_emb]*x.shape[0], 0)

        x = torch.cat([x, group_emb_batch], -1)

        feat = F.relu(self.gc1(x, self.adj))
        feat += F.relu(self.nn_gc1(x))
        feat /= (1. * self.n_agents)
        out = F.relu(self.gc2(feat, self.adj))
        out += F.relu(self.nn_gc2(feat))
        out /= (1. * self.n_agents)

        # Pooling
        if self.pool_type == 'avg':
            ret = out.mean(1)  # Pooling over the agent dimension.
        elif self.pool_type == 'max':
            ret, _ = out.max(1)

        # Compute V
        V = self.V(ret)
        return V


class GraphNetV(nn.Module):

    # A graph net that supports different edge attributes and outputs an vector

    def __init__(self, sa_dim, n_agents, hidden_size, agent_groups, agent_id=0,
                 pool_type='avg', use_agent_id=False):
        """
        :param sa_dim: integer
        :param n_agents: integer
        :param hidden_size: integer
        :param agent_groups: list, represents number of agents in each group, agents in the same
        group are homogeneous. Agents in different groups are heterogeneous
        ex. agent_groups = [4] --> Group three has has agent 0, agent 1, agent 2, agent 3
            agent_groups =[2, 3] --> Group one has agent 0, agent 1.
                                     Group two has agent 2, agent 3, agent 4
            agent_groups =[2,3,4] --> Group one has agent 0, agent 1.
                                      Group two has agent 2, agent 3, agent 4.
                                      Group three has has agent 5, agent 6, agent 7
        """
        super(GraphNetV, self).__init__()
        assert n_agents == sum(agent_groups)

        self.sa_dim = sa_dim
        self.n_agents = n_agents
        self.pool_type = pool_type
        self.agent_groups = agent_groups

        group_emb_dim = 2  # Dimension for the group embedding.

        if use_agent_id:
            agent_id_attr_dim = 2
            self.gc1 = GraphConvLayer(sa_dim + agent_id_attr_dim, hidden_size)
            self.nn_gc1 = nn.Linear(sa_dim + agent_id_attr_dim, hidden_size)
        else:
            self.gc1 = GraphConvLayer(sa_dim + group_emb_dim, hidden_size)
            self.nn_gc1 = nn.Linear(sa_dim + group_emb_dim, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.nn_gc2 = nn.Linear(hidden_size, hidden_size)

        # Create group embeddings.
        num_groups = len(agent_groups)

        self.group_emb = nn.ParameterList([nn.Parameter(torch.randn(1, 1, group_emb_dim), requires_grad=True) for k in range(num_groups)])

        # Assumes a fully connected graph.
        self.register_buffer('adj', (torch.ones(n_agents, n_agents) - torch.eye(n_agents)) / self.n_agents)

        self.use_agent_id = use_agent_id

        self.agent_id = agent_id

        if use_agent_id:
            self.curr_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim, 1), requires_grad=True)
            self.other_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim, 1), requires_grad=True)

            agent_att = []
            for k in range(self.n_agents):
                if k == self.agent_id:
                    agent_att.append(self.curr_agent_attr.unsqueeze(-1))
                else:
                    agent_att.append(self.other_agent_attr.unsqueeze(-1))
            agent_att = torch.cat(agent_att, -1)
            self.agent_att = agent_att.unsqueeze(0)

    def forward(self, x):
        """
        :param x: [batch_size, self.sa_dim, self.n_agent] tensor
        :return: [batch_size, self.output_dim] tensor
        """
        if self.use_agent_id:
            agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
            x = torch.cat([x, agent_att], 1)

        # Concat group embeddings, concat to input layer.
        group_emb_list = []
        for k_idx, k in enumerate(self.agent_groups):
          group_emb_list += [self.group_emb[k_idx]]*k
        group_emb = torch.cat(group_emb_list, 1)
        group_emb_batch = torch.cat([group_emb]*x.shape[0], 0)

        x = torch.cat([x, group_emb_batch], -1)

        feat = F.relu(self.gc1(x, self.adj))
        feat += F.relu(self.nn_gc1(x))
        feat /= (1. * self.n_agents)
        out = F.relu(self.gc2(feat, self.adj))
        out += F.relu(self.nn_gc2(feat))
        out /= (1. * self.n_agents)

        # Pooling
        if self.pool_type == 'avg':
            ret = out.mean(1)  # Pooling over the agent dimension.
        elif self.pool_type == 'max':
            ret, _ = out.max(1)
        return ret

class GraphNet(nn.Module):
    """
    A graph net that is used to pre-process actions and states, and solve the order issue.
    """

    def __init__(self, sa_dim, n_agents, hidden_size, agent_id=0,
                 pool_type='avg', use_agent_id=False):
        super(GraphNet, self).__init__()
        self.sa_dim = sa_dim
        self.n_agents = n_agents
        self.pool_type = pool_type
        if use_agent_id:
            agent_id_attr_dim = 2
            self.gc1 = GraphConvLayer(sa_dim + agent_id_attr_dim, hidden_size)
            self.nn_gc1 = nn.Linear(sa_dim + agent_id_attr_dim, hidden_size)
        else:
            self.gc1 = GraphConvLayer(sa_dim, hidden_size)
            self.nn_gc1 = nn.Linear(sa_dim, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.nn_gc2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        # Assumes a fully connected graph.
        self.register_buffer('adj', (torch.ones(n_agents, n_agents) - torch.eye(n_agents)) / self.n_agents)

        self.use_agent_id = use_agent_id

        self.agent_id = agent_id

        if use_agent_id:
            self.curr_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim), requires_grad=True)
            self.other_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim), requires_grad=True)

            agent_att = []
            for k in range(self.n_agents):
                if k == self.agent_id:
                    agent_att.append(self.curr_agent_attr.unsqueeze(-1))
                else:
                    agent_att.append(self.other_agent_attr.unsqueeze(-1))
            agent_att = torch.cat(agent_att, -1)
            self.agent_att = agent_att.unsqueeze(0)

    def forward(self, x):
        """
        :param x: [batch_size, self.sa_dim, self.n_agent] tensor
        :return: [batch_size, self.output_dim] tensor
        """
        if self.use_agent_id:
            agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
            x = torch.cat([x, agent_att], 1)

        feat = F.relu(self.gc1(x, self.adj))
        feat += F.relu(self.nn_gc1(x))
        feat /= (1. * self.n_agents)
        out = F.relu(self.gc2(feat, self.adj))
        out += F.relu(self.nn_gc2(feat))
        out /= (1. * self.n_agents)

        # Pooling
        if self.pool_type == 'avg':
            ret = out.mean(1)  # Pooling over the agent dimension.
        elif self.pool_type == 'max':
            ret, _ = out.max(1)

        # Compute V
        V = self.V(ret)
        return V


class MsgGraphNet(nn.Module):
    """
    A message-passing GNN
    """

    def __init__(self, sa_dim, n_agents, hidden_size):
        super(MsgGraphNet, self).__init__()
        self.sa_dim = sa_dim
        self.n_agents = n_agents

        self.msg1 = MessageFunc(sa_dim * 2, hidden_size)
        self.msg2 = MessageFunc(hidden_size * 2, hidden_size)
        self.update1 = UpdateFunc(sa_dim, n_agents, hidden_size)
        self.update2 = UpdateFunc(sa_dim, n_agents, hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.non_linear = F.relu  # tanh, sigmoid
        self.adj = torch.ones(n_agents, n_agents) - \
                   torch.eye(n_agents)
        self.register_buffer('extended_adj', self.extend_adj())

    def extend_adj(self):
        ret = torch.zeros(self.n_agents, self.n_agents * self.n_agents)
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if self.adj[i, j] == 1:
                    ret[i, j * self.n_agents + i] = 1
        return ret

    def forward(self, x):
        """
        :param x: [batch_size, self.n_agent, self.sa_dim, ] tensor
        :return: [batch_size, self.output_dim] tensor
        """

        e1 = self.non_linear(self.msg1(x))
        v1 = self.non_linear(self.update1(e1, x, self.extended_adj))

        e2 = self.non_linear(self.msg2(v1))
        v2 = self.non_linear(self.update2(e2, x, self.extended_adj))
        out = torch.max(v2, dim=1)[0]

        # Compute V
        return self.V(out)


class MsgGraphNetHard(nn.Module):
    """
    A message-passing GNN with 3-clique graph, will extend to general case.
    """

    def __init__(self, sa_dim, n_agents, hidden_size):
        super(MsgGraphNetHard, self).__init__()
        self.sa_dim = sa_dim
        self.n_agents = n_agents

        self.fe1 = nn.Linear(sa_dim * 2, hidden_size)
        self.fe2 = nn.Linear(hidden_size * 2, hidden_size)

        self.fv1 = nn.Linear(hidden_size + sa_dim, hidden_size)
        self.fv2 = nn.Linear(hidden_size + sa_dim, hidden_size)

        self.msg1 = MessageFunc(sa_dim * 2, hidden_size)
        self.msg2 = MessageFunc(hidden_size * 2, hidden_size)
        self.update1 = UpdateFunc(sa_dim, n_agents, hidden_size)
        self.update2 = UpdateFunc(sa_dim, n_agents, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.non_linear = F.relu  # tanh, sigmoid
        self.adj = torch.ones(n_agents, n_agents) - \
                   torch.eye(n_agents)
        self.extended_adj = self.extend_adj()

    def extend_adj(self):
        ret = torch.zeros(self.n_agents, self.n_agents * self.n_agents)
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if self.adj[i, j] == 1:
                    ret[i, j * self.n_agents + i] = 1
        return ret

    def forward(self, x):
        """
          :param x: [batch_size, self.n_agent, self.sa_dim, ] tensor
          :return: [batch_size, self.output_dim] tensor
        """
        x = torch.transpose(x, 1, 2)
        h1_01 = self.non_linear(self.fe1(torch.cat((x[:, :, 0], x[:, :, 1]), dim=1)))
        h1_02 = self.non_linear(self.fe1(torch.cat((x[:, :, 0], x[:, :, 2]), dim=1)))

        h1_10 = self.non_linear(self.fe1(torch.cat((x[:, :, 1], x[:, :, 0]), dim=1)))
        h1_12 = self.non_linear(self.fe1(torch.cat((x[:, :, 1], x[:, :, 2]), dim=1)))

        h1_20 = self.non_linear(self.fe1(torch.cat((x[:, :, 2], x[:, :, 0]), dim=1)))
        h1_21 = self.non_linear(self.fe1(torch.cat((x[:, :, 2], x[:, :, 1]), dim=1)))

        h2_0 = self.non_linear(self.fv1(torch.cat(((h1_10 + h1_20) / 2, x[:, :, 0]), dim=1)))
        h2_1 = self.non_linear(self.fv1(torch.cat(((h1_01 + h1_21) / 2, x[:, :, 1]), dim=1)))
        h2_2 = self.non_linear(self.fv1(torch.cat(((h1_12 + h1_02) / 2, x[:, :, 2]), dim=1)))

        h2_01 = self.non_linear(self.fe2(torch.cat((h2_0, h2_1), dim=1)))
        h2_02 = self.non_linear(self.fe2(torch.cat((h2_0, h2_2), dim=1)))

        h2_10 = self.non_linear(self.fe2(torch.cat((h2_1, h2_0), dim=1)))
        h2_12 = self.non_linear(self.fe2(torch.cat((h2_1, h2_2), dim=1)))

        h2_20 = self.non_linear(self.fe2(torch.cat((h2_2, h2_0), dim=1)))
        h2_21 = self.non_linear(self.fe2(torch.cat((h2_2, h2_1), dim=1)))

        h3_0 = self.non_linear(self.fv2(torch.cat(((h2_10 + h2_20) / 2, x[:, :, 0]), dim=1)))
        h3_1 = self.non_linear(self.fv2(torch.cat(((h2_01 + h2_21) / 2, x[:, :, 1]), dim=1)))
        h3_2 = self.non_linear(self.fv2(torch.cat(((h2_02 + h2_12) / 2, x[:, :, 2]), dim=1)))

        out = torch.max(torch.max(h3_0, h3_1), h3_2)
        # Compute V
        return self.V(out)


class GraphNetNN(nn.Module):
    """
    A graph net that is used to pre-process actions and states, and solve the order issue.
    """

    def __init__(self, sa_dim, n_agents, hidden_size, agent_id=0,
                 pool_type='avg', use_agent_id=False):
        super(GraphNetNN, self).__init__()
        self.sa_dim = sa_dim
        self.n_agents = n_agents
        self.pool_type = pool_type
        if use_agent_id:
            agent_id_attr_dim = 2
            self.gc1 = GraphConvLayer(sa_dim + agent_id_attr_dim, hidden_size)
            self.nn_gc1 = nn.Linear(sa_dim + agent_id_attr_dim, hidden_size)
        else:
            self.gc1 = GraphConvLayer(sa_dim, hidden_size)
            self.nn_gc1 = nn.Linear(sa_dim, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.nn_gc2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        # Assumes a fully connected graph.
        self.use_agent_id = use_agent_id

        self.agent_id = agent_id

        if use_agent_id:
            self.curr_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim), requires_grad=True)
            self.other_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim), requires_grad=True)

            agent_att = []
            for k in range(self.n_agents):
                if k == self.agent_id:
                    agent_att.append(self.curr_agent_attr.unsqueeze(-1))
                else:
                    agent_att.append(self.other_agent_attr.unsqueeze(-1))
            agent_att = torch.cat(agent_att, -1)
            self.agent_att = agent_att.unsqueeze(0)

    def forward(self, x, adj):
        """
        :param x: [batch_size, self.sa_dim, self.n_agent] tensor
        :return: [batch_size, self.output_dim] tensor
        """
        if self.use_agent_id:
            agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
            x = torch.cat([x, agent_att], 1)

        feat = F.relu(self.gc1(x, adj))
        feat += F.relu(self.nn_gc1(x))
        feat /= (1. * self.n_agents)
        out = F.relu(self.gc2(feat, adj))
        out += F.relu(self.nn_gc2(feat))
        out /= (1. * self.n_agents)

        # Pooling
        if self.pool_type == 'avg':
            ret = out.mean(1)  # Pooling over the agent dimension.
        elif self.pool_type == 'max':
            ret, _ = out.max(1)

        # Compute V
        V = self.V(ret)
        return V
