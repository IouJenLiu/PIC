"""Implements some basic test."""

import unittest
import torch
from models.graph_net import GraphNetHetro
from models.mlp_net import MlpNet
from models.graph_layers import GraphConvLayer
from models.model_factory import get_model_fn


class TestNet(unittest.TestCase):
  """Basic tests for graph net."""

  def setUp(self,):
    batch_size = 2
    self.batch_size = batch_size
    sa_dim = 5
    self.sa_dim = sa_dim
    n_agent = 3
    self.n_agent = n_agent
    hidden_dim = 5
    self.hidden_dim = hidden_dim
    self.x = torch.randn(batch_size, sa_dim, n_agent)
    self.x = torch.transpose(self.x, 1,2)
    self.graph_model = GraphNetHetro(sa_dim, n_agent, hidden_dim,
                                     agent_groups=[2, 1])

  def test_io_dim_attr(self):
    """Tests the input/output dimension of the network."""
    y = self.graph_model(self.x)
    self.assertEqual(y.shape[0], self.batch_size)
    self.assertEqual(y.shape[1], 1)


if __name__ == '__main__':
  unittest.main()
