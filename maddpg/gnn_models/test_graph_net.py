"""Implements some basic test."""

import unittest
import torch
from graph_net import GraphNet
from gnn_models.graph_layers import GraphConvLayer


class TestGraphNet(unittest.TestCase):
  """Basic tests for graph net."""

  def setUp(self,):
    batch_size = 2
    self.batch_size = batch_size
    sa_dim = 5
    self.sa_dim = sa_dim
    n_agent = 3
    self.n_agent = n_agent
    output_dim = 4
    self.output_dim = output_dim

    self.x = torch.randn(batch_size, sa_dim, n_agent)
    self.graph_model = GraphNet(sa_dim, n_agent, output_dim)

  def test_io_dim(self):
    """Tests the input/output dimension of the network."""
    y = self.graph_model(self.x)
    self.assertEqual(y.shape[0], self.batch_size)
    self.assertEqual(y.shape[1], self.output_dim)

  def test_permute_invar(self):
    """Tests the overall network is permutation invariant."""
    y = self.graph_model(self.x)
    x_permute = self.x[:, :, [2, 1, 0]]
    y_permute = self.graph_model(x_permute)
    assert((y-y_permute).mean() < 1e-8)

  def test_layer_equivar(self):
    """Tests each GCN layer is permutation equivaraint."""
    adj = torch.eye(self.n_agent)
    gc1 = GraphConvLayer(self.sa_dim, 7)
    x = torch.transpose(self.x, 1, 2)
    y = gc1(x, adj)
    x_permute = x[:, [2, 1, 0], :]
    y_permute = gc1(x_permute, adj)
    y_permute_back = y_permute[:, [2, 1, 0], :]
    assert((y-y_permute_back).mean() < 1e-8)


if __name__ == '__main__':
  unittest.main()
