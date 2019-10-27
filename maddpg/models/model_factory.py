"""Implements a model factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from models import graph_net, mlp_net

MODEL_MAP = {
    'mlp': mlp_net.MlpNet,
    'mlp_module': mlp_net.MlpNetM,
    'mlp_shuffle_all': functools.partial(mlp_net.MlpNet, agent_shuffle='all'),
    'mlp_shuffle_others': functools.partial(mlp_net.MlpNet, agent_shuffle='others'),
    'gcn_mean': functools.partial(graph_net.GraphNet, pool_type='avg'),
    'gcn_max': functools.partial(graph_net.GraphNet, pool_type='max'),
    'gcn_max_hetero': functools.partial(graph_net.GraphNetHetro, pool_type='max'),
    'gcn_max_nn': functools.partial(graph_net.GraphNetNN, pool_type='max'),
    'gcn_max_v': functools.partial(graph_net.GraphNetV, pool_type='max'),
    'gcn_mean_id': functools.partial(graph_net.GraphNet, pool_type='avg', use_agent_id=True),
    'gcn_max_id': functools.partial(graph_net.GraphNet, pool_type='max', use_agent_id=True),
    'msg_gnn': graph_net.MsgGraphNet,
    'msg_gnn_hard': graph_net.MsgGraphNetHard
}


def get_model_fn(name):
  assert name in MODEL_MAP
  return MODEL_MAP[name]
