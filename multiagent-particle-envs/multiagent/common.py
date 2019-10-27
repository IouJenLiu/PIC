from gym import spaces
from multiagent.core import Action
from multiagent.multi_discrete import MultiDiscrete
import numpy as np
import torch


def action_callback(agent, self):
    """
    scripted agent take an action
    :param agent: the agent
    :param self: wolrd
    :return: Action
    """
    obs = self.obs_callback(agent, self)
    action = self.s_agents.select_action(torch.Tensor([obs]), action_noise=True, param_noise=None).squeeze().numpy()
    return _get_action(action, agent, self)


def _get_action(action, agent, self):
    """

    :param action: the action from the model

    :return: the Action
    """
    discrete_action_input = False
    force_discrete_action = self.discrete_action if hasattr(self, 'discrete_action') else False
    discrete_action_space = True
    world_dim_p = 2
    action_space = spaces.Discrete(world_dim_p * 2 + 1)
    agent_action = Action()
    # set action
    agent_action.u = np.zeros(self.dim_p)
    agent_action.c = np.zeros(self.dim_c)
    # process action
    if isinstance(action_space, MultiDiscrete):
        act = []
        size = action_space.high - action_space.low + 1
        index = 0
        for s in size:
            act.append(action[index:(index + s)])
            index += s
        action = act
    else:
        action = [action]

    if agent.movable:
        # physical action
        if discrete_action_input:
            agent_action.u = np.zeros(self.dim_p)
            # process discrete action
            if action[0] == 1: agent_action.u[0] = -1.0
            if action[0] == 2: agent_action.u[0] = +1.0
            if action[0] == 3: agent_action.u[1] = -1.0
            if action[0] == 4: agent_action.u[1] = +1.0
        else:
            if force_discrete_action:
                d = np.argmax(action[0])
                action[0][:] = 0.0
                action[0][d] = 1.0
            if discrete_action_space:
                agent_action.u[0] += action[0][1] - action[0][2]
                agent_action.u[1] += action[0][3] - action[0][4]
            else:
                agent_action.u = action[0]
        sensitivity = 5.0
        if agent.accel is not None:
            sensitivity = agent.accel
        agent_action.u *= sensitivity
        action = action[1:]
    if not agent.silent:
        # communication action
        if discrete_action_input:
            agent_action.c = np.zeros(self.dim_c)
            agent_action.c[action[0]] = 1.0
        else:
            agent_action.c = action[0]
        action = action[1:]
    # make sure we used all elements of action
    assert len(action) == 0

    return agent_action