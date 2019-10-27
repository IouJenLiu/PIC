import numpy as np
from multiagent.core_vec import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import torch
import os

from multiagent.common import action_callback

class Scenario(BaseScenario):
    def __init__(self):
        obs_path = os.path.dirname(os.path.abspath(__file__))
        obs_path = os.path.dirname(os.path.dirname(obs_path))
        scripted_agent_ckpt = os.path.join(obs_path, 'scripted_agent_ckpt/simple_tag_n6_train_prey/agents_best.ckpt')
        self.scripted_agents = torch.load(scripted_agent_ckpt)['agents']

    def make_world(self):
        world = World(self.scripted_agents, self.observation)
        self.np_rnd = np.random.RandomState(0)
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 30
        num_adversaries = 100
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 20
        self.world_radius = 1
        self.num_agents = num_good_agents + num_adversaries
        self.num_landmarks = num_landmarks
        self.num_adversaries = num_adversaries
        self.num_good_agents = num_good_agents
        world.collaborative = True
        # add agents
        world.agents = [Agent() for _ in range(num_adversaries)] \
                       + [Agent(action_callback) for _ in range(num_good_agents)]
        #world.agents = [Agent(), Agent(), Agent(), Agent(action_callback)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.01 if agent.adversary else 0.005
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.id = i
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.05
            landmark.boundary = False
        # make initial conditions
        self.collide_th = self.good_agents(world)[0].size + self.adversaries(world)[0].size
        self.n_visible_agent = 1
        self.n_visible_landmark = 3
        self.n_visible_adv = 6
        self.n_adv_visible_agent = 5
        self.n_adv_visible_landmark = 5
        self.n_adv_visible_adv = 5

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.np_rnd.uniform(-self.world_radius, +self.world_radius, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = self.np_rnd.uniform(-(self.world_radius - 0.1), self.world_radius  - 0.1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew, rew1 = 0, 0
        n_col, n_collide = 0, 0
        if agent == world.agents[0]:
            agents = self.good_agents(world)
            adversaries = self.adversaries(world)

            adv_pos = np.array([[adv.state.p_pos for adv in adversaries]]).repeat(len(agents), axis=0)
            a_pos = np.array([[a.state.p_pos for a in agents]])
            a_pos1 = a_pos.repeat(len(adversaries), axis=0)
            a_pos1 = np.transpose(a_pos1, axes=(1, 0, 2))
            dist = np.sqrt(np.sum(np.square(adv_pos - a_pos1), axis=2))
            rew = np.min(dist, axis=0)
            rew = -0.1 * np.sum(rew)
            if agent.collide:
                n_collide = (dist < self.collide_th).sum()
            rew += 10 * n_collide

        return rew

    def observation(self, agent, world):
        if not agent.adversary:
            if agent.id == self.num_adversaries:
                l_pos = np.array([[l.state.p_pos for l in world.landmarks]]).repeat(self.num_good_agents, axis=0)
                a_pos = np.array([[a.state.p_pos for a in world.agents[self.num_adversaries:]]])
                a_pos1 = a_pos.repeat(self.num_adversaries, axis=0)
                a_pos1 = np.transpose(a_pos1, axes=(1, 0, 2))
                a_pos2 = a_pos.repeat(self.num_good_agents, axis=0)
                a_pos3 = a_pos.repeat(len(world.landmarks), axis=0)
                a_pos3 = np.transpose(a_pos3, axes=(1, 0, 2))
                a_pos4 = a_pos.repeat(self.num_good_agents, axis=0)
                a_pos4 = np.transpose(a_pos4, axes=(1, 0, 2))
                adv_pos = np.array([[a.state.p_pos for a in world.agents[:self.num_adversaries]]])
                adv_pos = adv_pos.repeat(self.num_good_agents, axis=0)
                entity_pos = l_pos - a_pos3
                other_adv_pos = adv_pos - a_pos1
                other_agent_pos = a_pos2 - a_pos4
                other_agent_vel = np.array([[a.state.p_vel for a in world.agents[self.num_adversaries:]]]).repeat(self.num_good_agents, axis=0)

                entity_dist = np.sqrt(np.sum(np.square(entity_pos), axis=2))
                entity_dist_idx = np.argsort(entity_dist, axis=1)
                row_idx = np.arange(self.num_good_agents).repeat(self.num_landmarks)
                self.sorted_entity_pos = entity_pos[row_idx, entity_dist_idx.reshape(-1)].reshape(self.num_good_agents,
                                                                                                  self.num_landmarks, 2)

                other_dist = np.sqrt(np.sum(np.square(other_adv_pos), axis=2))
                other_dist_idx = np.argsort(other_dist, axis=1)
                row_idx = np.arange(self.num_good_agents).repeat(self.num_adversaries)
                self.sorted_other_adv_pos = other_adv_pos[row_idx, other_dist_idx.reshape(-1)].reshape(self.num_good_agents,
                                                                                               self.num_adversaries, 2)
                other_dist = np.sqrt(np.sum(np.square(other_agent_pos), axis=2))
                other_dist_idx = np.argsort(other_dist, axis=1)
                row_idx = np.arange(self.num_good_agents).repeat(self.num_good_agents)
                self.sorted_other_agent_pos = other_agent_pos[row_idx, other_dist_idx.reshape(-1)].reshape(self.num_good_agents,
                                                                                               self.num_good_agents, 2)[:,
                                        1:, :]

                row_idx = np.arange(self.num_good_agents).repeat(self.num_good_agents)
                self.sorted_other_agent_vel = other_agent_vel[row_idx, other_dist_idx.reshape(-1)].reshape(self.num_good_agents,
                                                                                               self.num_good_agents, 2)[:,
                                        1:, :]

                self.sorted_other_agent_vel = self.sorted_other_agent_vel[:, :self.n_visible_agent, :]
                self.sorted_other_agent_pos = self.sorted_other_agent_pos[:, :self.n_visible_agent, :]
                self.sorted_other_adv_pos = self.sorted_other_adv_pos[:, :self.n_visible_adv, :]
                self.sorted_entity_pos = self.sorted_entity_pos[:, :self.n_visible_landmark, :]

            id_ = agent.id - self.num_adversaries
            obs = np.concatenate((np.array([agent.state.p_vel]), np.array([agent.state.p_pos]),
                                  self.sorted_entity_pos[id_, :, :],
                                  self.sorted_other_adv_pos[id_, :, :], self.sorted_other_agent_pos[id_, :, :], self.sorted_other_agent_vel[id_, :, :]), axis=0).reshape(-1)
            return obs
        else:
            # get positions of all entities in this agent's reference frame
            if agent.id == 0:
                l_pos = np.array([[l.state.p_pos for l in world.landmarks]]).repeat(self.num_adversaries, axis=0)
                adv_pos = np.array([[a.state.p_pos for a in world.agents[:self.num_adversaries]]])
                adv_pos1 = adv_pos.repeat(self.num_adversaries, axis=0)
                adv_pos1 = np.transpose(adv_pos1, axes=(1, 0, 2))
                adv_pos2 = adv_pos.repeat(self.num_adversaries, axis=0)
                adv_pos3 = adv_pos.repeat(len(world.landmarks), axis=0)
                adv_pos3 = np.transpose(adv_pos3, axes=(1, 0, 2))
                adv_pos4 = adv_pos.repeat(self.num_good_agents, axis=0)
                adv_pos4 = np.transpose(adv_pos4, axes=(1, 0, 2))
                a_pos = np.array([[a.state.p_pos for a in world.agents[self.num_adversaries:]]])
                a_pos2 = a_pos.repeat(self.num_adversaries, axis=0)
                entity_pos = l_pos - adv_pos3
                other_adv_pos = adv_pos2 - adv_pos1
                other_agent_pos = a_pos2 - adv_pos4
                other_agent_vel = np.array([[a.state.p_vel for a in world.agents[self.num_adversaries:]]]).repeat(self.num_adversaries, axis=0)

                entity_dist = np.sqrt(np.sum(np.square(entity_pos), axis=2))
                entity_dist_idx = np.argsort(entity_dist, axis=1)
                row_idx = np.arange(self.num_adversaries).repeat(self.num_landmarks)
                self.sorted_entity_pos = entity_pos[row_idx, entity_dist_idx.reshape(-1)].reshape(self.num_adversaries,
                                                                                                  self.num_landmarks, 2)

                other_dist = np.sqrt(np.sum(np.square(other_adv_pos), axis=2))
                other_dist_idx = np.argsort(other_dist, axis=1)
                row_idx = np.arange(self.num_adversaries).repeat(self.num_adversaries)
                self.sorted_other_adv_pos = other_adv_pos[row_idx, other_dist_idx.reshape(-1)].reshape(self.num_adversaries,
                                                                                               self.num_adversaries, 2)[:,
                                        1:, :]
                other_dist = np.sqrt(np.sum(np.square(other_agent_pos), axis=2))
                other_dist_idx = np.argsort(other_dist, axis=1)
                row_idx = np.arange(self.num_adversaries).repeat(self.num_good_agents)
                self.sorted_other_agent_pos = other_agent_pos[row_idx, other_dist_idx.reshape(-1)].reshape(self.num_adversaries,
                                                                                               self.num_good_agents, 2)

                row_idx = np.arange(self.num_adversaries).repeat(self.num_good_agents)
                self.sorted_other_agent_vel = other_agent_vel[row_idx, other_dist_idx.reshape(-1)].reshape(self.num_adversaries,
                                                                                               self.num_good_agents, 2)

                self.sorted_other_agent_vel = self.sorted_other_agent_vel[:, :self.n_adv_visible_agent, :]
                self.sorted_other_agent_pos = self.sorted_other_agent_pos[:, :self.n_adv_visible_agent, :]
                self.sorted_other_adv_pos = self.sorted_other_adv_pos[:, :self.n_adv_visible_adv, :]
                self.sorted_entity_pos = self.sorted_entity_pos[:, :self.n_adv_visible_landmark, :]

            obs = np.concatenate((np.array([agent.state.p_vel]), np.array([agent.state.p_pos]),
                                  self.sorted_entity_pos[agent.id, :, :],
                                  self.sorted_other_adv_pos[agent.id, :, :], self.sorted_other_agent_pos[agent.id, :, :], self.sorted_other_agent_vel[agent.id, :, :]), axis=0).reshape(-1)

            return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)