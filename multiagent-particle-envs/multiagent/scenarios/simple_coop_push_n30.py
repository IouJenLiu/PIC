import numpy as np
import random
from multiagent.core_vec import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, sort_obs=True):
        world = World()
        self.np_rnd = np.random.RandomState(0)
        self.sort_obs = sort_obs
        # set any world properties first
        world.dim_c = 2
        num_agents = 30
        num_landmarks = 2
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        world.collaborative = True
        self.world_radius = 2
        self.n_others = 5
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.06
            agent.id = i
            if i < 10:
                agent.collide = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            if i < num_landmarks / 2:
                landmark.name = 'landmark %d' % i
                landmark.collide = True
                landmark.movable = True
                landmark.size = 0.7
                landmark.initial_mass = 4
            else:
                landmark.name = 'target %d' % (i - num_landmarks / 2)
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.05
                landmark.initial_mass = 4.0
        # make initial conditions
        self.color = {
                      'green': np.array([0.35, 0.85, 0.35]), 'blue': np.array([0.35, 0.35, 0.85]),'red': np.array([0.85, 0.35, 0.35]),
                      'light_blue': np.array([0.35, 0.85, 0.85]), 'yellow': np.array([0.85, 0.85, 0.35]), 'black': np.array([0.0, 0.0, 0.0])}
        self.collide_th = world.agents[0].size + world.landmarks[0].size
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.0, 0.0, 0.0])
        # random properties for landmarks
        color_keys = list(self.color.keys())
        for i, landmark in enumerate(world.landmarks):
            if i < len(world.landmarks) / 2:
                landmark.color = self.color[color_keys[i]] - 0.1
            else:
                landmark.color = self.color[color_keys[int(i / 2)]] + 0.1
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.np_rnd.uniform(-self.world_radius, +self.world_radius, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        num_landmark = int(len(world.landmarks) / 2)
        for i, landmark in enumerate(world.landmarks[:num_landmark]):
            landmark.state.p_pos = self.np_rnd.uniform(-(self.world_radius - 0.2) , +self.world_radius - 0.2, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, target in enumerate(world.landmarks[num_landmark:]):
            for _ in range(2):
                target.state.p_pos = self.np_rnd.uniform(-(self.world_radius - 0.2) , +self.world_radius - 0.2, world.dim_p)
                if np.sqrt(np.sum(np.square(target.state.p_pos - world.landmarks[i].state.p_pos))) > 1.2:
                    break
                print('regenerate l location')
            target.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew, rew1 = 0, 0
        n_collide, n_collide1 = 0, 0
        if agent == world.agents[0]:
            l, t = world.landmarks[0], world.landmarks[1]
            a_pos = np.array([[a.state.p_pos for a in world.agents]])
            dist = np.sqrt(np.sum(np.square(l.state.p_pos - t.state.p_pos)))
            rew -= 2 * dist
            dist2 = np.sqrt(np.sum(np.square(a_pos - l.state.p_pos), axis=2))
            rew -= 0.1 * np.min(dist2)
            n_collide = (dist2 < self.collide_th).sum()
            rew += 0.1 * n_collide
        return rew

    def observation(self, agent, world):
        """
        :param agent: an agent
        :param world: the current world
        :return: obs: [18] np array,
        [0-1] self_agent velocity
        [2-3] self_agent location
        [4-9] landmarks location
        [10-11] agent_i's relative location
        [12-13] agent_j's relative location
        Note that i < j
        """
        # get positions of all entities in this agent's reference frame
        if agent.id == 0:
            l_pos = np.array([[l.state.p_pos for l in world.landmarks]]).repeat(len(world.agents), axis=0)
            a_pos = np.array([[a.state.p_pos for a in world.agents]])
            a_pos1 = a_pos.repeat(len(world.agents), axis=0)
            a_pos1 = np.transpose(a_pos1, axes=(1, 0, 2))
            a_pos2 = a_pos.repeat(len(world.agents), axis=0)
            a_pos3 = a_pos.repeat(len(world.landmarks), axis=0)
            a_pos3 = np.transpose(a_pos3, axes=(1, 0, 2))
            entity_pos = l_pos - a_pos3
            other_pos = a_pos2 - a_pos1



            other_dist = np.sqrt(np.sum(np.square(other_pos), axis=2))
            other_dist_idx = np.argsort(other_dist, axis=1)
            row_idx = np.arange(self.num_agents).repeat(self.num_agents)
            self.sorted_other_pos = other_pos[row_idx, other_dist_idx.reshape(-1)].reshape(self.num_agents,
                                                                                            self.num_agents, 2)[:, 1:, :]
            self.sorted_other_pos = self.sorted_other_pos[:, :self.n_others, :]
            self.sorted_entity_pos = entity_pos

        obs = np.concatenate((np.array([agent.state.p_vel]), np.array([agent.state.p_pos]),
                              self.sorted_entity_pos[agent.id, :, :],
                              self.sorted_other_pos[agent.id, :, :]), axis=0).reshape(-1)
        return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
