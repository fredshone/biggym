import numpy as np
from .utils import revert_state  # TODO am bad at relative imports, is this okay?
import copy


class Tabular_Q:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.epsilon = config.EPS_START
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.eps_slope = 1 / (config.EPS_DELAY * (config.NUM_EPISODES * env.steps))

    def observe(self, vals):
        pass

    def act(self, obs, step, key):
        # validate which actions can be picked
        invalid_acts = self.env.get_illegal_moves(revert_state(obs, step - 1))
        valid_acts = self.env.get_legal_moves(revert_state(obs, step - 1))
        valid_Q = copy.deepcopy(self.Q)
        valid_Q[:, invalid_acts] = -1e10

        # Epsilon greedy for some weak exploration, idk if this is good for tabular q but eyo
        if np.random.uniform() < self.epsilon:
            act = np.random.choice(valid_acts)
        else:
            act = np.argmax(valid_Q[obs, :])

        return act

    def update(self, obs, act, reward, nobs, step, key):
        invalid_acts_nobs = self.env.get_illegal_moves(revert_state(nobs, step))
        valid_acts_nobs = self.env.get_legal_moves(revert_state(nobs, step))
        valid_Q_nobs = copy.deepcopy(self.Q)
        valid_Q_nobs[:, invalid_acts_nobs] = -1e10

        # q learning below
        self.Q[obs, act] = self.Q[obs, act] + self.config.LR * (reward + self.config.GAMMA * np.max(valid_Q_nobs[nobs, :]) - self.Q[obs, act])

        self.epsilon -= self.eps_slope
        self.epsilon = np.clip(self.epsilon, 0, 1)  # prevents epsilon from being outside this range
