import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import revert_state  # TODO am bad at relative imports, is this okay?
import sys
from typing import Any
import math


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def numpy_to_cuda(numpy_array):
    return torch.from_numpy(numpy_array).float().to(DEVICE)


class DQNNet(nn.Module):
    """Outputs Q-values for each action"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQNNet, self).__init__()

        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   )

        self.q = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        l = self.layer(obs)
        q_values = self.q(l)

        return q_values


class DuellingDQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DuellingDQNNet, self).__init__()

        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)

        self.adv = nn.Linear(hidden_dim, action_dim)
        self.val = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        obs = obs.float()

        l1 = F.relu(self.layer_1(obs))
        l2 = F.relu(self.layer_2(l1))
        l3 = F.relu(self.layer_3(l2))

        advantages = self.adv(l3)
        value = self.val(l3)
        a_values = value + (advantages - advantages.mean(0, keepdim=True))

        return a_values


class DQN:
    """DQN implementation with epsilon greedy actions selection"""

    def __init__(self, env, config, gamma=0.99, lr=0.002357, tau=0.0877, rho=0.7052, epsilon=1., polyak=False,
                 decay=0.5, step_decay=50000):

        self.env = env

        action_dim = env.action_space.n
        state_dim = 3  # env.observation_space.n

        # create simple networks that output Q-values, both target and policy are identical
        self.target_net = self.create_net(state_dim, action_dim, duelling=False).to(DEVICE)
        self.policy_net = self.create_net(state_dim, action_dim, duelling=False).to(DEVICE)

        # We use the Adam optimizer
        self.lr = lr
        self.decay = decay
        self.step_decay = step_decay
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=self.decay, step_size=step_decay)
        self.action_size = action_dim
        self.gamma = gamma

        # For copying networks
        self.tau = tau
        self.counter = 0
        self.polyak = polyak

        # We decay epsilon according to the following formula
        self.t = 1
        self.rho = rho
        self.epsilon = lambda t: 0.01 + epsilon / (t ** self.rho)
        self.total_steps = env.steps * config.NUM_EPISODES * (1 + (1 - config.EPS_DELAY))
        self.epsilon = lambda t: (((config.EPS_START - config.EPS_END) *
                                  math.exp(-1 * t / config.EPS_DECAY) + config.EPS_END))
        # self.epsilon = config.EPS

        self.eps_slope = 1 / (config.EPS_DECAY * (config.NUM_EPISODES * env.steps))


    # def observe(self, vals):  # added random extra thing as needed
    #     pass

    @staticmethod
    def create_net(s_dim, a_dim, duelling):
        if duelling:
            net = DuellingDQNNet(s_dim, a_dim)
        else:
            net = DQNNet(s_dim, a_dim)
        return net

    @torch.no_grad()
    def act(self, state: int, step: int, key: Any, testing: bool = False) -> int:
        state = revert_state(state, step - 1)
        invalid_acts = self.env.get_illegal_moves(state)
        valid_acts = self.env.get_legal_moves(state)
        state = F.one_hot(torch.tensor(state), num_classes=3).unsqueeze(0)
        self.t += 1
        if np.random.uniform() < self.epsilon(step) or testing:
            act = np.random.choice(valid_acts)
        else:
            q_values = self.policy_net((state.to(DEVICE))).cpu().numpy()
            q_values[:, invalid_acts] = -1e10
            act = np.argmax(q_values)

        return act

    def update(self, batch_sample, weights=None):
        """To update our networks"""
        # Unpack batch: 5-tuple
        obs, action, next_state_index, reward, nobs, done = batch_sample

        # convert to torch.cuda
        states = numpy_to_cuda(obs)
        actions = numpy_to_cuda(action).type(torch.int64).unsqueeze(1)
        next_state_index = numpy_to_cuda(next_state_index)
        next_states = numpy_to_cuda(nobs)
        rewards = numpy_to_cuda(reward)
        done = numpy_to_cuda(done)

        q_vals = self.policy_net(states).gather(1, actions).squeeze(1)
        # TODO do need to add avail actions here as theres no chance a non valid action could be taken?

        next_q_vals = self.next_state_value_estimation(next_states, next_state_index, done)
        # TODO adjusted it in this value function I think, cus it found max Q vals so needed to invalidate

        target_q = (rewards + self.gamma * (1 - done) * next_q_vals.squeeze(1))

        if weights is not None:
            weights = numpy_to_cuda(weights)
            loss = ((target_q - q_vals).pow(2) * weights).mean()
        else:
            loss = nn.MSELoss(q_vals, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.scheduler.step()

        # self.epsilon -= self.eps_slope
        # self.epsilon = np.clip(self.epsilon, 0, 1)  # prevents epsilon from being outside this range

        self.copy_nets()

        return loss, (q_vals - target_q).detach()

    @torch.no_grad()
    def next_state_value_estimation(self, next_states, next_state_indices, done):
        """Function to define the value of the next state, makes inheritance cleaner"""
        next_state_values = self.target_net(next_states).max(1)[0].detach()

        print("WRONG ONE MATEY")
        sys.exit()
        return next_state_values.unsqueeze(1)

    def copy_nets(self):
        """Copies the parameters from the policy network to the target network, either all at once or incrementally."""
        self.counter += 1
        if not self.polyak and self.counter >= 1 / self.tau:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.counter = 0
        else:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def __str__(self):
        return "DQN"


class D3QN(DQN):
    """Implementation of DuelDDQN, inspired by RAINBOW"""

    def __init__(self, env, config, lr=0.004133, rho=0.5307, tau=0.01856, **kwargs):
        super(D3QN, self).__init__(env, config, lr=lr, rho=rho, tau=tau, **kwargs)
        # create duelling networks that output Q-values, both target and policy are identical
        action_dim = env.action_space.n
        state_dim = 3  # env.observation_space.n
        self.target_net = self.create_net(state_dim, action_dim, duelling=True).to(DEVICE)
        self.policy_net = self.create_net(state_dim, action_dim, duelling=True).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=self.decay, step_size=self.step_decay)

    @torch.no_grad()
    def next_state_value_estimation(self, next_states, next_state_indices, done):
        """next state value estimation is different for DDQN,
        decouples action selection and evaluation for reduced estimation bias"""
        # find max valued action with policy net
        max_actions = self.policy_net(next_states)
        max_actions = max_actions + next_state_indices  # TODO is this okay?
        max_actions = max_actions.argmax(1).unsqueeze(1)
        # estimate value of best action with target net
        next_state_values = self.target_net(next_states).gather(1, max_actions)

        return next_state_values

    def __str__(self):
        return "D3QN"