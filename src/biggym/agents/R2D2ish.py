import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import revert_state  # TODO am bad at relative imports, is this okay?
import sys
from typing import Any
import math
from typing import Dict

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
    def __init__(self, state_dim, action_dim, config, hidden_dim=64):
        super(DuellingDQNNet, self).__init__()

        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, config.HIDDEN_SIZE, batch_first=True)  # TODO check these are the best sizes or so

        self.adv = nn.Linear(hidden_dim, action_dim)
        self.val = nn.Linear(hidden_dim, 1)

    def forward(self, obs, hidden_state, carry_state):
        obs = obs.float()

        l1 = F.relu(self.layer_1(obs))
        l2 = F.relu(self.layer_2(l1))
        l3 = F.relu(self.layer_3(l2))

        recurrent_output, (hidden_state_new, carry_state_new) = self.lstm(l3, (hidden_state, carry_state))  # TODO check this

        advantages = self.adv(recurrent_output)
        value = self.val(recurrent_output)
        a_values = value + (advantages - advantages.mean(0, keepdim=True))

        return a_values, hidden_state_new, carry_state_new


class R2D2:
    """DQN implementation with epsilon greedy actions selection"""

    def __init__(self, env, config, gamma=0.99, lr=0.004133, tau=0.01856, rho=0.5307, epsilon=1., polyak=False,
                 decay=0.5, step_decay=50000):
        self.env = env
        self.config = config

        action_dim = env.action_space.n
        state_dim = 3  # env.observation_space.n

        # create simple networks that output Q-values, both target and policy are identical
        self.target_net = self.create_net(state_dim, action_dim, duelling=True).to(DEVICE)
        self.policy_net = self.create_net(state_dim, action_dim, duelling=True).to(DEVICE)

        # We use the Adam optimizer
        self.lr = lr
        self.decay = decay
        self.step_decay = step_decay
        self.optimiser = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, gamma=self.decay, step_size=step_decay)
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

    def init_hidden_state(self, training):
        if training is True:
            return (torch.zeros([1, self.config.EP_BATCH_SIZE, self.config.HIDDEN_SIZE]),
                    torch.zeros([1, self.config.EP_BATCH_SIZE, self.config.HIDDEN_SIZE]))
        else:
            return (torch.zeros([1, 1, self.config.HIDDEN_SIZE]),
                    torch.zeros([1, 1, self.config.HIDDEN_SIZE]))
            # TODO check this copies okay cus lol

    def observe(self, vals):  # added random extra thing as needed
        pass

    def create_net(self, s_dim, a_dim, duelling):
        if duelling:
            net = DuellingDQNNet(s_dim, a_dim, self.config)
        else:
            net = DQNNet(s_dim, a_dim)
        return net

    @torch.no_grad()
    def act(self, state: int, step: int, h_s, c_s, testing: bool = False):  # TODO update restrictions
        state = revert_state(state, step - 1)
        invalid_acts = self.env.get_illegal_moves(state)
        valid_acts = self.env.get_legal_moves(state)
        state = F.one_hot(torch.tensor(state), num_classes=3).unsqueeze(0).unsqueeze(0)
        self.t += 1

        q_values, hidden_state, carry_state = self.policy_net((state.to(DEVICE)),
                                                              h_s.to(DEVICE),
                                                              c_s.to(DEVICE))
        q_values = q_values.cpu().numpy()  # TODO do we need this?

        if np.random.uniform() < self.epsilon(step) or testing:
            act = np.random.choice(valid_acts)
        else:
            q_values[:, :, invalid_acts] = -1e10
            act = np.argmax(q_values)

        return act, hidden_state, carry_state

    def update(self, episode_memory, weights=None):
        """To update our networks"""
        samples, seq_len = episode_memory.sample()

        states = []
        actions = []
        next_state_index = []
        rewards = []
        next_states = []
        done = []

        for i in range(self.config.EP_BATCH_SIZE):
            states.append(samples[i]["obs"])
            actions.append(samples[i]["acts"])
            next_state_index.append(samples[i]["next_state_index"])
            rewards.append(samples[i]["rews"])
            next_states.append(samples[i]["next_obs"])
            done.append(samples[i]["done"])

        obs = np.array(states)
        action = np.array(actions)
        next_state_index = np.array(next_state_index)
        rewards = np.array(rewards)
        nobs = np.array(next_states)
        done = np.array(done)  # TODO fix the above and below as it is SO BAD

        states = torch.FloatTensor(obs.reshape(self.config.EP_BATCH_SIZE, seq_len, -1)).to(DEVICE)
        actions = torch.LongTensor(action.reshape(self.config.EP_BATCH_SIZE, seq_len, -1)).to(DEVICE)
        next_state_index = torch.LongTensor(next_state_index.reshape(self.config.EP_BATCH_SIZE, seq_len, -1)).to(DEVICE)
        rewards = torch.FloatTensor(rewards.reshape(self.config.EP_BATCH_SIZE, seq_len, -1)).to(DEVICE)
        next_states = torch.FloatTensor(nobs.reshape(self.config.EP_BATCH_SIZE, seq_len, -1)).to(DEVICE)
        done = torch.FloatTensor(done.reshape(self.config.EP_BATCH_SIZE, seq_len, -1)).to(DEVICE)

        # # Unpack batch: 6-tuple
        # # obs, action, next_state_index, reward, nobs, done = samples[0]
        #
        # # convert to torch.cuda
        # states = numpy_to_cuda(obs)
        # actions = numpy_to_cuda(action).type(torch.int64).unsqueeze(1)
        # next_state_index = numpy_to_cuda(next_state_index)
        # next_states = numpy_to_cuda(nobs)
        # rewards = numpy_to_cuda(reward)
        # done = numpy_to_cuda(done)

        h_s, c_s = self.init_hidden_state(training=True)
        q_vals, _, _ = self.policy_net(states, h_s.to(DEVICE), c_s.to(DEVICE))
        q_vals = q_vals.gather(-1, actions).squeeze(-1)
        # TODO do need to add avail actions here as theres no chance a non valid action could be taken?

        h_s, c_s = self.init_hidden_state(training=True)  # TODO is this correct to reinitialise?
        next_q_vals = self.next_state_value_estimation(next_states, next_state_index, h_s, c_s)
        # TODO adjusted it in this value function I think, cus it found max Q vals so needed to invalidate

        target_q = (rewards + self.gamma * (1 - done) * next_q_vals).squeeze(-1)  # TODO idk if dimensions are correct?

        loss = F.smooth_l1_loss(q_vals, target_q)  # nn.MSELoss(q_vals, target_q)  # TODO figure out otehr loss

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.scheduler.step()

        self.copy_nets()  # TODO this shouldn't always happen rather be every now and again

        return loss, (q_vals - target_q).detach()

    @torch.no_grad()
    def next_state_value_estimation(self, next_states, next_state_indices, h_s, c_s):
        """next state value estimation is different for DDQN,
        decouples action selection and evaluation for reduced estimation bias"""
        # find max valued action with policy net
        max_actions, _, _ = self.policy_net(next_states, h_s.to(DEVICE), c_s.to(DEVICE))
        max_actions = max_actions + next_state_indices  # TODO is this okay?
        max_actions = max_actions.argmax(-1).unsqueeze(-1)
        # estimate value of best action with target net
        h_s, c_s = self.init_hidden_state(training=True)  # TODO do we need to reinit again?
        next_state_values, _, _ = self.target_net(next_states, h_s.to(DEVICE), c_s.to(DEVICE))
        next_state_values = next_state_values.gather(-1, max_actions)

        return next_state_values

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
        return "R2D2"


class EpisodeMemory:
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False,
                 max_epi_num=100, max_epi_len=500,
                 batch_size=1,
                 lookup_step=None):
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        if (random_update is False) and (self.batch_size > 1):
            sys.exit(
                'It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update:  # Random upodate
            sampled_episodes = random.sample(self.memory, self.batch_size)

            check_flag = True  # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))  # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode) - min_step + 1)  # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################
        else:  # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs'])  # buffers, sequence_length

    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.next_move_index = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.next_move_index.append(transition[2])
        self.reward.append(transition[3])
        self.next_obs.append(transition[4])
        self.done.append(transition[5])

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        next_move_index = np.array(self.next_move_index)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx + lookup_step]
            action = action[idx:idx + lookup_step]
            next_move_index = next_move_index[idx:idx + lookup_step]  # TODO check this is correct
            reward = reward[idx:idx + lookup_step]
            next_obs = next_obs[idx:idx + lookup_step]
            done = done[idx:idx + lookup_step]

        return dict(obs=obs,
                    acts=action,
                    next_state_index=next_move_index,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)
