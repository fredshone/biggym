from envs.scheduler import SchedulerEnv
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sys
from config import get_config
import wandb
from agents import PSRLAgent, Tabular_Q, Tabular_SARSA, D3QN, R2D2, EpisodeBuffer, EpisodeMemory
import jax.random as jrandom
import jax.numpy as jnp  # TODO sorry will write in pytorch
import jax
import os
from agents.utils import state_index, revert_state, update_tot_state
import utils
import torch
import torch.nn.functional as F


# TODO dirichlet needs x64 it seems? setting x64 allowed below
jax.config.update("jax_enable_x64", True)

config = get_config()
wandb.init(config=config,
           project="BIGGYM",
           entity=config.WANDB_ENTITY,
           group="INITIAL_TESTS",
           mode=config.WANDB)

env = SchedulerEnv()
# env = gym.make("biggym/scheduler-v0")  # TODO get an error of ModuleNotFoundError: No module named 'gym_examples'

observation, info = env.reset(seed=config.SEED)  # initial fake reset to get required shape, there may be better way to do this
trajectory_array = np.zeros_like(info["trace_2"])[np.newaxis, :]

action_policy = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

key = jrandom.PRNGKey(42)
key, _key = jrandom.split(key)
# agent = PSRLAgent(env, config, _key)
# agent = Tabular_SARSA(env, config)
# agent = Tabular_Q(env, config)
agent = D3QN(env, config)  # , rho=0.01)
# agent = R2D2(env, config)

if str(agent) == "D3QN" or str(agent) == "R2D2":  # TODO think it needs own replay buffer vibe
    alpha = 1.0
    decay_number = 10
    max_frames = config.NUM_EPISODES * env.steps
    agent.step_decay = int(max_frames / (decay_number + 1))
    beta = 0.7389
    memory = utils.PER_IS_ReplayBuffer(config.BUFFER_SIZE, alpha=alpha, state_dim=3)

if str(agent) == "R2D2":
    episode_memory = EpisodeMemory(random_update=config.RANDOM_UPDATE,
                                   max_epi_num=100, max_epi_len=env.steps,
                                   batch_size=config.EP_BATCH_SIZE,
                                   lookup_step=config.LOOKUP_STEP)

for episode in range(config.NUM_EPISODES):
    step = 0
    obs, info = env.reset(seed=42)
    obs = state_index(obs, step)
    reward_tot = 0
    done = False

    reward_list = []

    tot_obs = torch.zeros((env.steps))
    tot_obs = update_tot_state(tot_obs, obs, step)

    if str(agent) == "R2D2":
        episode_record = EpisodeBuffer()
        h_s, c_s = agent.init_hidden_state(training=False)

    while True:
        step += 1

        key, _key = jrandom.split(key)
        if str(agent) == "R2D2":
            action, h_s, c_s = agent.act(obs, step, h_s, c_s)
        elif str(agent) == "D3QN":
            # action = agent.act(tot_obs, step, _key)
            action = agent.act(obs, step, _key)
        else:
            action = agent.act(obs, step, _key)
        # action = 2
        # action = action_policy[step-1]

        nobs, reward, done, _, info = env.step(action)  # TODO check it does done when it is meant to
        nobs = state_index(nobs, step)
        tot_nobs = update_tot_state(tot_obs.clone(), nobs, step)
        reward_list.append(reward)

        # below is for sparse rewards, also should change in scheduler
        # if (step - 1) == env.steps:
        #     reward = env._get_reward(last=True)

        if str(agent) == "D3QN" or str(agent) == "R2D2":
            epsilon = agent.epsilon(agent.t)
            dqn_obs = revert_state(obs, step-1)
            dqn_nobs = revert_state(nobs, step)
            invalid_acts = env.get_illegal_moves(dqn_nobs)
            next_move_index = torch.zeros(3)
            next_move_index[invalid_acts] = -1e10
            dqn_obs = F.one_hot(torch.tensor(dqn_obs), num_classes=3)
            dqn_nobs = F.one_hot(torch.tensor(dqn_nobs), num_classes=3)
            if str(agent) == "D3QN":
                memory.push(dqn_obs, action, next_move_index, reward, dqn_nobs, done)
                if len(memory) > config.BATCH_SIZE:
                    # if we are using prioritised experience replay buffer with importance sampling
                    beta = 1 - (1 - beta) * np.exp(-0.05 * episode)
                    sample = memory.sample(config.BATCH_SIZE, beta)
                    loss, tds = agent.update(
                        [sample['obs'], sample['action'], sample["next_state_indices"], sample['reward'], sample['next_obs'], sample['done']],
                        weights=sample['weights']
                    )
                    new_tds = np.abs(tds.cpu().numpy()) + 1e-6
                    memory.update_priorities(sample['indexes'], new_tds)
            else:
                episode_record.put([dqn_obs, action, next_move_index, reward, dqn_nobs, done])
                if len(episode_memory) >= config.MIN_EP_NUM:
                    agent.update(episode_memory)
        elif str(agent) == "PSRL":
            agent.observe((obs, action, reward, nobs))
            key, _key = jrandom.split(key)
            agent.update(obs, action, reward, nobs, step, _key)
        else:
            agent.update(obs, action, reward, nobs, step, _key)
            epsilon = agent.epsilon

        reward_tot += reward
        obs = nobs
        tot_obs = tot_nobs.clone()  # TODO check this works

        if done:
            trajectory_array = np.concatenate([trajectory_array, info["trace_2"][np.newaxis, :]])
            wandb.log(data={"episode_reward": reward_tot, "epsilon": epsilon})
            if str(agent) == "R2D2":
                episode_memory.put(episode_record)
            sys.exit()
            break
env.close()

# print(info["trace"])
print(reward_tot)
print(env._trace_scorer.score(trace=info["trace"], obs_map=env._observation_space_mapping))
# TODO the above two should match, they don't but they have the exact same difference between optimal and stay at home
# TODO there is 113.204295429 reward going missing
# TODO added a questionable fix so will see if that helps
# print(env._trace)
# print(env._trace_2)
# plt.plot(reward_list)
# plt.show()

if config.PLOT:
    # TODO added some visualisation, have done this outside env below, but if you're happy could include in the env
    traj_array = trajectory_array[1:]
    traj_array = traj_array[-1]
    state_grid = np.zeros((traj_array.shape[1], traj_array.shape[0]))
    row_indices = traj_array[:, 0].astype(int)
    col_indices = np.arange(traj_array.shape[0])
    state_grid[row_indices, col_indices] = 1
    state_grid[[1, 2]] = state_grid[[2, 1]]
    plt.pcolormesh(state_grid, edgecolors="k", linewidth=1)
    ax = plt.gca()
    ax.set_aspect(4)
    ax.set_xticks(ticks=np.linspace(0, 96, 9), labels=[0, 3, 6, 9, 12, 15, 18, 21, 24])  # TODO improve this
    ax.set_yticks(ticks=[0.5, 1.5, 2.5], labels=["Home", "Travel", "Work"])
    plt.show()
