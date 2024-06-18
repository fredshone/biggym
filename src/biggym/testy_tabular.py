from envs.scheduler import SchedulerEnv
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sys
from config import get_config
import wandb
from agents.PSRL import PSRLAgent
from agents.Tabular_Q import Tabular_Q
from agents.Tabular_SARSA import Tabular_SARSA
import jax.random as jrandom
import jax.numpy as jnp
import jax
import os
from agents.utils import state_index


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
agent = PSRLAgent(env, config, _key)
# agent = Tabular_SARSA(env, config)
# agent = Tabular_Q(env, config)

for i in range(config.NUM_EPISODES):
    step = 0
    obs, info = env.reset(seed=42)
    obs = state_index(obs, step)
    reward_tot = 0
    done = False

    reward_list = []

    while True:
        step += 1

        # BAYESIAN EXPLORATION BABYYYY  - with PSRL onlY
        key, _key = jrandom.split(key)
        action = agent.act(obs, step, _key)
        # action = 2
        # action = action_policy[step-1]

        nobs, reward, done, _, info = env.step(action)
        nobs = state_index(nobs, step)

        reward_list.append(reward)

        # below is for sparse rewards, also should change in scheduler
        # if (step - 1) == env.steps:
        #     reward = env._get_reward(last=True)

        agent.observe((obs, action, reward, nobs))
        key, _key = jrandom.split(key)
        agent.update(obs, action, reward, nobs, step, _key)

        reward_tot += reward
        obs = nobs

        if done:
            trajectory_array = np.concatenate([trajectory_array, info["trace_2"][np.newaxis, :]])
            wandb.log(data={"episode_reward": reward_tot, "epsilon": agent.epsilon})
            break
env.close()

# print(reward_tot)
# print(env._trace)
# print(env._trace_2)
# plt.plot(reward_list)
# plt.show()
# sys.exit()

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
