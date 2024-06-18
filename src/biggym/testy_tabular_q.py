from envs.scheduler import SchedulerEnv
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sys
from config import get_config
import wandb


def state_index(obs: np.array, step: int) -> int:
    # takes in the int of 0,1,2 from the state and then converts to extended temporal state idk if this is right
    state_num = int(obs["curr_state"])
    return state_num + (3 * step)


def revert_state(state_num: int, step: int) -> np.array:
    return state_num - (3 * step)


config = get_config()
wandb.init(config=config,
           project="BIGGYM",
           entity=config.WANDB_ENTITY,
           group="INITIAL_TESTS",
           mode=config.WANDB)

env = SchedulerEnv()
# env = gym.make("biggym/scheduler-v0")  # TODO get an error of ModuleNotFoundError: No module named 'gym_examples'

# TABULAR Q LEARNING BELOW
observation, info = env.reset(seed=config.SEED)  # initial fake reset to get required shape, there may be better way to do this
trajectory_array = np.zeros_like(info["trace"])[np.newaxis, :]

Q = np.zeros([env.observation_space.n, env.action_space.n])

epsilon = config.EPS
tot_steps = config.NUM_EPISODES * env.steps  # tied to env.steps
eps_slope = 1 / (config.EPS_DECAY * tot_steps)  # leaves the last (1-decay)% of steps for full exploit, can adjust

action_policy = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

for i in range(config.NUM_EPISODES):
    step = 0
    obs, info = env.reset(seed=42)
    obs = state_index(obs, step)
    reward_tot = 0
    done = False

    while step < env.steps:  # TODO double check it does enough steps
        step += 1

        # validate which actions can be picked
        invalid_acts = env.get_illegal_moves(revert_state(obs, step-1))
        valid_acts = env.get_legal_moves(revert_state(obs, step-1))
        valid_Q = Q.copy()
        valid_Q[:, invalid_acts] = -1e10

        # Epsilon greedy for some weak exploration, idk if this is good for tabular q but eyo
        if np.random.uniform() < epsilon:
            act = np.random.choice(valid_acts)
        else:
            act = np.argmax(valid_Q[obs, :])
        # act = 2
        # act = action_policy[step-1]

        nobs, reward, done, _, info = env.step(act)  # TODO theres no temporally truncated episodes right?
        nobs = state_index(nobs, step)

        # below is for sparse rewards, also should change in scheduler
        if (step - 1) == env.steps:
            reward = env._get_reward(last=True)

        invalid_acts_nobs = env.get_illegal_moves(revert_state(nobs, step))
        valid_acts_nobs = env.get_legal_moves(revert_state(nobs, step))
        valid_Q_nobs = Q.copy()
        valid_Q_nobs[:, invalid_acts_nobs] = -1e10

        if np.random.uniform() < epsilon:
            act_nobs = np.random.choice(valid_acts_nobs)
        else:
            act_nobs = np.argmax(valid_Q_nobs[nobs, :])

        # sarsa below
        Q[obs, act] = Q[obs, act] + config.LR * (reward + config.GAMMA * valid_Q_nobs[nobs, act_nobs] - Q[obs, act])

        # q learning below
        # Q[obs, act] = Q[obs, act] + config.LR * (reward + config.GAMMA * np.max(valid_Q_nobs[nobs, :]) - Q[obs, act])
        reward_tot += reward
        obs = nobs

        epsilon = epsilon - eps_slope
        epsilon = np.clip(epsilon, 0, 1)  # prevents epsilon from being outside this range

        if done:
            trajectory_array = np.concatenate([trajectory_array, info["trace"][np.newaxis, :]])
            wandb.log(data={"episode_reward": reward_tot, "epsilon": epsilon})
            # print(env._trace)
            break
env.close()

# print(f"Q-Table Vals: \n {Q}")
# Q = np.swapaxes(Q, 0, 1)
# Q = Q.reshape(-1, Q.shape[1] // 3, order="F")
# plt.pcolormesh(Q)
# ax = plt.gca()
# ax.set_aspect(4)
# ax.set_yticks(ticks=[0.5, 1.5, 2.5], labels=["Home", "Travel", "Work"])
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
