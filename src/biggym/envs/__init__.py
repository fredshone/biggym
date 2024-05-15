from gymnasium.envs.registration import register

from biggym.envs.scheduler import SchedulerEnv

register(
    id="biggym/scheduler-v0",
    entry_point="gym_examples.envs:SchedulingEnv",
    max_episode_steps=300,
)
