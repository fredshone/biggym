from gymnasium.envs.registration import register

from biggym.envs.scheduler import SchedulerEnv
from biggym.envs.scheduler_modes import SchedulerModeEnv

register(
    id="biggym/scheduler-v0",
    # entry_point="gym_examples.envs:SchedulingEnv",
    entry_point="gym_examples.envs:SchedulerEnv",
    max_episode_steps=1440,
)
