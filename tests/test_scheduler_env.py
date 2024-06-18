import pytest
from biggym.envs import SchedulerEnv
import random


@pytest.mark.parametrize(
    "duration, steps", [(24, 6), (24, 12), (24, 24), (24, 96)]
)
def test_env_random_actions(duration, steps):
    env = SchedulerEnv(duration=duration, steps=steps, distance=10.0)
    for _ in range(10):
        obs, info = env.reset()
        step_counter = 0
        while True:
            step_counter += 1
            action_choice = env.get_legal_moves(obs["curr_state"])
            action = random.choice(action_choice)
            obs, reward, done, _, info = env.step(action)
            if done:
                break
        assert step_counter == steps - 1
        assert sum([d for _, d, _ in info["trace"]]) == duration


@pytest.mark.parametrize(
    "duration, steps", [(24, 6), (24, 12), (24, 24), (24, 96)]
)
def test_env_flipper_actions(duration, steps):
    env = SchedulerEnv(duration=duration, steps=steps, distance=10.0)
    obs, info = env.reset()
    print("obs", obs)
    step_counter = 0
    while True:
        step_counter += 1
        current_state = obs["curr_state"]
        actions = env.get_legal_moves(current_state)
        action = min(actions)
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    assert step_counter == steps - 1
    assert sum([d for _, d, _ in info["trace"]]) == duration


@pytest.mark.parametrize(
    "duration, steps", [(24, 6), (24, 12), (24, 24), (24, 96)]
)
def test_env_stay_at_home(duration, steps):
    env = SchedulerEnv(duration=duration, steps=steps, distance=10.0)
    obs, info = env.reset()
    step_counter = 0
    while True:
        step_counter += 1
        obs, reward, done, _, info = env.step(2)
        if done:
            break
    assert step_counter == steps - 1
    assert sum([d for _, d, _ in info["trace"]]) == duration
    assert reward > 0


@pytest.mark.parametrize("duration, steps", [(24, 24), (24, 96)])
def test_env_work_9_5(duration, steps):
    env = SchedulerEnv(duration=duration, steps=steps, distance=10.0)
    obs, info = env.reset()
    step_counter = 0
    while True:
        time = obs["time"]
        if time == 8:
            action = 1
        elif time == 18:
            action = 0
        else:
            action = 2
        step_counter += 1
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    assert step_counter == steps - 1
    assert sum([d for _, d, _ in info["trace"]]) == duration
    assert reward > 0
