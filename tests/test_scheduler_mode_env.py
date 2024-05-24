import pytest
from biggym.envs import SchedulerModeEnv


@pytest.mark.parametrize(
    "duration, steps", [(24, 6), (24, 12), (24, 24), (24, 96)]
)
def test_env_random_actions(duration, steps):
    env = SchedulerModeEnv(duration=duration, steps=steps)
    for _ in range(10):
        obs, info = env.reset()
        step_counter = 1
        while True:
            step_counter += 1
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            if done:
                break
        assert step_counter == steps
        assert sum([d for _, d, _ in info["trace"]]) == duration


@pytest.mark.parametrize(
    "duration, steps", [(24, 6), (24, 12), (24, 24), (24, 96)]
)
def test_env_flipper_actions(duration, steps):
    env = SchedulerModeEnv(duration=duration, steps=steps)
    obs, info = env.reset()
    print("obs", obs)
    step_counter = 1
    while True:
        step_counter += 1
        current_state = obs["current_state"]
        action = 1 - current_state
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    assert step_counter == steps
    assert sum([d for _, d, _ in info["trace"]]) == duration


@pytest.mark.parametrize(
    "duration, steps", [(24, 6), (24, 12), (24, 24), (24, 96)]
)
def test_env_stay_at_home(duration, steps):
    env = SchedulerModeEnv(duration=duration, steps=steps)
    obs, info = env.reset()
    step_counter = 1
    while True:
        step_counter += 1
        obs, reward, done, _, info = env.step(0)
        if done:
            break
    assert step_counter == steps
    assert sum([d for _, d, _ in info["trace"]]) == duration
    assert reward > 0


@pytest.mark.parametrize("duration, steps", [(24, 24), (24, 96)])
def test_env_work_9_5_car(duration, steps):
    env = SchedulerModeEnv(duration=duration, steps=steps)
    obs, info = env.reset()
    step_counter = 1
    while True:
        time = obs["time"]
        if 11 < time < 13:
            action = 8
        if 8 < time < 18:
            action = 3
        else:
            action = 0
        step_counter += 1
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    assert step_counter == steps
    assert sum([d for _, d, _ in info["trace"]]) == duration
    assert reward > 0


@pytest.mark.parametrize("duration, steps", [(24, 24), (24, 96)])
def test_env_work_9_5_bus(duration, steps):
    env = SchedulerModeEnv(duration=duration, steps=steps)
    obs, info = env.reset()
    step_counter = 1
    while True:
        time = obs["time"]
        if 11 < time < 13:
            action = 8
        if 8 < time < 18:
            action = 4
        else:
            action = 0
        step_counter += 1
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    assert step_counter == steps
    assert sum([d for _, d, _ in info["trace"]]) == duration
    assert reward > 0
