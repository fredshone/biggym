import numpy as np

import gymnasium as gym
from gymnasium import spaces

from biggym.scorers import SimpleMATSimTraceScorer


class SchedulerEnv(gym.Env):
    metadata = {}

    def __init__(self, duration: int = 24, steps: int = 96, distance: float = 10.0):
        """
        Single agent environment for scheduling work activities and associated travel.

        At each step, agent is either at home, at work, or traveling (by car). At each step, agent
        chooses to travel to work or travel to home. This can result in no-op if agent is already
        traveling or already at chosen activity.

        Args:
            duration (int): total duration of simulation in hours
            steps (int): number of steps in simulation
            distance (float): distance between home and work in km
        """
        self.duration = duration
        self.steps = steps
        self.distance = distance
        self.step_size = duration / steps

        self.observation_space = spaces.Dict(
            {
                "current_state": spaces.Discrete(3),  # at home or at work or traveling
                "time": spaces.Box(
                    low=0, high=duration, shape=(1,)
                ),  # time of day (~progress)
            }
        )
        self._observation_space_mapping = {
            0: "act:home",
            1: "act:work",
            2: "trip:car",
        }

        self.action_space = spaces.Discrete(2)  # travel to home (0) or work (1)
        self._action_space_mapping = {0: "travel_to_home", 1: "travel_to_work"}
        # note that these need to match obs space for activities

        # simulate travel time
        self._travel_sim = RandomTravelSim()

        # score trace
        self._trace_scorer = SimpleMATSimTraceScorer()

    def _get_obs(self):
        return {"current_state": self._agent_state, "time": self._time}

    def _get_info(self):
        return {"trace": self._trace}

    def reset(self):
        self._time = 0
        self._trace = []  # [(label, start, end, distance)]
        self._agent_state = 0
        self._destination = None
        self._travel_time = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        self._time += self.step_size

        if self._agent_state == 2:  # traveling
            self._update_travel()
            self._extent_trace()
            return (
                self._get_obs(),
                self._reward(),
                self._terminated(),
                False,
                self._get_info(),
            )

        if action == self._agent_state:  # no travel, stay at current activity
            self._extend_trace()
            return (
                self._get_obs(),
                self._reward(),
                self._terminated(),
                False,
                self._get_info(),
            )

        # else start travel
        self._travel_to_new_activity(action)
        self._extent_trace()

    def _reward(self):
        return self._trace_scorer.score(
            trace=self._trace, act_map=self._observation_space_mapping
        )

    def _travel_to_new_activity(self, action):
        """Move state to traveling, change destination, calc travel time."""
        self._agent_state = 2
        self._destination = action
        self._travel_time = self._travel_sim.sample(self.distance, self._time)

    def _update_travel(self):
        if self._travel_time <= 0:  # new activity started
            self._agent_state = self._destination
            self._extent_trace()
            self._destination = None
        else:  # still traveling
            self._travel_time -= self.step_size

    def _extent_trace(self):
        if (
            self._agent_state == self._trace[-1][0]
        ):  # no change, extend current activity
            self._trace[-1][1] += self.step_size
        elif self._agent_state == 2:  # new travel
            self._trace.append((self._destination, self.step_size, self.distance))
        else:  # new act
            self._trace.append((self._agent_state, self.step_size, 0))

    def _terminated(self):
        return self._time >= self.duration


class RandomTravelSim:
    """
    Sample travel times based on time of day.

    peak time
            |
           /\          |
          /  \         | max delay
         /    \        |
    ____/      \___    |
        |-------|
          peak duration

    0  6  9  12  15  18  21  24
    """

    def __init__(self) -> None:
        self.speed = 30  # km/h
        self.peaks = [
            (9, 6, 0.5),
            (12, 5, 0.25),
            (17, 6, 0.5),
        ]  # [(peak_time, peak_duration, max_delay)]

    def _delay(self, time: float) -> float:
        delay = 0
        for peak_time, peak_duration, max_delay in self.peaks:
            peak_distance = abs(time - peak_time)
            peak_shoulder = peak_duration / 2
            if peak_distance < peak_shoulder:
                peak_delay = (1 - (peak_distance / peak_shoulder)) * max_delay
                if peak_delay > delay:
                    delay = peak_delay
        return delay

    def sample(self, distance: float, time: float) -> float:
        return (distance / self.speed) + self._delay(time)
