from typing import Optional

import gymnasium as gym
from gymnasium import spaces

from biggym.rewards import SimpleMATSimTraceScorer
from biggym.sims import RandomTravelSim


class SchedulerModeEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        duration: int = 24,
        steps: int = 96,
        distances: Optional[dict] = None,
        initial: int = 0,
    ):
        """
        Single agent environment for scheduling work activities and associated travel.

        Agent starts at home, work or shop (based on 'initial', default home), and can travel between
        home, work and shop at each time step. Agent can travel by car, bus or walk.

        At each time step the agent is either at home, at work, shopping or traveling.
        This results in a no-op if agent is already at a chosen activity or is traveling.

        Args:
            duration (int): total duration of simulation in hours
            steps (int): number of steps in simulation
            distances (dict): distance between activities as nested dict
            initial (int): initial activity idx
        """
        self.duration = duration
        self.steps = steps
        self.distances = distances
        self.initial = initial
        self.distances = distances
        if self.distances is None:
            self.distances = {
                "home": {"work": 10.0, "shop": 11.0},
                "work": {"home": 10.0, "shop": 1.0},
                "shop": {"home": 11.0, "work": 1.0},
            }

        self.time_step = duration / steps

        self.observation_space = spaces.Dict(
            {
                "current_state": spaces.Discrete(
                    6
                ),  # at home/work/shop or traveling by car/bus/walk
                "time": spaces.Box(
                    low=0, high=duration, shape=(1,)
                ),  # time of day (~progress)
            }
        )
        self._obs_space_map = {
            0: "act:home",
            1: "act:work",
            2: "act:shop",
            3: "trip:car",
            4: "trip:bus",
            5: "trip:walk",
        }
        self._obs_space_map_inv = {v: k for k, v in self._obs_space_map.items()}

        self.action_space = spaces.Discrete(9)
        self._action_space_map = {
            0: "home:car",
            1: "home:bus",
            2: "home:walk",
            3: "work:car",
            4: "work:bus",
            5: "work:walk",
            6: "shop:car",
            7: "shop:bus",
            8: "shop:walk",
        }
        self._action_space_mapping_inv = {
            v: k for k, v in self._action_space_map.items()
        }

        # simulate travel time
        self._travel_sim = {
            "car": RandomTravelSim(speed=30, constant=0.05),  # default peaks
            "bus": RandomTravelSim(speed=15, peaks=[], constant=0.2),
            "walk": RandomTravelSim(speed=4, peaks=[], constant=0.0),
        }

        # score trace
        self._trace_scorer = SimpleMATSimTraceScorer()

    def _get_obs(self):
        return {"current_state": self._agent_state, "time": self._time}

    def _get_info(self):
        return {"trace": self._trace}

    def reset(self):
        self._agent_state = self.initial
        self._time = 0
        self._trace = [
            [self._agent_state, self.time_step, 0]
        ]  # [[label, duration, distance],]
        self._destination = None
        self._remaining_travel_time = 0
        return self._get_obs(), self._get_info()

    def step(self, action_idx):
        self._time += self.time_step

        if self._agent_state > 2:  # traveling
            self._update_travel()
            self._extent_trace()
            return (
                self._get_obs(),
                self._reward(),
                self._terminated(),
                False,
                self._get_info(),
            )

        current_act = self._obs_space_map[self._agent_state].split(":")[1]
        action_act = self._action_space_map[action_idx].split(":")[0]

        if current_act == action_act:  # no travel, stay at current activity
            self._extent_trace()
            return (
                self._get_obs(),
                self._reward(),
                self._terminated(),
                False,
                self._get_info(),
            )

        # else start travel
        self._travel_to_new_activity(action_idx)
        self._extent_trace()
        return (
            self._get_obs(),
            self._reward(),
            self._terminated(),
            False,
            self._get_info(),
        )

    def _reward(self):
        return self._trace_scorer.score(
            trace=self._trace, obs_map=self._obs_space_map
        )

    def _travel_to_new_activity(self, action_idx):
        """Move state to traveling, change destination, calc travel time."""
        action = self._action_space_map[action_idx]
        current_act = self._obs_space_map[self._agent_state].split(":")[1]
        action_act, action_mode = action.split(":")
        print(current_act, action_act, action_mode)
        distance = self.distances[current_act][action_act]
        self._agent_state = self._obs_space_map_inv[f"trip:{action_mode}"]
        self._destination = self._obs_space_map_inv[f"act:{action_act}"]
        mode_travel_sim = self._travel_sim[action_mode]
        self._remaining_travel_time = (
            mode_travel_sim.sample(distance, self._time) - self.time_step
        )

    def _update_travel(self):
        if self._remaining_travel_time <= 0:  # arrive!
            self._agent_state = self._destination
            self._destination = None
        else:  # still traveling
            self._remaining_travel_time -= self.time_step

    def _extent_trace(self):
        if (
            self._agent_state == self._trace[-1][0]
        ):  # no change, extend current activity
            self._trace[-1][1] += self.time_step
        elif self._agent_state == 2:  # new travel
            self._trace.append(
                [self._agent_state, self.time_step, self.distances]
            )
        else:  # new act
            self._trace.append([self._agent_state, self.time_step, 0])

    def _terminated(self):
        return self._time + self.time_step >= self.duration
