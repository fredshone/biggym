import gymnasium as gym
from gymnasium import spaces

from biggym.rewards import SimpleMATSimTraceScorer
from biggym.sims import RandomTravelSim

import numpy as np
import sys


class SchedulerEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        duration: int = 24,
        steps: int = 96,
        distance: float = 10.0,
        initial: int = 0,
    ):
        """
        Single agent environment for scheduling work activities and associated travel.

        Agent starts at home or work (based on 'initial'), and can travel between
        home and work at each time step.

        At each time step the agent is either at home, at work or traveling.
        This results in a no-op if agent is already at a chosen activity or is traveling.

        Args:
            duration (int): total duration of simulation in hours
            steps (int): number of steps in simulation
            distance (float): distance between home and work in km
            initial (int): initial activity idx
        """
        self.duration = duration
        self.steps = steps
        self.distance = distance
        self.initial = initial

        self.time_step = duration / steps

        self.observation_space = spaces.Discrete(3 * self.steps)
        self._observation_space_mapping = {
            0: "act:home",
            1: "act:work",
            2: "trip:car",
        }

        self.action_space = spaces.Discrete(3)  # travel to home (0) or work (1) or noop (3)
        self._action_space_mapping = {0: "travel_to_home", 1: "travel_to_work", 2: "noop"}  # TODO test just having two actions, travel or noop
        # note that these need to match obs space for activities

        # simulate travel time
        self._travel_sim = RandomTravelSim()

        # score trace
        self._trace_scorer = SimpleMATSimTraceScorer()

    def _get_obs(self):
        return {"curr_state": self._agent_state}

    def _get_info(self):
        return {"trace": self._trace_2}  # TODO this is trace_2 even tho keeping trace for reward calcs

    def reset(self, seed: int = None):  # TODO added seed as think has one in standard gym format altho not needed here
        self._agent_state = self.initial
        self._time = 0
        self._trace = [
            [self._agent_state, self.time_step, 0]
        ]  # [[label, duration, distance],]
        self._last_trace = self._trace.copy()
        self._trace_2 = np.zeros((self.steps, 3))  # TODO hardcode 3 for the shape but lmk if theres a variable for it
        self._trace_2[0] = [self._agent_state, 0, 0]
        self._destination = None
        self._remaining_travel_time = 0
        return self._get_obs(), self._get_info()

    def get_illegal_moves(self, state):
        # if travelling then it must be noop
        if state == 2:
            actions = [0, 1]
        elif state == 1:
            actions = [1]
        elif state == 0:
            actions = [0]
        else:
            print("ERROR IN STATE DEFINITION")
            sys.exit()

        return actions

    def get_legal_moves(self, state):  # TODO added in case not fully reversible, but maybe can remove
        # if travelling then it must be noop
        if state == 2:
            actions = [2]
        elif state == 1:
            actions = [0, 2]
        elif state == 0:
            actions = [1, 2]
        else:
            print("ERROR IN STATE DEFINITION")
            sys.exit()

        return actions

    def step(self, action):
        self._time += self.time_step

        # TODO check these conditionals are okay
        # if travelling state need to figure out how much time left and update this or so for legal actions
        if self._agent_state == 2:
            self._update_travel()
        if action == 0 or action == 1:
            self._travel_to_new_activity(action)

        # calc trace
        self._extent_trace()
        self._extent_trace_2()

        # finally calc the rest
        return (
            self._get_obs(),
            self._get_reward(),
            self._terminated(),
            False,
            self._get_info(),
        )

    def _get_reward(self, last=True):  # TODO updated reward to give per step? test if this is better or not
        if last:
            return (self._trace_scorer.score(
                trace=self._trace, obs_map=self._observation_space_mapping
            )
                    # - self._trace_scorer.score(
                # trace=self._last_trace, obs_map=self._observation_space_mapping)
            )
        else:
            return 0

    def _travel_to_new_activity(self, action):
        """Move state to traveling, change destination, calc travel time."""
        self._agent_state = 2
        self._destination = action
        self._remaining_travel_time = (
            self._travel_sim.sample(self.distance, self._time) - self.time_step
        )

    def _update_travel(self):
        if self._remaining_travel_time <= 0:  # arrive!
            self._agent_state = self._destination
            self._destination = None
        else:  # still traveling
            self._remaining_travel_time -= self.time_step

    def _extent_trace(self):
        self._last_trace = self._trace.copy()
        if (
            self._agent_state == self._trace[-1][0]
        ):  # no change, extend current activity
            self._trace[-1][1] += self.time_step
        elif self._agent_state == 2:  # new travel
            self._trace.append(
                [self._agent_state, self.time_step, self.distance]
            )
        else:  # new act
            self._trace.append([self._agent_state, self.time_step, 0])

    def _extent_trace_2(self):
        # TODO updated trace so it gives every timestep, think might make easier but lmk if this ruins the functionality
        curr_time = self._time
        curr_idx = int(self._time * self.steps / self.duration)

        if self._agent_state == 2:
            self._trace_2[curr_idx] = [self._agent_state, curr_time, self.distance]
        else:
            self._trace_2[curr_idx] = [self._agent_state, curr_time, 0]

    def _terminated(self):
        return self._time + self.time_step >= self.duration
