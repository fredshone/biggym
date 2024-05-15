import gymnasium as gym
from gymnasium import spaces

from biggym.rewards import SimpleMATSimTraceScorer
from biggym.sims import RandomTravelSim


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

        self.observation_space = spaces.Dict(
            {
                "current_state": spaces.Discrete(
                    3
                ),  # at home or at work or traveling
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
        self._agent_state = self.initial
        self._time = 0
        self._trace = [
            [self._agent_state, self.time_step, 0]
        ]  # [[label, duration, distance],]
        self._destination = None
        self._remaining_travel_time = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        self._time += self.time_step

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
            self._extent_trace()
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
        return (
            self._get_obs(),
            self._reward(),
            self._terminated(),
            False,
            self._get_info(),
        )

    def _reward(self):
        return self._trace_scorer.score(
            trace=self._trace, act_map=self._observation_space_mapping
        )

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

    def _terminated(self):
        return self._time + self.time_step >= self.duration
