import numpy as np
from typing import Optional


class SimpleMATSimTraceScorer:
    def __init__(self):
        """
        MATSim scoring function for a single agent trace.

        Where a trace is a list of tuples, each tuple containing:
        - activity index
        - start time in hours
        - end time in hours
        - mode
        - distance

        Scoring is based on the following components for activities:
        - activity duration
        - waiting time
        - late arrival
        - early departure

        Scoring is based on the following components for travel:
        - travel duration
        - distance
        - mode constant
        """
        self.config = {
            "mUM": 1,
            "performing": 10,
            "waiting": -1,
            "lateArrival": -10,
            "earlyDeparture": -10,
            "acts": {
                "work": {
                    "typicalDuration": 8,
                    "openingTime": 6,
                    "closingTime": 20,
                    "latestStartTime": 9.5,
                    "earliestEndTime": 16,
                    "minimalDuration": 1,
                },
                "home": {"typicalDuration": 12, "minimalDuration": 1},
                "shop": {
                    "typicalDuration": 0.5,
                    "openingTime": 8,
                    "closingTime": 18,
                    "minimalDuration": 0.5,
                },
            },
            "modes": {
                "car": {
                    "constant": -1,
                    "dailyMonetaryConstant": 0,
                    "dailyUtilityConstant": 0,
                    "marginalUtilityOfDistance": 0,
                    "marginalUtilityOfTravelling": 0,
                    "monetaryDistanceRate": -0.0001,
                },
                "bus": {
                    "constant": -1,
                    "dailyMonetaryConstant": -5,
                    "dailyUtilityConstant": 0,
                    "marginalUtilityOfDistance": 0,
                    "marginalUtilityOfTravelling": -1,
                    "monetaryDistanceRate": 0,
                },
                "walk": {
                    "constant": 0,
                    "dailyMonetaryConstant": 0,
                    "dailyUtilityConstant": 0,
                    "marginalUtilityOfDistance": -0.001,
                    "marginalUtilityOfTravelling": 0,
                    "monetaryDistanceRate": 0,
                },
            },
        }

    def min(self, distance: Optional[float] = None):
        # assume 24 hour period
        # assume zero benefit from activities
        # assume 2 trips of 12 hours each
        mum = self.config.get("mUM", 1)
        if distance is None:
            distance = 100
        costs = []
        # find most expensive mode
        for _, params in self.config["modes"].items():
            cost = 0
            cost += params.get("constant", 0)
            cost += params.get("dailyMonetaryConstant", 0) * mum
            cost += params.get("dailyUtilityConstant", 0)
            cost += params.get("marginalUtilityOfDistance", 0) * distance
            cost += params.get("marginalUtilityOfTravelling", 0) * 12
            cost += params.get("monetaryDistanceRate", 0) * distance * mum
            costs.append(cost)
        return min(costs) * 2

    def max(self):
        # assume 24 hour period
        # assume zero cost from trips
        # assume max value of performing achieved without penalty
        return self.config["performing"] * 24

    def score(self, trace: list, obs_map: dict):
        score = 0.0
        modes = set()
        wrapped_trace = wrap_trace(trace)
        for idx, start, end, duration, distance in wrapped_trace:
            label = obs_map[idx]
            if label.startswith("act"):
                act = label.split(":")[1]
                score += self.score_act_duration(act, start, end, duration)
                score += self.waiting_score(act, start, end)
                score += self.late_arrival_score(act, start, end)
                score += self.early_departure_score(act, start, end)
            elif label.startswith("trip"):
                mode = label.split(":")[1]
                modes.add(mode)
                score += self.mode_constant(mode)
                score += self.score_travel_duration(mode, duration)
                score += self.score_distance(mode, distance)
        score += self.score_daily(modes)
        return score

    def score_daily(self, modes) -> float:
        return sum([self.score_day_mode_use(mode) for mode in modes])

    def score_day_mode_use(self, mode) -> float:
        return self.config["modes"][mode].get("dailyUtilityConstant", 0) + (
            self.config["modes"][mode].get("dailyMonetaryConstant", 0)
            * self.config.get("mUM", 1)
        )

    def mode_constant(self, mode) -> float:
        return self.config["modes"][mode].get("constant", 0)

    def score_travel_duration(self, mode, duration) -> float:
        return (
            self.config["modes"][mode].get("marginalUtilityOfTravelling", 0)
            * duration
        )

    def score_distance(self, mode, distance) -> float:
        mum = self.config.get("mUM", 1)
        dist_util = (
            self.config["modes"][mode].get("marginalUtilityOfDistance", 0)
            * distance
        )
        monetary = (
            self.config["modes"][mode].get("monetaryDistanceRate", 0) * distance
        )
        return dist_util + (monetary * mum)

    def score_act_duration(self, activity, start, end, duration) -> float:
        prio = 1
        performing = self.config["performing"]
        typical_dur = self.config["acts"][activity]["typicalDuration"]

        opening_time = self.config["acts"][activity].get("openingTime")
        if opening_time is not None and opening_time > start:
            duration -= opening_time - start

        closing_time = self.config["acts"][activity].get("closingTime")
        if closing_time is not None and closing_time < end:
            duration -= end - closing_time

        if duration < typical_dur / np.e:
            return (duration * np.e - typical_dur) * performing

        return (
            performing
            * typical_dur
            * (np.log(duration / typical_dur) + (1 / prio))
        )

    def waiting_score(self, activity, start, end) -> float:
        waiting = self.config.get("waiting")
        if not waiting:
            return 0.0
        opening_time = self.config["acts"][activity].get("openingTime")
        if opening_time is None:
            return 0.0
        if start < opening_time:
            return waiting * (opening_time - start)
        return 0.0

    def late_arrival_score(self, activity, start, end) -> float:
        if self.config["acts"][activity].get(
            "latestStartTime"
        ) is not None and self.config.get("lateArrival"):
            latest_start_time = self.config["acts"][activity]["latestStartTime"]
            if start > latest_start_time:
                return self.config["lateArrival"] * (start - latest_start_time)
        return 0.0

    def early_departure_score(self, activity, start, end) -> float:
        if self.config["acts"][activity].get(
            "earliestEndTime"
        ) is not None and self.config.get("earlyDeparture"):
            earliest_end_time = self.config["acts"][activity]["earliestEndTime"]
            if end < earliest_end_time:
                return self.config["earlyDeparture"] * (earliest_end_time - end)
        return 0.0


def wrap_trace(trace):
    time = 0
    wrapped = []
    for component, duration, distance in trace:
        wrapped.append((component, time, time + duration, duration, distance))
        time += duration
    if len(wrapped) < 2:
        return wrapped
    if wrapped[0][0] == wrapped[-1][0]:
        wrapped[-1] = (
            wrapped[-1][0],
            wrapped[-1][1],
            wrapped[0][2],
            wrapped[-1][3] + wrapped[0][3],
            wrapped[-1][4] + wrapped[0][4],
        )
        wrapped.pop(0)
    return wrapped
