class ScheduleExpectation:
    """
    Base class for schedule expectation given an input (assumed partially completed) trace.
    """

    def __call__(self, trace: list, kwargs) -> list:
        return []


class Miopic(ScheduleExpectation):

    def __call__(self, trace: list) -> list:
        return trace


class ContinueDay(ScheduleExpectation):

    def __init__(self, duration: int) -> None:
        """
        Simple continuation of the current or planned activity until end of day.

        Args:
            duration (int): duration of the day.
        """
        self.duration = duration

    def __call__(self, trace: list, next_act: int) -> list:
        time_remaining = self.duration - sum([d for _, d, _ in trace])
        if time_remaining <= 0:
            return trace
        if trace[-1][0] == next_act:
            new_trace = trace.copy()
            new_trace[-1][1] += time_remaining
            return new_trace
        new_trace = trace.copy()
        new_trace.append([next_act, time_remaining, 0])
        return new_trace


class ReturnDay(ScheduleExpectation):

    def __init__(
        self,
        duration: int,
        return_act: int,
        target_duration: float,
        trip_idx: int,
    ) -> None:
        """
        Return to a specified activity (such as home) before the end of the day if possible.

        todo: ignores current trip remaining time.

        Args:
            duration (int): duration of the day.
            return_act (int): activity to return to.
            target_duration (float): target (maximum) return activity duration.
            trip_idx (int): index of trip in the trace.
        """
        self.duration = duration
        self.return_act = return_act
        self.target_duration = target_duration
        self.trip_idx = trip_idx

    def __call__(
        self,
        trace: list,
        next_act: int,
        trip_duration: float,
        trip_distance: float,
    ) -> list:
        # todo: make this less bad
        # todo: regret this weird trace encoding - better as arrays of time steps?
        time_remaining = self.duration - sum([d for _, d, _ in trace])

        if time_remaining <= 0:  # finished
            return trace

        if trace[-1][0] == self.return_act:
            ## already at target act, use ContinueValue
            return ContinueDay(self.duration)(trace, next_act)

        surplus = time_remaining - (self.target_duration + trip_duration)
        if surplus >= 0:
            # enough time for target duration and trip
            new_trace = trace.copy()
            if new_trace[-1][0] == next_act:
                new_trace[-1][1] += surplus
            else:
                new_trace.append([next_act, surplus, 0])
            new_trace.append([self.trip_idx, trip_duration, trip_distance])
            new_trace.append([self.return_act, self.target_duration, 0])
            return new_trace
        if surplus == 0:
            # enough time for target duration and trip only?
            new_trace = trace.copy()
            new_trace.append([self.trip_idx, trip_duration, trip_distance])
            new_trace.append([self.return_act, self.target_duration, 0])
            return new_trace

        surplus = time_remaining - trip_duration
        if surplus > 0:
            # enough time for part of target
            new_trace = trace.copy()
            new_trace.append([self.trip_idx, trip_duration, trip_distance])
            new_trace.append([self.return_act, surplus, 0])
            return new_trace
        if surplus == 0:
            # enough time for trip only
            new_trace = trace.copy()
            new_trace.append([self.trip_idx, trip_duration, trip_distance])
            return new_trace
        return ContinueDay(self.duration)(trace, next_act)
