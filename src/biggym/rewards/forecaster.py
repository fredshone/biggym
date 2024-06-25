class ScheduleValue:

    def __call__(self, trace: list, kwargs) -> list:
        return []


class Miopic(ScheduleValue):
    
        def __call__(self, trace: list) -> list:
            return trace
        

class ContinueValue(ScheduleValue):

    def __init__(self, duration: int) -> None:
        self.duration = duration
    
    def __call__(self, trace: list, next_act: int) -> list:
        time_remaining = self.duration - sum([d for _, d, _ in trace])
        if time_remaining <= 0:
            return trace
        new_trace = trace.copy().append([next_act, time_remaining, 0])
        return new_trace
    

class SimplePlanValue(ScheduleValue):
         
        def __init__(self, duration: int, return_act: int, target_duration: float) -> None:
            self.duration = duration
            self.return_act = return_act
            self.target_duration = target_duration
        
        def __call__(self, trace: list, next_act: int, act_duration: float, trip_duration: float, trip_distance: float) -> list:
            time_remaining = self.duration - sum([d for _, d, _ in trace])
            
            if time_remaining <= 0:  # finished
                return trace

            if trace[-1][0] == self.return_act:
                ## already at target act, use ContinueValue
                return ContinueValue(self.duration)(trace, next_act)
            
            # enough time for target duration and trip plus
            surplus = time_remaining - self.target_duration - trip_duration
            if surplus > 0:
                new_trace = trace.copy().append([next_act, time_remaining, 0])
                return new_trace
            # enough time for target duration and trip only?
            # enough time for trip and some of target duration?
            # time enough for trip only?



            new_trace = trace.copy().append([next_act, time_remaining, 0])
            return new_trace