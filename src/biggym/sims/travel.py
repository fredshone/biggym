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

    def __init__(self, speed=30, peaks=None, constant=0) -> None:
        self.speed = speed  # km/h
        self.constant = constant  # fixed delay in h
        if peaks is not None:
            self.peaks = peaks
        else:
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
        return self.constant + (distance / self.speed) + self._delay(time)
