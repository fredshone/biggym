import pytest
from biggym.sims import RandomTravelSim


@pytest.mark.parametrize(
    "time, expected",
    [(0, 0), (7, 0), (8, 0.5), (9, 1), (10, 0.5), (11, 0), (24, 0)],
)
def test_delay(time, expected):
    sim = RandomTravelSim(speed=30, peaks=[(9, 4, 1)])
    delay = sim._delay(time)
    assert delay == expected


@pytest.mark.parametrize(
    "time, expected",
    [(0, 1), (7, 1), (8, 1.5), (9, 2), (10, 1.5), (11, 1), (24, 1)],
)
def test_sample(time, expected):
    sim = RandomTravelSim(speed=30, peaks=[(9, 4, 1)])
    sample = sim.sample(distance=30, time=time)
    assert sample == expected
