import pytest
from biggym.rewards import expectation as exp


def test_miopic():
    miopic = exp.Miopic()
    trace = [[1, 10, 0], [2, 10, 0]]
    assert miopic(trace) == trace


@pytest.mark.parametrize(
    "trace_in,next_act,expected",
    [
        [[[0, 6, 0]], 0, [[0, 24, 0]]],
        [[[0, 6, 0], [1, 0.5, 1]], 2, [[0, 6, 0], [1, 0.5, 1], [2, 17.5, 0]]],
        [[[0, 24, 0]], 0, [[0, 24, 0]]],
    ],
)
def test_continue_day(trace_in, next_act, expected):
    continue_day = exp.ContinueDay(24)
    trace = continue_day(trace_in, next_act)
    assert trace == expected


@pytest.mark.parametrize(
    "trace_in,next_act,expected",
    [
        [[[1, 8, 0]], 1, [[1, 24, 0]]],
        [
            [[1, 8, 0], [0, 0.5, 1]],
            2,
            [[1, 8, 0], [0, 0.5, 1], [2, 8.5, 0], [0, 1, 1], [1, 6, 0]],
        ],
        [
            [[1, 8, 0], [0, 1, 1], [2, 4, 0]],
            2,
            [[1, 8, 0], [0, 1, 1], [2, 8, 0], [0, 1, 1], [1, 6, 0]],
        ],
        [
            [[1, 8, 0], [0, 1, 1], [2, 8, 0]],
            2,
            [[1, 8, 0], [0, 1, 1], [2, 8, 0], [0, 1, 1], [1, 6, 0]],
        ],
        [
            [[1, 8, 0], [0, 1, 1], [2, 9, 0]],
            2,
            [[1, 8, 0], [0, 1, 1], [2, 9, 0], [0, 1, 1], [1, 5, 0]],
        ],
        [
            [[1, 8, 0], [0, 1, 1], [2, 14, 0]],
            2,
            [[1, 8, 0], [0, 1, 1], [2, 14, 0], [0, 1, 1]],
        ],
        [
            [[1, 8, 0], [0, 1, 1], [2, 14.5, 0]],
            2,
            [[1, 8, 0], [0, 1, 1], [2, 15, 0]],
        ],
    ],
)
def test_return_day(trace_in, next_act, expected):
    return_day = exp.ReturnDay(
        duration=24, return_act=1, target_duration=6, trip_idx=0
    )
    trace = return_day(trace_in, next_act, trip_duration=1, trip_distance=1)
    assert trace == expected
