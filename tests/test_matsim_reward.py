import pytest
from biggym.rewards import SimpleMATSimTraceScorer
from biggym.rewards.matsim import wrap_trace


def test_wrap_empty():
    trace = []
    wrapped = wrap_trace(trace)
    assert wrapped == []


def test_wrap_single():
    trace = [("home", 6, 0)]
    wrapped = wrap_trace(trace)
    assert wrapped == [("home", 0, 6, 6, 0)]


def test_wrap():
    trace = [
        ("home", 6, 0),
        ("trip", 1, 0.3),
        ("work", 8, 0),
        ("trip", 1, 0.3),
        ("home", 8, 0),
    ]
    wrapped = wrap_trace(trace)
    assert wrapped == [
        ("trip", 6, 7, 1, 0.3),
        ("work", 7, 15, 8, 0),
        ("trip", 15, 16, 1, 0.3),
        ("home", 16, 6, 14, 0),
    ]


@pytest.mark.parametrize(
    "trace",
    [
        [(0, 6, 0)],
        [(0, 6, 0), (0, 6, 0)],
        [(0, 6, 0), (2, 1, 0.3), (1, 8, 0), (2, 1, 0.3), (0, 8, 0)],
        [
            (0, 6, 0),
            (2, 1, 0.3),
            (1, 8, 0),
            (2, 1, 0.3),
            (0, 8, 0),
            (2, 1, 0.3),
        ],
        [
            (0, 6, 0),
            (2, 1, 0.3),
            (1, 8, 0),
            (2, 1, 0.3),
            (0, 8, 0),
            (2, 1, 0.3),
            (1, 8, 0),
        ],
    ],
)
def test_score(trace):
    mapper = {0: "act:home", 1: "act:work", 2: "trip:car"}
    scorer = SimpleMATSimTraceScorer()
    score = scorer.score(trace, mapper)
    assert isinstance(score, float)


def test_min():
    scorer = SimpleMATSimTraceScorer()
    score = scorer.min_score()
    assert score < 0.0


def test_max():
    scorer = SimpleMATSimTraceScorer()
    score = scorer.max_score()
    assert score > 0.0
