from biggym.scorers import SimpleMATSimTraceScorer


def test_wrap():
    trace = [
        ("home", 6, 0),
        ("trip", 1, 0.3),
        ("work", 8, 0),
        ("trip", 1, 0.3),
        ("home", 8, 0),
    ]
    scorer = SimpleMATSimTraceScorer()
    wrapped = scorer.wrap_trace(trace)
    assert wrapped == [
        ("trip", 6, 7, 1, 0.3),
        ("work", 7, 15, 8, 0),
        ("trip", 15, 16, 1, 0.3),
        ("home", 16, 6, 14, 0),
    ]
