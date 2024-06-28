from biggym.rewards.expectation import Miopic


class DiffScore:
    def __init__(self, scorer):
        self.scorer = scorer
        self.last = scorer.min_score()

    def reset(self):
        self.last = self.scorer.min_score()

    def __call__(self, **kwargs):
        score = self.scorer(**kwargs)
        diff = score - self.last
        self.last = score
        return diff


class NormalisedScorer:
    def __init__(self, scorer, expectation=None):
        self.scorer = scorer
        if expectation is None:
            expectation = Miopic()
        self.expectation = expectation
        self.min = scorer.min_score()
        self.max = scorer.max_score()

    def __call__(self, trace, obs_map, **kwargs):
        trace = self.expectation(trace, **kwargs)
        score = self.scorer(trace, obs_map, **kwargs)
        return (score - self.min) / (self.max - self.min)

    def reset(self):
        self.scorer.reset()

    def min_score(self):
        return 0.0

    def max_score(self):
        return 1.0
