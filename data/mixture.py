import random

class WeightedMixture:
    def __init__(self, sources, weights, seed=123):
        if len(sources) != len(weights):
            raise ValueError("sources and weights must have the same length")
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("invalid weights")
        self.sources = sources
        self.weights = [w / sum(weights) for w in weights]
        self.rng = random.Random(seed)

    def __iter__(self):
        its = [iter(x) for x in self.sources]
        while True:
            i = self.rng.choices(range(len(its)), weights=self.weights, k=1)[0]
            try:
                yield next(its[i])
            except StopIteration:
                its[i] = iter(self.sources[i])
                yield next(its[i])
