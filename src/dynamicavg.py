import numpy as np

class DynamicAvg():
    def __init__(self):
        self.n = 0
        self.a = 0
        self.samples = []
    def update(self, v):
        alpha = 1/(self.n + 1)
        self.a = self.a * (1-alpha) + v *alpha
        self.n += 1
        self.samples.append(v)
    def get(self):
        assert len(self.samples) or np.abs(self.a - np.mean(self.samples)) < 1e-5, f'{self.a} != {np.mean(self.samples)}'
        return self.a

