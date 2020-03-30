from abc import ABC, abstractmethod

class Solver(ABC):

    def __init__(self, manifold, cost, *args, **kwargs):
        self.manifold = manifold
        self.cost = cost
        super().__init__()

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass

    @abstractmethod
    def _step(self, xx, *args, **kwargs):
        pass
