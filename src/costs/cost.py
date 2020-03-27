from abc import ABC, abstractmethod

class Cost(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _eval(self, xx):
        pass

    @abstractmethod
    def _euclideanGradient(self, xx):
        pass

    @abstractmethod
    def _euclideanHessian(self, xx, hh):
        pass
