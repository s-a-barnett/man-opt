from abc import ABC, abstractmethod

class Manifold(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _reprToPoint(self, xx):
        # takes the representation of a point on the manifold as a tuple xx and
        # returns the point being represented
        pass

    @abstractmethod
    def _inner(self, xx, yy):
        pass

    @abstractmethod
    def _norm(self, xx):
        pass

    @abstractmethod
    def _retract(self, xx, uu):
        # retraction R_xx (uu)
        pass

    @abstractmethod
    def _project(self, xx, zz):
        # Projection of zz to tangent space at xx
        pass

    @abstractmethod
    def _riemannianGradient(self, euclidenGradient):
        pass

    @abstractmethod
    def _riemannianHessian(self, euclideanHessian):
        pass

    @abstractmethod
    def _randomPoint(self):
        pass
