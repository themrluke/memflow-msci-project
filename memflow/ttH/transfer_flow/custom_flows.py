import math
import torch
from torch.distributions import Distribution, Transform, constraints
from zuko.flows.autoregressive import MAF
from zuko.flows import UnconditionalDistribution
from zuko.distributions import BoxUniform
from zuko.transforms import CircularShiftTransform, ComposedTransform, MonotonicRQSTransform
from functools import partial

class PSTransform(Transform):
    r"""
        Creates transformation from [-1,1] to [0,1]
        f(x) = (x+1)/2
    """
    domain = constraints.interval(-1,1)
    codomain = constraints.interval(0,1)
    bijective = True
    sign = +1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def _call(self,x):
        #print ('x',x.min(),x.max(),((x+1)/2).min(),((x+1)/2).max())
        return (x+1)/2

    def _inverse(self,y):
        return 2*y-1

    def log_abs_det_jacobian(self,x, y):
        return torch.ones_like(x) * math.log(1./2)

class CustomCircularRQSTransform:
    def __init__(self,bound):
        self.bound = bound

    def __call__(self,*x):
        return ComposedTransform(
            CircularShiftTransform(bound=self.bound),
            MonotonicRQSTransform(*x, bound=self.bound),
        )

class PSCustomRQSTransform:
    def __call__(self,*x):
        return ComposedTransform(
            PSTransform(),
            MonotonicRQSTransform(*x, bound=1.0),
        )

class CustomMonotonicRQSTransform:
    def __init__(self,bound):
        self.bound = bound

    def __call__(self,*x):
        return MonotonicRQSTransform(*x, bound=self.bound)


class UniformNCSF(MAF):
    def __init__(
        self,
        features: int,
        context: int,
        bins: int,
        bound: int,
        **kwargs,
    ):
        super().__init__(
            features = features,
            context = context,
            univariate = CustomCircularRQSTransform(bound),
            shapes = [(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )

        self.base = UnconditionalDistribution(
            BoxUniform,
            torch.full((features,), -bound - 1e-5),
            torch.full((features,), bound + 1e-5),
            buffer=True,
        )

class UniformNSF(MAF):
    def __init__(
        self,
        features: int,
        context: int,
        bins: int,
        bound: int,
        **kwargs,
    ):
        super().__init__(
            features = features,
            context = context,
            univariate = CustomMonotonicRQSTransform(bound),
            shapes = [(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )

        self.base = UnconditionalDistribution(
            BoxUniform,
            torch.full((features,), -bound),
            torch.full((features,), bound),
            buffer=True,
        )

class PSNSF(MAF):
    def __init__(
        self,
        features: int,
        context: int,
        bins: int,
        **kwargs,
    ):
        super().__init__(
            features = features,
            context = context,
            univariate = PSCustomRQSTransform(),
            shapes = [(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )

        self.base = UnconditionalDistribution(
            BoxUniform,
            torch.full((features,), 0.0),
            torch.full((features,), 1.0),
            buffer=True,
        )
