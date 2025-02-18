import torch
from zuko.flows.autoregressive import MAF
from zuko.flows import UnconditionalDistribution
from zuko.distributions import BoxUniform
from zuko.transforms import CircularShiftTransform, ComposedTransform, MonotonicRQSTransform
from functools import partial

class CustomCircularRQSTransform:
    def __init__(self,bound):
        self.bound = bound

    def __call__(self,*x):
        return ComposedTransform(
            CircularShiftTransform(bound=self.bound),
            MonotonicRQSTransform(*x, bound=self.bound),
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
            #torch.full((features,), -bound - 1e-5),
            #torch.full((features,), bound + 1e-5),
            torch.full((features,), -bound),
            torch.full((features,), bound),
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
            #univariate = MonotonicRQSTransform,
            shapes = [(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )

        self.base = UnconditionalDistribution(
            BoxUniform,
            torch.full((features,), -bound - 1e-5),
            torch.full((features,), bound + 1e-5),
            buffer=True,
        )
