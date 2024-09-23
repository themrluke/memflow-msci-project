r"""Spline flows."""

import torch

from math import pi
from torch.distributions import Transform

# isort: local
from zuko.flows.autoregressive import MAF
from zuko.flows.core import Unconditional
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal
from zuko.transforms import CircularShiftTransform, ComposedTransform, MonotonicRQSTransform


def get_customTransform(bound = 1.1):

    def RQSTransform_custom(*phi) -> Transform:
        r"""Creates a custom rational-quadratic spline (RQS) transformation."""
    
        return ComposedTransform(
            MonotonicRQSTransform(*phi, bound=bound),
        )

    return RQSTransform_custom


def RQSTransform_custom_2(*phi, bound=1.1) -> Transform:
    r"""Creates a custom rational-quadratic spline (RQS) transformation."""

    return ComposedTransform(
        MonotonicRQSTransform(*phi, bound=bound),
    )


class Custom_spline_flow_ps(MAF):
    r"""Creates a neural circular spline flow (NCSF).

    Circular spline transformations are obtained by composing circular domain shifts
    with regular spline transformations. Features are assumed to lie in the half-open
    interval :math:`[-\pi, \pi[`.

    See also:
        :class:`zuko.transforms.CircularShiftTransform`

    References:
        | Normalizing Flows on Tori and Spheres (Rezende et al., 2020)
        | https://arxiv.org/abs/2002.02428

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        bound: float = 1.1,
        mean_gaussian: float = 0.0,
        std_gaussian: float = 0.35,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=get_customTransform(bound=bound),
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )

        self.base = Unconditional(
            DiagNormal,
            mean_gaussian * torch.ones((features,)),
            std_gaussian * torch.ones((features,)),
            buffer=True,
        )

