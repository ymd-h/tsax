"""
Stat (:mod:`tsax.stat`)
=======================
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode, field

from tsax.typing import ArrayLike


__all__ = [
    "MetricsSummary",
]


class MetricsSummary(PyTreeNode):
    N: int = field(pytree_node=False)
    mean: Array = field(pytree_node=True)
    mean_error: Array = field(pytree_node=True)
    unbiased_variance: Array = field(pytree_node=True)

    @staticmethod
    def create(runs: ArrayLike) -> MetricsSummary:
        """
        Create MetricsSummary from Multiple Runs

        Parameters
        ----------
        runs : ArrayLike
            Metrics of Multiple Runs. [# of runs, # of steps]
        """
        assert runs.shape[0] > 1, "BUG"

        N = runs.shape[0]
        V = jnp.var(runs, axis=0, ddof=1)
        return MetricsSummary(
            N=N,
            mean=jnp.mean(runs, axis=0),
            mean_error=jnp.sqrt(V/N),
            unbiased_variance=V
        )

    def add(other: MetricsSummary) -> MetricsSummary:
        """
        Add 2 MetricsSummary and Returns New One

        Parameters
        ----------
        other : MetricsSummary
            Other metrics

        Returns
        -------
        MetricsSummary
            Newly created summed MetricsSummay
        """
        N = self.N + other.N
        mean = (self.N * self.mean + other.N * other.mean) / N

        def f(n, m, uv):
            return n * (m**2 + ((n-1)/n * uv) ** 2)

        # Ref: https://stats.stackexchange.com/a/43183
        uV = (
            (
                f( self.N,  self.mean,  self.unbiased_variance) +
                f(other.N, other.mean, other.unbiased_variance)
            ) / N
            - (mean ** 2)
        ) * (N/(N-1))

        return MetricsSummary(
            N=N,
            mean=mean,
            mean_error=jnp.sqrt(uV/N),
            unbiased_variance=uV,
        )
