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

        return MetricsSummary(
            N=runs.shape[0],
            mean=jnp.mean(runs, axis=0),
            mean_error=jnp.std(runs, axis=0, ddof=1) / jnp.sqrt(runs.shape[0]),
            unbiased_variance=jnp.var(runs, axis=0, ddof=1)
        )
