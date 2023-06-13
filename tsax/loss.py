"""
TSax Loss (:mod:`tsax.loss`)
============================
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.tree_util import tree_flatten

from tsax.typing import Array, ArrayLike, DataT

__all__ = [
    "AE",
    "SE",
]


def _extract(true_y: DataT) -> Array:
    t, _ = tree_flatten(true_y)
    return t[0]


def AE(pred_y: ArrayLike, true_y: DataT) -> Array:
    """
    Absolute Error.

    Parameters
    ----------
    pred_y : ArrayLike
        Predicted
    true_y : DataT
        True

    Returns
    -------
    loss : Array
        Sum Absolute Error
    """
    t = _extract(true_y)
    assert pred_y.shape == t.shape, "BUG"

    return jnp.sum(jnp.abs(pred_y - t))


def SE(pred_y: ArrayLike, true_y: DataT) -> Array:
    """
    Squared Error.

    Parameters
    ----------
    pred_y : ArrayLike
        Predicted
    true_y : DataT
        True

    Returns
    -------
    loss : Array
        Sum Squared Error
    """
    t = _extract(true_y)
    assert pred_y.shape == t.shape, "BUG"

    return jnp.sum((pred_y - t) ** 2)
