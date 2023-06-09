from __future__ import annotations
from typing import cast

import jax
import jax.numpy as jnp

from tsax.typing import Array

__all__ = [
    "SubsequentMask",
]


def SubsequentMask(L: int) -> Array:
    """
    Generate Subsequent Mask for Attention

    Parameters
    ----------
    L : int
       Length of Sequence

    Returns
    -------
    mask : Array
        Subsequent Mask. [L, L]
    """
    return cast(Array, jnp.tril(jnp.ones((L, L), dtype=int)))
