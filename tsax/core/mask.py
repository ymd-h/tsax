from __future__ import annotations

import jax
import jax.numpy as jnp

from tsax.typing import Array


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
