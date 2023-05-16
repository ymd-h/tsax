"""
TSax Random (:mod:`tsax.random`)
================================
"""
from __future__ import annotations
import os

import jax
import wblog

from tsax.typing import KeyArray


logger = wblog.getLogger()


def initPRNGKey(seed: Optional[int] = None) -> KeyArray:
    """
    Initialize Pseudo Random Number Generator's Key

    Parameters
    ----------
    seed : int, optional
        Seed. If ``None`` (default), use ``os.urandom(4)`` as seed.

    Returns
    -------
    key : KeyArray
        PRNG Key
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    logger.info("Random Seed: %x", seed)

    key = jax.random.PRNGKey(seed)
    return key
