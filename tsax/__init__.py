"""
TSax (:mod:`tsax`)
==================

Time Series Forecasting on JAX/Flax


See Also
--------
Informer
Autoformer


Notes
-----
TSax has 3 different level Usage.

  1. Command Line Interface. See ``tsax.optional.cli``
  2. Predefined Experiment API. See ``tsax.optional.experiment``
  3. Use only Predefined Models. See ``tsax.model``
"""

__all__ = [
    "Informer",
    "Autoformer",
]

from . import (
    typing,
    logging,
    loss,
    random,
    stat,
    version,
    core,
)


from .model.informer import Informer
from .model.autoformer import Autoformer
