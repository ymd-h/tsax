from __future__ import annotations
from typing import Dict, Tuple

import flax.linen as nn

from tsax.typing import KeyArray


class Model(nn.Module):
    @staticmethod
    def split_key(key: KeyArray, *,
                  train: bool = False) -> Tuple[KeyArray, Dict[str, KeyArray]]:
        """
        Split PRNG Key for this model

        Parameters
        ----------
        key : KeyArray
            Key will be split
        train : bool, optional
            Whether train or not

        Returns
        -------
        key : KeyArray
            New Key
        key_for_model : KeyArray
            Keys can be consumed by this model.
        """
        return key, {}

    def log_model(self) -> None:
        """
        Log Model Spec
        """
        pass
