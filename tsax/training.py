"""
Train (:mode:`tsax.training`)
===============================
"""
from __future__ import annotations
from typing import Callable, Dict

import flax

from tsax.typing import KeyArray
from tsax.core import Model
from tsax.data import SeqData

__all__ = [
    "TrainState",
]


class TrainState(flax.training.train_state.TrainState):
    split_fn: Callable[[KeyArray], Dict[str, KeyArray]]

    @staticmethod
    def create_for(key: KeyArray, model: Model, data: Union[SeqData[DataT], DataT]):
        if isinstance(data, SeqData):
            x, _ = data.ibatch(0)
        else:
            x = data

        if isinstance(x, [tuple, list]):
            params = model.init(key, *x)
            def apply_fn(variables, _x, *args, **kwargs):
                return model.apply(variables, *_x, *args, **kwargs)
        else:
            params = model.init(key, x)
            def apply_fn(variables, *args, **kwargs):
                return model.apply(variables, *args, **kwargs)

        return TrainState.create(
            apply_fn=apply_fn,
            params=params,
            split_fn=model.split_fn,
        )
