"""
Train (:mode:`tsax.training`)
===============================
"""
from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Union

from flax.training import train_state

from tsax.typing import KeyArray, DataT
from tsax.core import Model
from tsax.data import SeqData


__all__ = [
    "TrainState",
]


class TrainState(train_state.TrainState):
    split_fn: Callable[[KeyArray], Dict[str, KeyArray]]

    @staticmethod
    def create_for(key: KeyArray, model: Model, data: Union[SeqData[DataT], DataT]):
        """
        Create TrainState for Model & Data

        Parameters
        ----------
        key : KeyArray
            PRNG Key
        model : Model
            TSax model
        data : SeqData or DataT
            Input Data

        Returns
        -------
        state : TrainState
            Training State
        """
        if isinstance(data, SeqData):
            x, _ = data.ibatch(0)
        else:
            x = data

        key_p, key = model.split_key(key, train=False)
        key["params"] = key_p
        if isinstance(x, Union[Tuple,List]):
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
