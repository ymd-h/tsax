"""
Experiment (:mod:`tsax.experiment`)
===================================

Notes
-----
This module requires additional dependencies,
which can be installed by `pip install tsax[experiment]`
"""
from __future__ import annotations
import os
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import time

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
from flax.struct import field
import optax
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    Checkpointer,
    PyTreeCheckpointHandler,
)
import wblog

from tsax.typing import Array, ArrayLike, KeyArray, DataT
from tsax.core import Model
from tsax.data import SeqData

__all__ = [
    "TrainState",
    "train",
    "load",
    "predict",
]


logger = wblog.getLogger()


class TrainState(train_state.TrainState):
    split_fn: Callable[[KeyArray], Dict[str, KeyArray]] = field(pytree_node=False)

    @staticmethod
    def create_for(key: KeyArray,
                   model: Model,
                   data: Union[SeqData[DataT], DataT],
                   tx: optax.GradientTransformation) -> TrainState:
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
        tx : optax.GradientTransformation
            Optax Optimizer

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
            tx=tx,
            split_fn=model.split_key,
        )


def train(key: KeyArray,
          state: TrainState,
          train_data: SeqData[DataT],
          epoch: int,
          loss_fn: Callable[[DataT, DataT], Array],
          valid_data: Optional[SeqData[DataT]] = None,
          valid_freq: int = 10,
          checkpoint_directory: str = "./tsax-ckpt",
          checkpoint_options: Optional[CheckpointManagerOptions] = None,
          checkpoint_metadata: Optional[Dict[str, Any]] = None) -> TrainState:
    """
    Train Model

    Parameters
    ----------
    key : KeyArray
        PRNG Key
    state : TrainState
        Training State
    train_data : SeqData
        Training Data
    epoch : int
        Training Epoch
    loss_fn : callable
        Loss Function
    valid_data : SeqData, optional
        Validation Data
    valid_freq : int, optional
        Validation Frequency
    checkpoint_directory : str, optional
        Directory for CheckpointManager
    checkpoint_options : CheckpointManagerOptions, optional
        Options for CheckpointManager
    checkpoint_metadata : dict, optional
        Metadata for CheckpointManager

    Returns
    -------
    state : TrainState
        Trained State
    """
    t0 = time.perf_counter()

    logger.info("Checkpoint Directory: %s", checkpoint_directory)
    os.makedirs(checkpoint_directory, exist_ok=True)
    ckpt = CheckpointManager(
        checkpoint_directory,
        Checkpointer(PyTreeCheckpointHandler()),
        options=checkpoint_options,
        metadata=checkpoint_metadata
    )

    train_fn = jax.value_and_grad(
        lambda p, k, x, y: loss_fn(state.apply_fn(p, x, train=True, rngs=k), y)
    )
    valid_fn = jax.jit(
        lambda p, k, x, y: loss_fn(state.apply_fn(p, x, train=False, rngs=k), y)
    )

    def train_scan_fn(skl, x, y):
        s, k, l = skl

        k, k_use = s.split_fn(k, train=True)
        loss, grad = train_fn(s.params, k_use, x, y)

        l += l
        s = s.apply_gradients(grads=grad)

        return (s, k, l), None

    def valid_scan_fn(skl, x, y):
        s, k, l = skl

        k, k_use = s.split_fn(k, train=False)
        l += valid_fn(s.params, k_use, x, y)

        return (s, k , l), None

    train_size: int = train_data.batch_size * train_data.nbatch
    valid_size: int = valid_data.batch_size * valid_data.nbatch

    logger.info("Start Training")
    for ep in range(epoch):
        t = time.perf_counter()

        key, key_use = jax.random.split(key, 2)
        train_data.shuffle(key_use)

        (state, key, epoch_loss), _ = train_data.scan(train_scan_fn, (state, key, 0))

        dt = (time.perf_counter() - t)
        logger.info("Train: Epoch %d, Loss: %.6f, Elapssed: %.3f sec",
                    ep, epoch_loss / train_size, dt)

        save_args = flax.training.orbax_utils.save_args_from_target(s.params)
        ckpt.save(ep,
                  s.params,
                  save_kwargs={"save_args": save_args},
                  metrics={"train_loss": epoch_loss / train_size})

        if (valid_data is not None) and (ep % valid_freq == 0):
            (_, key, valid_loss), _ = valid_data.scan(valid_scan_fn, (state, key, 0))
            logger.info("Valid: Epoch: %d, Loss: %.6f", ep, valid_loss / valid_size)


    if valid_data is not None:
        (_, key, final_loss), _ = valid_data.scan(valid_scan_fn, (state, key, 0))
        logger.info("Final Valid: Total Epoch %d, Loss: %.6f, Elapsed: %.3f sec",
                    epoch, final_loss / valid_size, time.perf_counter() - t0)

    save_args = flax.training.orbax_utils.save_args_from_target(s.params)
    ckpt.save(epoch,
              s.params,
              save_kwargs={"save_args": save_args},
              metrics={"train_loss": epoch_loss / train_size},
              force=True)
    return state


def load(state: TrainState,
         checkpoint_directory: str,
         which: Union[int,
                      Literal["latest"],
                      Literal["best"]] = "latest") -> TrainState:
    """
    Load Checkpoint

    Parameters
    ----------
    state : TrainState
        State to be set
    checkpoint_directory : str
        Checkpoint Directory
    which : int, "latest", "best", optional
        Checkpoiint Step.

    Returns
    -------
    state : TrainState
        State with loaded Checkpoint
    """
    ckpt = CheckpointManager(checkpoint_directory,
                             Checkpointer(PyTreeCheckpointHandler()))
    if which == "latest":
        which = ckpt.latest_step()
    elif which == "best":
        which = ckpt.best_step()

    restore_args = jax.training.orbax_utils.restore_args_from_target(state.params)
    state.params = ckpt.restore(which,
                                items=state.params,
                                restore_kwargs={"restore_args": restore_args})

    return state

def predict(key: KeyArray,
            state: TrainState,
            data: Union[DataT, SeqData[DataT]]) -> Array:
    """
    Predict with Model

    Parameters
    ----------
    key : KeyArray
        PRNG Key
    state : TrainState
        Trained State
    data : DataT
        Predict Data. [L, d]

    Returns
    -------
    pred : Array
        Predicted
    """
    @jax.jit
    def pred_fn(k, x):
        _, k = state.split_fn(k, train=False)
        return state.apply_fn(state.params, x, train=False, rngs=k)


    if isinstance(data, SeqData):
        idx = jnp.arange(data.nbatch)
        key = jax.random.split(key, data.nbath)

        return jax.vmap(lambda i, k: pred_fn(k, data._vxget(i)))(idx, key)


    data = ensure_BatchSeqShape(data)
    pred = pred_fn(state.params, key, data)

    return pred
