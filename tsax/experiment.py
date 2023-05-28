"""
Experiment (:mod:`tsax.experiment`)
===================================

Notes
-----
This module requires additional dependencies,
which can be installed by `pip install tsax[experiment]`
"""
from __future__ import annotations
from datetime import datetime
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import time

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state, orbax_utils
from flax.struct import field
import optax
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    Checkpointer,
    PyTreeCheckpointHandler,
)
from tqdm import tqdm
import wblog

from tsax.typing import Array, ArrayLike, KeyArray, DataT
from tsax.core import Model
from tsax.data import SeqData, ensure_BatchSeqShape

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
        model.log_model()

        if isinstance(data, SeqData):
            x, _ = data.ibatch(0)
        else:
            x = data

        logger.info("Create TrainState for Shape: %s", x.shape)

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
          checkpoint_metadata: Optional[Dict[str, Any]] = None) -> Tuple[TrainState,
                                                                         str]:
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
    directory : str
        Checkpoint Directory
    """
    t0 = time.perf_counter()

    directory = os.path.join(checkpoint_directory,
                             datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger.info("Checkpoint Directory: %s", directory)
    os.makedirs(directory, exist_ok=True)
    if checkpoint_options is None:
        # To save metrics, add (pseudo) best_fn.
        checkpoint_options = CheckpointManagerOptions(
            best_fn=lambda metrics: metrics.get("train_loss", 1e+10),
            best_mode="max",
        )

    ckpt = CheckpointManager(
        directory,
        Checkpointer(PyTreeCheckpointHandler()),
        options=checkpoint_options,
        metadata=checkpoint_metadata
    )

    logger.info("Epoch: %d", epoch)
    if valid_data is not None:
        logger.info("Valid Freq: %d", valid_freq)

    logger.info("Train Data: Batch Size: %d, # of Batch: %d",
                train_data.batch_size, train_data.nbatch)
    logger.info("Valid Data: Batch Size: %d, # of Batch: %d",
                valid_data.batch_size, valid_data.nbatch)

    train_fn = jax.value_and_grad(
        lambda p, k, x, y: loss_fn(state.apply_fn(p, x, train=True, rngs=k), y)
    )
    valid_fn = jax.jit(
        lambda p, k, x, y: loss_fn(state.apply_fn(p, x, train=False, rngs=k), y)
    )

    # @jax.jit # JIT compile is too slow
    def train_step_fn(s, k, l, x, y):
        k, k_use = s.split_fn(k, train=True)
        loss, grad = train_fn(s.params, k_use, x, y)

        s = s.apply_gradients(grads=grad)

        return s, k, l+loss

    # @jax.jit # JIT compile is too slow.
    def valid_step_fn(s, k, l, x, y):
        k, k_use = s.split_fn(k, train=False)
        loss = valid_fn(s.params, k_use, x, y)

        return k, l+loss

    train_size: int = train_data.batch_size * train_data.nbatch
    valid_size: int = valid_data.batch_size * valid_data.nbatch

    logger.info("Start Training")
    for ep in range(epoch):
        t = time.perf_counter()

        key, key_use = jax.random.split(key, 2)
        train_data.shuffle(key_use)

        epoch_loss = jnp.zeros((1,))
        for i in tqdm(range(train_data.nbatch),
                      ascii=True, leave=False, desc=f"Train: Epoch: {ep}"):
            x, y = train_data.ibatch(i)
            state, key, epoch_loss = train_step_fn(state, key, epoch_loss, x, y)

        dt = (time.perf_counter() - t)
        logger.info("Train: Epoch: %d, Loss: %.6f, Elapssed: %.3f sec",
                    ep, float(epoch_loss) / train_size, dt)

        save_args = orbax_utils.save_args_from_target(state.params)
        ckpt.save(ep,
                  state.params,
                  save_kwargs={"save_args": save_args},
                  metrics={"train_loss": epoch_loss / train_size},
                  force=(ep == epoch -1))

        if (valid_data is not None) and (ep % valid_freq == 0):
            valid_loss = jnp.zeros((1,))
            for i in tqdm(range(valid_data.nbatch),
                          ascii=True, leave=False, desc=f"Valid: Epoch: {ep}"):
                x, y = valid_data.ibatch(i)
                key, valid_loss = valid_step_fn(state, key, valid_loss, x, y)

            logger.info("Valid: Epoch: %d, Loss: %.6f",
                        ep, float(valid_loss) / valid_size)


    if valid_data is not None:
        final_loss = jnp.zeros((1,))
        for i in tqdm(range(valid_data.nbatch),
                      ascii=True, leave=False,
                      desc=f"Final Valid: Total Epoch: {epoch}"):
            x, y = valid_data.ibatch(i)
            key, final_loss = valid_step_fn(state, key, final_loss, x, y)

        logger.info("Final Valid: Total Epoch %d, Loss: %.6f, Elapsed: %.3f sec",
                    epoch, float(final_loss) / valid_size, time.perf_counter() - t0)

    return state, directory


def load(state: TrainState,
         checkpoint_directory: str,
         which: Union[int,
                      Literal["latest"],
                      Literal["best"]] = "latest",
         best_fn: Optional[Callable[[Any], float]] = None) -> TrainState:
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
    best_fn : callable, optional
        Function Determine Best Step

    Returns
    -------
    state : TrainState
        State with loaded Checkpoint
    """
    if best_fn is None:
        best_fn = lambda metrics: metrics.get("train_loss", 1e+10)

    ckpt = CheckpointManager(checkpoint_directory,
                             Checkpointer(PyTreeCheckpointHandler()),
                             CheckpointManagerOptions(best_fn=best_fn,
                                                      best_mode="min",
                                                      create=False))
    logger.debug("Stored Checkpoints: %s", ckpt.all_steps())
    logger.debug([c.metrics for c in ckpt._checkpoints])

    if which == "best":
        logger.info("Use best step")
        which = ckpt.best_step()
        if which is None:
            logger.warning("No metrics are saved. Use latest step")
            which = "latest"

    if which == "latest":
        logger.info("Use latest step")
        which = ckpt.latest_step()

    if which is None:
        logger.warning("No Checkpoints are saved. Return without load.")
        return state

    logger.info("Load %d step", which)
    restore_args = orbax_utils.restore_args_from_target(state.params)
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
