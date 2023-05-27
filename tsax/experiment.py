"""
Experiment (:mod:`tsax.experiment`)
===================================

Notes
-----
This module requires additional dependencies,
which can be installed by `pip install tsax[experiment]`
"""
from __future__ import annotations
from typing import Callable, Optional
import time

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import wblog

from tsax.typing import Array, ArrayLike, KeyArray, DataT
from tsax.data import SeqData


logger = wblog.getLogger()


class TrainState(flax.training.train_state.TrainState):
    split_fn: Callable[[KeyArray], Dict[str, KeyArray]]


def train(key: KeyArray,
          state: TrainState,
          train_data: SeqData[DataT],
          ephoch: int,
          batch_size: int,
          loss_fn: Callable[[DataT, DataT], Array],
          valid_data: Optional[SeqData[DataT]] = None,
          valid_freq: int = 10) -> TrainState:
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
    batch_size : int
        Batch Size
    loss_fn : callable
        Loss Function
    valid_data : SeqData, optional
        Validation Data
    valid_freq : int, optional
        Validation Frequency

    Returns
    -------
    state : TrainState
        Trained State
    """
    t0 = time.perf_counter()

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

        key, key_use = jax.random.split(rngkey, 2)
        train_data.shuffle(key_use)

        (state, key, epoch_loss), _ = train_data.scan(train_scan_fn, (state, key, 0))

        dt = (time.perf_counter() - t)
        logger.info("Train: Epoch %d, Loss: %.6f, Elapssed: %.3f sec",
                    ep, epoch_loss / train_size, dt)

        if (valid_data is not None) and (ep % valid_freq == 0):
            (_, key, valid_loss), _ = valid_data.scan(valid_scan_fn, (state, key, 0))
            logger.info("Valid: Epoch: %d, Loss: %.6f", ep, valid_loss / valid_size)


    if valid_data is not None:
        (_, key, final_loss), _ = valid_data.scan(valid_scan_fn, (state, key, 0))
        logger.info("Final Valid: Total Epoch %d, Loss: %.6f, Elapsed: %.3f sec",
                    epoch, final_loss / valid_size, time.perf_counter() - t0)

    return state


def inference(model, rngkey: KeyArray) -> Array:
    pass
