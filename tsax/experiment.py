from __future__ import annotations
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from tqdm import tqdm
import wblog

from tsax.typing import Array, ArrayLike


logger = wblog.getLogger()


def train(model,
          rngkey: KeyArray,
          train_x,
          train_y,
          valid_x,
          valid_y,
          ephoch: int,
          batch_size: int,
          optimizer: optax.GradientTransformation) -> None:
    """
    Train Model

    Parameters
    ----------
    model
        Model
    rngkey : KeyArray
        PRNG Key
    epoch : int
        Training Epoch
    batch_size : int
        Batch Size
    """
    t0 = time.perf_counter()

    rngkey, key_use = jax.random.split(rngkey, 2)
    params = model.init(key_use, ...)
    opt_sate = optimizer.init(params)

    idx = jnp.arange(train_size)
    nbatch: int = train_size // batch_size
    logger.info("Start Training")
    for ep in range(epoch):
        t = time.perf_counter()
        rngkey, key_use = jax.random.split(rngkey, 2)
        idx = jax.random.permutation(key_use, idx)

        epoch_loss: float = 0
        for i in tqdm(range(nbatch), ascii=True, leave=False):
            bidx = idx.at[i:i+batch_size].get()
            rngkey, key_use = jax.random.split(rngkey, 2)
            loss, grad = train_fn(params,
                                  key_use, ...)

            epoch_loss += loss
            update, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, update)

        dt = (time.perf_counter() - t)
        logger.info("Train: Epoch %d, Loss: %.6f, Elapssed: %.3f sec",
                    ep, epoch_loss / (batch_size * nbatch), dt)

        if ep % valid_freq == 0:
            valid_loss = loss_fn(params, None, ...)
            logger.info("Valid: Epoch: %d, Loss: %.6f",
                        ep, valid_loss / valid_size)


    final_loss = loss_fn(params, None, ...)
    logger.info("Final Valid: Total Epoch %d, Loss: %.6f, Elapsed: %.3f sec",
                epoch, final_loss / valid_size, time.perf_counter() - t0)


def inference(model, rngkey: KeyArray) -> Array:
    pass
