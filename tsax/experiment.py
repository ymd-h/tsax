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
    rngkey, key_use = jax.random.split(rngkey, 2)
    params = model.init(key_use,
                        )

    logger.info("Start Training")



def inference(model, rngkey: KeyArray) -> Array:
    pass
