"""
Example 01: Experiment
======================

Require Optional Dependencies (`pip install tsax[experiment]`)
* optax: https://optax.readthedocs.io/
* orbax: https://github.com/google/orbax/tree/main

Require Additional Dependencies (`pip install matplotlib`)
* matplotlib: https://matplotlib.org/
"""
import argparse
from logging import getLogger, INFO, DEBUG
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from tsax.random import initPRNGKey
from tsax.data import SeqData
from tsax.loss import SE
from tsax.optional.experiment import train, predict, load, TrainState
from tsax.logging import enable_logging
from tsax import Informer, Autoformer


logger = getLogger("tsax.example.example01")

def example01(L: int,
              stride: int,
              xL: int,
              yL: int,
              model: Literal["informer", "autoformer"],
              c: int,
              nE: int,
              nD: int,
              nH: int,
              dm: int,
              dff: int,
              Pdrop: float,
              lr: float,
              batch: int,
              epoch: int,
              seed: int,
              checkpoint_directory: str,
              log_level: int) -> None:
    """
    Example 01

    Parameters
    ----------
    L : int
        Length of Data
    stride : int
        Stride for Sequence Division
    xL : int
        Lookback Horizon
    yL : int
        Prediction Horizon
    model : str
        Model
    c : int
        Hyperparameter: Sampling Factor
    nE : int
        Number of Encoder Stack
    nD : int
        Number of Decoder Stack
    dm : int
        Internal Model Dimension
    dff : int
        Hidden Units at Positional FeedFoward Layer
    Pdrop : float
        Dropout Rate
    lr : int
        Learning Rate of Optimizer
    batch : int
        Batch Size
    epoch : int
        Training Epoch
    seed : int
        Random Seed
    checkpoint_directory : str
        Checkpoint Directory
    log_level : int
        Logging Level
    """
    enable_logging(log_level)
    key = initPRNGKey(seed)

    d: int = 2
    logger.info("Generated Data: (%d, %d), stride: %d", L, d, stride)

    seq = (jnp.zeros((L, d))
           .at[:,0].add(jnp.sin(0.07 * jnp.pi * jnp.arange(L)))
           .at[:,1].add(jnp.cos(0.03 * jnp.pi * jnp.arange(L)))
           .at[:,0].add(jnp.sin(-0.22 * jnp.pi * jnp.arange(L)))
           .at[:,1].add(jnp.sin(0.013 * jnp.pi * (jnp.arange(L) ** 2))))
    train_size = int(seq.shape[0] * 0.8)

    fig, axes = plt.subplots(nrows=2, figsize=(10, 5))
    for i, a in enumerate(axes):
        a.plot(jnp.arange(train_size),
               seq.at[:train_size, i].get(),
               marker=".", linestyle=":", label="train")
        a.plot(jnp.arange(train_size, seq.shape[0]),
               seq.at[train_size:, i].get(),
               marker=".", linestyle=":", label="valid")
        a.legend()
    plt.title("Original Data")
    plt.savefig("01-experiment-data.png")
    plt.close()

    train_seq = SeqData(seq.at[:train_size].get(),
                        xLen=xL, yLen=yL, batch_size=batch, stride=stride)
    valid_seq = SeqData(seq.at[train_size:].get(),
                        xLen=xL, yLen=yL, batch_size=batch, stride=stride)

    if model == "informer":
        m = Informer(c=c, d=d, I=xL, O=yL, Ltoken=xL, dm=dm,
                     nE=nE, nD=nD, nH=nH, dff=dff, Pdrop=Pdrop)
    elif model == "autoformer":
        m = Autoformer(c=c, d=d, I=xL, O=yL,
                       nE=nE, nD=nD, nH=nH, dff=dff, Pdrop=Pdrop)

    logger.info("Adam(lr=%f)", lr)
    tx = optax.adam(lr)

    key, key_use = jax.random.split(key, 2)
    state = TrainState.create_for(key_use, m, train_seq, tx)

    key, key_use = jax.random.split(key, 2)
    state, ckpt = train(key_use, state, train_seq, epoch, SE, valid_seq,
                        checkpoint_directory=checkpoint_directory)

    state = load(state, ckpt, "best")

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    for j, (x, y, t) in enumerate([[seq.at[  :xL   ,:].get(),
                                    seq.at[xL:xL+yL,:].get(),
                                    "train"],
                                   [seq.at[-xL-yL:-yL,:].get(),
                                    seq.at[   -yL:   ,:].get(),
                                    "valid"]]):
        key, key_use = jax.random.split(key, 2)
        pred = predict(key_use, state, x)

        for i in range(2):
            axes[j][i].plot(y   .at[  :,i].get(),
                            marker=".", linestyle=":", label="true")
            axes[j][i].plot(pred.at[0,:,i].get(),
                            marker=".", linestyle=":", label="pred")
            axes[j][i].legend()
            axes[j][i].set_title(f"{t} / dimension[{i}]")

    plt.savefig("01-experiment.png")
    plt.close()
    logger.info("Draw: 01-experiment.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser("example-01")
    p.add_argument("--debug", action="store_true", help="Enable Debug")

    d = p.add_argument_group("Data")
    d.add_argument("--L", type=int, default=3000, help="Data Length")
    d.add_argument("--stride", type=int, default=1, help="Stride for Sequence")

    m = p.add_argument_group("Model")
    m.add_argument("--model", choices=["informer", "autoformer"],
                   help="Model", default="autoformer")
    m.add_argument("--xL", type=int, default=20, help="X Length")
    m.add_argument("--yL", type=int, default=10, help="Y Length")
    m.add_argument("--c", type=int, default=5, help="Hyperparameter: Sampling Factor")
    m.add_argument("--nE", type=int, default=3,
                   help="Number of Layers at Encoder Stack")
    m.add_argument("--nD", type=int, default=2,
                   help="Number of Layers at Decoder Stack")
    m.add_argument("--nH", type=int, default=8, help="Number of Multi Head")
    m.add_argument("--dm", type=int, default=64, help="Number of Model Dimension")
    m.add_argument("--dff", type=int, default=64,
                   help="Number of Units at Hidden Layer in Feed Forward")
    m.add_argument("--drop-rate", type=float, default=0.1,
                   help="Rate of Dropout for Training")

    e = p.add_argument_group("Experiment")
    e.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    e.add_argument("--batch", type=int, default=32, help="Batch Size")
    e.add_argument("--epoch", type=int, default=10, help="Number of Epoch")
    e.add_argument("--seed", type=int, default=None, help="Random Seed")
    e.add_argument("--checkpoint-directory", default="./tsax-ckpt",
                   help="Checkpoint Directory")

    args = p.parse_args()
    example01(L=args.L,
              stride=args.stride,
              xL=args.xL,
              yL=args.yL,
              model=args.model,
              c=args.c,
              nE=args.nE,
              nD=args.nD,
              nH=args.nH,
              dm=args.dm,
              dff=args.dff,
              Pdrop=args.drop_rate,
              lr=args.lr,
              batch=args.batch,
              epoch=args.epoch,
              seed=args.seed,
              checkpoint_directory=args.checkpoint_directory,
              log_level=DEBUG if args.debug else INFO)
