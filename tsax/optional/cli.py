"""
CLI module (:mod:`tsax.optional.cli`)
=====================================

See Also
--------
tsax.optional.io
tsax.optional.experiment
tsax.optional.oam


Notes
-----
This module requires optional dependancies.
``pip install tsax[cli]``
"""
from __future__ import annotations
from argparse import ArgumentDefaultsHelpFormatter
from dataclasses import dataclass
from logging import INFO, DEBUG, StreamHandler, Formatter
import subprocess
import sys
from typing import Any, Callable, Literal, Optional, Tuple, Type, Union

import jax
import optax
import wblog

from tsax.typing import KeyArray, DataT
from tsax.logging import enable_logging
from tsax.random import initPRNGKey
from tsax.data import SeqData
from tsax.loss import AE, SE
from tsax.version import get_version
from tsax.core import Model
from tsax.model import (
    Informer,
    Autoformer,
)
from tsax.optional.io import inferTimeStampFeaturesOption, read_csv
from tsax.optional.experiment import (
    TrainState,
    PredictState,
    train,
    load,
    predict,
)
from tsax.optional.oam import arg, ArgumentParser


__all__ = [
    "CLIArgs",
    "cli",
]

EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1

logger = wblog.getLogger()


@dataclass
class CLIArgs:
    """
    CLI Arguments
    """
    # Generic
    action: Literal["train", "predict", "board"] = arg(help="CLI Action")
    seed: Optional[int] = arg(
        default=None,
        help="Seed for PRNG. If None, hardware random is used."
    )
    verbose: bool = arg(default=False, help="Enable Verbose Logging")
    debug: bool = arg(default=False, help="Enable Debug Logging")

    data: str = arg(default="data.csv", help="Data")
    data_timestamp: Optional[int] = arg(
        default=None,
        help="TimeStamp Column at Data (optional)"
    )
    data_stride: int = arg(default=1, help="Data Stride")

    # Model
    model: Literal["informer", "autoformer"] = arg(
        default="autoformer",
        help="Model"
    )
    I: int = arg(default=10, help="Input Length. Lookback Horizon.")
    O: int = arg(default=10, help="Output Length. Prediction Horizon.")
    nE: int = arg(default=3, help="Number of Encoder Layers")
    nD: int = arg(default=3, help="Number of Decoder Layers")
    nH: int = arg(default=8, help="Number of Multi Head at Attention Layers")
    dff: int = arg(default=256, help="Number of Hidden Units at Feed Forward Layers")
    dm: int = arg(default=32, help="Internal Model Dimension. (Some models ignores.)")
    c: int = arg(
        default=3,
        help="Hyper Parameter. Sampling Factor for Informer and Autoformer."
    )
    alpha: float = arg(
        default=1.0,
        help="Scaling Factor for Data Embedding. " +
        "1.0 (default) is fine for Normalized Sequence."
    )
    Pdrop: float = arg(default=0.1, help="Dropout Rate")
    eps: float = arg(default=1e-12, help="Small Positive Value for LayerNorm")
    load_dir: Optional[str] = arg(default=None, help="Checkpoint Directory to Load")
    load_which: Literal["best", "latest"] = arg(
        default = "best",
        help="Which step to load",
    )

    # Train
    lr: float = arg(
        default=1e-5,
        help="Learning Rate for Optimizer"
    )
    loss: Literal["mse", "mae"] = arg(
        default="mse",
        help="Loss Function for Training"
    )
    batch: int = arg(default=32, help="Batch Size")
    epoch: int = arg(default=100, help="Training Epoch")
    valid_data: Optional[str] = arg(
        default=None,
        help="Validation Data"
    )
    valid_data_timestamp: Optional[int] = arg(
        default=None,
        help="TimeStamp Column at Validation Data (optional)"
    )
    valid_ratio: Optional[float] = arg(
        default=None,
        help="Validation Ratio for Splitting"
    )
    valid_freq: int = arg(
        default=10,
        help="Validation Frequency for Training"
    )


def setup_logging(args: CLIArgs) -> None:
    """
    Set up Logging

    Parameters
    ----------
    args : CLIArgs
        CLI Arguments
    """
    h = StreamHandler()
    Formatter.default_msec_format = '%s.%03d'
    h.setFormatter(Formatter("%(asctime)s: %(name)s: %(levelname)s: %(message)s"))
    if args.debug:
        enable_logging(DEBUG, h)
        logger.info("Enable Debug")
        return None

    if args.verbose:
        enable_logging(INFO, h)
        logger.info("Enable Verbose")
        return None


def createModel(args: CLIArgs, data: SeqData, Vs: Tuple[int, ...]) -> Model:
    """
    Create Model

    Parameters
    ----------
    args : CLIArgs
        CLI Arguments
    data : SeqData
        Data for Model
    Vs : tuple of ints
        Sizes of Categorical Features.
    """
    kwargs = dict(
        d=data.dimension(),
        I=args.I,
        O=args.O,
        dm=args.dm,
        Vs=Vs,
        alpha=args.alpha,
        nE=args.nE,
        nD=args.nD,
        nH=args.nH,
        dff=args.dff,
        c=args.c,
        eps=args.eps,
        Pdrop=args.Pdrop,
    )

    model_class: Type[Model]
    logger.info("Model: %s", args.model)
    if args.model == "informer":
        model_class = Informer
        kwargs["Ltoken"] = args.I // 2
    elif args.model == "autoformer":
        model_class = Autoformer
    else:
        raise ValueError(f"Unknown Model: {args.model}")

    return model_class(**kwargs)


def train_cli(args: CLIArgs,
              key: KeyArray,
              data: SeqData,
              model: Model) -> int:
    """
    Training CLI

    Parameters
    ----------
    args : CLIArgs
        CLI Arguments
    key : KeyArray
        PRNG Key
    data : SeqData
        Data for Train
    model : Model
        Model for Train

    Returns
    -------
    int
        Return Code
    """
    logger.info("Adam(lr=%f)", args.lr)
    tx = optax.adam(learning_rate=args.lr)

    key, key_use = jax.random.split(key)
    state = TrainState.create_for(key_use, model, data, tx)
    if args.load_dir is not None:
        state = load(state, args.load_dir, args.load_which)

    valid_data: Optional[SeqData]
    if args.valid_ratio is not None:
        logger.info("Split Train / Valid Data")
        data, valid_data = data.train_test_split(args.valid_ratio)
    elif args.valid_data is not None:
        logger.info("Use Separete Valid Data")
        timestamp = args.valid_data_timestamp
        if timestamp is None:
            timestamp = args.data_timestamp
        v, _ = read_csv(args.valid_data, timestamp)
        valid_data = SeqData(v, xLen=args.I, yLen=args.O,
                             batch_size=args.batch, stride=args.data_stride)
    else:
        logger.info("No Valid Data")
        valid_data = None

    loss_fn = {
        "mse": SE,
        "mae": AE,
    }[args.loss]

    train(key, state, data, args.epoch, loss_fn, valid_data, args.valid_freq)
    return EXIT_SUCCESS


def predict_cli(args: CLIArgs,
                key: KeyArray,
                data: SeqData,
                model: Model) -> int:
    """
    Predict CLI

    Parameters
    ----------
    args : CLIArgs
        CLI Arguments
    key : KeyArray
        PRNG Key
    data : SeqData
        Data for Predict
    model : Model
        Model for Predict

    Returns
    -------
    int
        Return Code
    """
    key, key_use = jax.random.split(key)
    state = PredictState.create_for(key_use, model, data)
    if args.load_dir is not None:
        state = load(state, args.load_dir, args.load_which)

    pred = predict(key, state, data)
    return EXIT_SUCCESS


def board_cli(args: CLIArgs) -> int:
    """
    Board CLI

    Parameters
    ----------
    args : CLIArgs
        CLI Arguments

    Returns
    -------
    int
        Return Code
    """
    ret = subprocess.run([
        "streamlit",
        "run",
        __file__.replace("cli.py", "board.py"),
        "--",
        "--directory",
        args.load_dir or ".",
    ])
    return ret.returncode


def load_data(args: CLIArgs) -> Tuple[SeqData, Tuple[int, ...]]:
    """
    Load Data

    Parameters
    ----------
    args : CLIArgs
        CLI Arguments

    Returns
    -------
    data : SeqData
        Loaded Data
    Vs : tuple of ints
        Sizes of Categorical Features
    """
    raw, Vs = read_csv(args.data, args.data_timestamp)
    data = SeqData(raw, xLen=args.I, yLen=args.O,
                   batch_size=args.batch, stride=args.data_stride)
    return data, Vs


def cli(args: Optional[CLIArgs] = None) -> int:
    """
    CLI

    Parameters
    ----------
    args : CLIArgs, optional
        CLI Arguments. If ``None`` (default), parsed from Command Line Option

    Returns
    -------
    int
        Return Code
    """
    if args is None:
        parser = ArgumentParser(
            CLIArgs,
            "python -m tsax",
            description=f"TSax Command Line Interface (v{get_version()})",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        args = parser.parse_args()

    setup_logging(args)

    if args.action == "board":
        return board_cli(args)

    key: KeyArray = initPRNGKey(args.seed)

    data, Vs = load_data(args)
    model = createModel(args, data, Vs)

    logger.info("Action: %s", args.action)
    if args.action == "train":
        return train_cli(args, key, data, model)
    elif args.action == "predict":
        return predict_cli(args, key, data, model)
    else:
        raise ValueError(f"Unkown Action: {args.action}")
