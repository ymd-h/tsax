"""
CLI module (:mod:`tsax.cli`)
============================

Notes
-----
This module requires optional dependancies.
`pip install tsax[cli]`
"""
from __future__ import annotations
from argparse import ArgumentDefaultsHelpFormatter
from dataclasses import dataclass, field, fields
from logging import INFO, DEBUG, StreamHandler
import sys
from typing import get_type_hints, Any, Callable, Literal, Optional, Union

import argparse_dataclass as argparse
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


EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1

logger = wblog.getLogger()

# https://github.com/mivade/argparse_dataclass/issues/47
def _patch_fields(cls, *args, **kwargs):
    t = get_type_hints(cls)

    def _update(_f):
        _f.type = t[_f.name]
        return _f

    return tuple(_update(f) for f in fields(cls, *args, **kwargs))

argparse.fields = _patch_fields

def arg(**kwargs):
    f = {}
    if "default" in kwargs:
        f["default"] = kwargs.pop("default")
    f = {**f, "metadata": kwargs}

    return field(**f)


@dataclass
class CLIArgs:
    """
    CLI Arguments
    """
    # Generic
    action: Literal["train", "predict"] = arg(help="CLI Action: {train, predict}")
    seed: Optional[int] = arg(
        default=None,
        help="Seed for PRNG. If None (default), hardware random is used."
    )
    verbose: bool = arg(default=False, help="Enable Verbose Logging")
    debug: bool = arg(default=False, help="Enable Debug Logging")

    data: str = arg(default="data.csv", help="Data")
    data_timestamp: Optional[int] = arg(
        default=None,
        help="TimeStamp Column at Data (optional)"
    )

    # Model
    model: Literal["informer", "autoformer"] = arg(
        default="autoformer",
        help="Model: {informer, autoformer}"
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
    if args.debug:
        enable_logging(DEBUG)
        logger.info("Enable Debug")
        return None

    if args.verbose:
        enable_logging(INFO)
        logger.info("Enable Verbose")
        return None


def createModel(args: CLIArgs, data: SeqData, Vs: Tuple[int, ...]) -> Model:
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

    logger.info("Model: %s", args.model)
    if args.model == "informer":
        model_class = Informer
    elif args.model == "autoformer":
        model_class = Autoformer
    else:
        raise ValueError(f"Unknown Model: {args.model}")

    return model_class(**kwargs)


def train_cli(args: TrainArgs,
              key: KeyArray,
              data: SeqData,
              model: Model) -> int:
    logger.info("Adam(lr=%f)", args.lr)
    tx = optax.adam(learning_rate=args.lr)

    key, key_use = jax.random.split(key)
    state = TrainState.create_for(key_use, model, data, tx)
    if args.load_dir is not None:
        state = load(state, load_dir, args.load_which)

    if args.valid_ratio is not None:
        data, valid_data = data.train_test_split(args.valid_ratio)
    elif args.valid_data is not None:
        timestamp = args.valid_data_timestamp
        if timestamp is None:
            timestamp = args.data_timestamp
        valid_data = read_csv(args.valid_data, timestamp)
    else:
        valid_data = None

    loss_fn = {
        "mse": SE,
        "mae": AE,
    }[args.loss]

    train(key, state, data, epoch, loss_fn, valid_data, args.valid_freq)
    return EXIT_SUCCESS


def predict_cli(args: PredictArgs,
                key: KeyArray,
                data: SeqData,
                model: Model) -> int:
    key, key_use = jax.random.split(key)
    state = PredictState.create_for(key_use, model, data)
    if args.load_dir is not None:
        state = load(state, load_dir, args.load_which)

    pred = predict(key, state, data)
    return EXIT_SUCCESS


def cli(args: Optional[CLIArgs] = None) -> int:
    if args is None:
        parser = argparse.ArgumentParser(
            CLIArgs,
            "python -m tsax",
            description=f"TSax Command Line Interface (v{get_version()})",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        args = parser.parse_args()

    setup_logging(args)
    key: KeyArray = initPRNGKey(args.seed)

    data, Vs = read_csv(args.data, args.data_timestamp)
    model = createModel(args, data, Vs)

    try:
        logger.info("Action: %s", args.action)
        if args.action == "train":
            return train_cli(args, key, data, model)
        elif args.action == "predict":
            return predict_cli(args, key, data, model)
        else:
            raise ValueError(f"Unkown Action: {args.action}")
    except Exception as e:
        logger.critical(e)
        return EXIT_FAILURE
