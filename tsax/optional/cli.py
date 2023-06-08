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

from tsax.logging import enable_logging
from tsax.random import initPRNGKey
from tsax.typing import KeyArray
from tsax.version import get_version
from tsax.model import (
    Informer,
    Autoformer,
)
from tsax.optional.io import inferTimeStampFeaturesOption, read_csv
from tsax.optional.experiment import (
    TrainState,
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
    data: str = arg(help="Data")
    action: Literal["train", "predict"] = arg(help="CLI Action: {train, predict}")
    model: Literal["informer", "autoformer"] = arg(
        help="Model: {informer, autoformer}"
    )
    I: int = arg(help="Input Length. Lookback Horizon.")
    O: int = arg(help="Output Length. Prediction Horizon.")
    nE: int = arg(default=3, help="Number of Encoder Layers")
    nD: int = arg(default=3, help="Number of Decoder Layers")
    nH: int = arg(default=8, help="Number of Multi Head at Attention Layers")
    dff: int = arg(default=256, help="Number of Hidden Units at Feed Forward Layers")
    dm: int = arg(default=32, help="Internal Model Dimension. (Some models ignores.)")
    Pdrop: float = arg(default=0.1, help="Dropout Rate")
    eps: float = arg(default=1e-12, help="Small Positive Value for LayerNorm")
    lr: float = arg(
        default=1e-5,
        help="Learning Rate for Optimizer"
    )
    batch: int = arg(default=32, help="Batch Size")
    epoch: int = arg(default=100, help="Training Epoch")
    seed: Optional[int] = arg(
        default=None,
        help="Seed for PRNG. If None (default), hardware random is used."
    )
    verbose: bool = arg(default=False, help="Enable Verbose Logging")
    debug: bool = arg(default=False, help="Enable Debug Logging")


def train_cli(args: CLIArgs, key: KeyArray) -> int:
    train(model=args.model)
    return EXIT_SUCCESS


def predict_cli(args: CLIArgs, key: KeyArray) -> int:
    predict(model=args.model)
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


    if args.debug:
        enable_logging(DEBUG)
        logger.info("Enable Debug")
    elif args.verbose:
        enable_logging(INFO)
        logger.info("Enable Verbose")

    key: KeyArray = initPRNGKey(args.seed)

    try:
        logger.info("action: %s", args.action)
        fn: Optional[Callable[[CLIArgs, KeyArray], int]] = {
            "train": train_cli,
            "predict": predict_cli
        }.get(args.action, key)

        if fn is None:
            logger.critical(f"Unknown Action: {args.action}")
            return EXIT_FAILURE

        return fn(args)
    except Exception as e:
        logger.critical(e)
        return EXIT_FAILURE
