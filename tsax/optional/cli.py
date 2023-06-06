"""
CLI module (:mod:`tsax.cli`)
============================

Notes
-----
This module requires optional dependancies.
`pip install tsax[cli]`
"""
from __future__ import annotations
from dataclasses import dataclass
from logging import INFO, DEBUG, StreamHandler
import sys
from typing import Any, Callable, Literal, Optional, Union

from argparse_dataclass import ArgumentParser
import wblog

from tsax.logging import enable_logging
from tsax.random import initPRNGKey
from tsax.typing import KeyArray
from tsax.version import get_version
from tsax.optional.io import inferTimeStampFeaturesOption, read_csv
from tsax.optional.experiment import train, load, predict


EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1


logger = wblog.getLogger()


@dataclass
class CLIArgs:
    """
    CLI Arguments
    """
    action: Literal["train", "predict"]
    model: Literal["informer", "autoformer"]
    I: int
    O: int
    nE: int = field(default=3)
    nD: int = field(default=3)
    nH: int = field(default=8)
    dff: int = field(default=256)
    dm: int = field(default=32)
    Pdrop: float = field(default=0.1)
    eps: float = field(default=1e-12)
    seed: Optional[int] = field(default=None)
    version: bool = field(default=False)
    verbose: bool = field(default=False)
    debug: bool = field(default=False)


def train_cli(args: CLIArgs,
              key: KeyArray) -> int:
    train(model=args.model)
    return EXIT_SUCCESS


def predict_cli(args: CLIArgs,
                key: KeyArray) -> int:
    predict(model=args.model)
    return EXIT_SUCCESS


def cli(args: Optional[CLIArgs] = None) -> int:
    if args is None:
        parser = ArgumentParser(CLIArgs,
                                "python -m tsax",
                                description="TSax Command Line Interface")
        args = parser.parse_args()


    if args.version:
        print(f"TSax v{get_version()}")
        return EXIT_SUCCESS

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
