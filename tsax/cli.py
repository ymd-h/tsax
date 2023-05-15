"""
CLI module (:mod:`tsax.cli`)
============================
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from logging import INFO, DEBUG, StreamHandler
import sys
from typing import Any, Dict, Literal, Optional, Union

import wblog

from tsax.experiment import train, inference
from tsax.logging import enable_logging
from tsax.version import get_version


EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1


logger = wblog.getLogger()


@dataclass
class CLIArgs:
    """
    CLI Arguments
    """
    action: Literal["train", "inference"]
    model: Literal["informer", "autoformer"]
    nE: int
    nD: int
    nH: int
    dff: int
    dm: int
    version: bool = False
    verbose: bool = False
    debug: bool = False


def train_cli(args: CLIArgs) -> int:
    train(args.model)
    return EXIT_SUCCESS


def inference_cli(args: CLIArgs) -> int:
    inference(args.model)
    return EXIT_SUCCESS


def cli(args: Optional[CLIArgs] = None) -> int:
    if args is None:
        parser = argparse.ArgumentParser("tsax.cli")
        parser.add_argument("--action",
                            choices=["train", "inference"],
                            default="train")
        parser.add_argument("--model",
                            choices=["informer", "autoformer"],
                            default="informer")

        m = parser.add_argument_group("model")
        m.add_argument("--nE", type=int, default=3,
                       help="Number of Encoder Stacks")
        m.add_argument("--nD", type=int, default=3,
                       help="Number of Decoder Stacks")
        m.add_argument("--nH", type=int, default=8,
                       help="Number of Multi Head")
        m.add_argument("--dff", type=int, default=256,
                       help="Hidden Units at FeedForward")
        m.add_argument("--dm", type=int, default=32,
                       help="Model Dimension")

        parser.add_argument("--version", action="store_true")
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--debug", action="store_true")

        args = CLIArgs(**vars(parser.parse_args()))


    if args.version:
        print(f"TSax v{get_version()}")
        return EXIT_SUCCESS

    if args.debug:
        enable_logging(DEBUG)
        logger.info("Enable Debug")
    elif args.verbose:
        enable_logging(INFO)
        logger.info("Enable Verbose")

    try:
        logger.info("action: %s", args.action)
        fn = {
            "train": train_cli,
            "inference": inference_cli
        }.get(args.action)

        if fn is None:
            logger.critical(f"Unknown Action: {args.action}")
            return EXIT_FAILURE

        return fn(args)
    except Exception as e:
        logger.critical(e)
        return EXIT_FAILURE
