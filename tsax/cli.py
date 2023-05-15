"""
CLI module (:mod:`tsax.cli`)
============================
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import sys
from typing import Any, Dict, Optional, Union


from tsax.version import get_version


EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1


@dataclass
class CLIArgs:
    version: bool = False



def cli(args: Optional[CLIArgs] = None) -> int:
    if args is None:
        parser = argparse.ArgumentParser("tsax.cli")
        parser.add_argument("--version", action="store_true", help="Show Version")

        args = CLIArgs(**vars(parser.parse_args()))

    if args.version:
        print(f"TSax v{get_version()}")
        return EXIT_SUCCESS


    return EXIT_SUCCESS

if __name__ == "__main__":
    sys.exit(cli())
