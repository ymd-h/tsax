"""
Version module (:mod:`tsax.version`)
====================================
"""
from __future__ import annotations
from importlib.metadata import version
from typing import Tuple

__all__ = [
    "get_version",
    "get_version_tuple",
]



def get_version(pkg: str = "tsax") -> str:
    """
    Get Version

    Parameters
    ----------
    pkg : str, optional
        Package Name. Default is ``"tsax"``

    Returns
    -------
    version : str
        Verion String
    """
    return version(pkg)


def get_version_tuple(pkg: str = "tsax") -> Tuple[int, ...]:
    """
    Get Version as tuple

    Parameters
    ----------
    pkg : str, optional
        Package Name. Default is ``"tsax"``

    Returns
    -------
    version : tuple of ints
        Version tuple
    """
    return tuple(int(v) for v in get_version(pkg).split("."))

