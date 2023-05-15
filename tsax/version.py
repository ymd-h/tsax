"""
Version module (:mod:`tsax.version`)
====================================
"""
from __future__ import annotations
from importlib.metadata import version

__all__ = [
    "get_version",
]



def get_version() -> str:
    """
    Get TSax Version

    Returns
    -------
    version : str
        Verion String
    """
    return version("tsax")
