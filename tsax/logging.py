from __future__ import annotations
from logging import Handler
from typing import Iterable, Optional, Union

import wblog

__all__ = [
    "enable_logging",
    "disable_logging",
]

def enable_logging(level: Optional[int] = None,
                   handlers:  Union[Handler, Iterable[Handler], None] = None,
                   propagate: bool = False) -> None:
    """
    Enable Logging for TSax

    Parameters
    ----------
    level : int, optional
        Logging Level
    handlers : logging.Handler, iterable of logging.Handler, optional
        Handlers
    propagate : bool, optional
        Whether propagate to parent logger
    """
    wblog.start_logging("tsax", level, handlers=handlers, propagate=propagate)


def disable_logging() -> None:
    """
    Disable Logging for TSax
    """
    wblog.stop_logging("tsax")
