"""
IO (:mod:`tsax.optional.io`)
============================

Notes
-----
This module requires optional dependancies.
`pip install tsax[io]`
"""
from __future__ import annotations
import io
import os
from typing import Optional, Union
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pandas as pd
import wblog

from tsax.typing import Array, ArrayLike


__all__ = [
    "TimeStampFeaturesOption",
    "extractTimeStampFeatures",
    "inferTimeStampFeaturesOption",
    "read_csv",
]


logger = wblog.getLogger()


@dataclass
class TimeStampFeaturesOption:
    month: bool
    day: bool
    week: bool
    hour: bool
    minute: bool
    second: bool

    def __post_init__(self):
        logger.debug(self)


def extractTimeStampFeatures(t: pd.DatetimeIndex,
                             opt: TimeStampFeaturesOption) -> Optional[Array]:
    """
    Extract TimeStamp Features

    Parameters
    ----------
    t : pandas.DatetimeIndex
        Time Stamp
    opt : TimeStampFeaturesOption
        Option

    Returns
    -------
    cat : Array, optional
        Time Stamp Features
    """
    cat = []

    if opt.month:
        cat.append(jnp.asarray(t.month - 1, dtype=int))

    if opt.day:
        cat.append(jnp.asarray(t.day - 1, dtype=int))

    if opt.week:
        cat.append(jnp.asarray(t.weekday, dtype=int))

    if opt.hour:
        cat.append(jnp.asarray(t.hour, dtype=int))

    if opt.minute:
        cat.append(jnp.asarray(t.minute, dtype=int))

    if opt.second:
        cat.append(jnp.asarray(t.second, dtype=int))

    if len(cat) == 0:
        logger.warning("Time Stamp Features are Empty")
        return None

    return jnp.stack(cat)


def inferTimeStampFeaturesOption(t: pd.DatetimeIndex) -> TimeStampFeaturesOption:
    """
    Infer Time Stamp Features Option

    Parameters
    ----------
    t : pandas.DatetimeIndex
        Time Stamp

    Returns
    -------
    opt : TimeStampFeaturesOption
        Option
    """
    logger.info("Infer TimeStampFeaturesOption")
    DT = t.iloc[-1] - t.iloc[0]
    dt = t.iloc[1] - t.iloc[0]

    logger.debug("DT: %s, dt: %s", DT, dt)

    return TimeStampFeaturesOption(
        month = (dt < pd.to_timedelta("month") < DT),
        day = (dt < pd.to_timedelta("day") < DT),
        week = (dt < pd.to_timedelta("7 day") < DT),
        hour = (dt < pd.to_timedelta("hour") < DT),
        minute = (dt < pd.to_timedelta("minute") < DT),
        second = (dt < pd.to_timedelta("second") < DT),
    )


def read_csv(filepath_or_buffer: Union[str, os.PathLike, io.RawIOBase],
             timestamp: Optional[int] = None,
             opt: Optional[TimeStampFeaturesOption] = None,
             **kwargs) -> Union[Array, Tuple[Array, Array]]:
    """
    Read CSV

    Parameters
    ----------
    filepath_or_buffer : str, os.PathLike, io.RawIOBase
        CSV
    timestampe : int, optional
        Time Stamp Column
    **kwargs
        Additional Keyword Arguments to be passed to ``pandas.read_csv()``

    Returns
    -------
    seq : Array
        Sequence Data
    cat : Array, optional
        Categorial Data from Time Stamp
    """
    if timestamp is not None:
        kwargs = {**kwargs, "index_col": timestamp, "parse_dates": True}

    d = pd.read_csv(filepath_or_buffer, **kwargs)

    seq = jnp.asarray(d, dtype=float)
    logger.debug("Sequence Shape: %s", seq.shape)

    if timestamp is None:
        return seq

    if opt is None:
        opt = inferTimeStampFeaturesOption(seq.index)

    cat = extractTimeStampFeatures(seq.index, opt)

    if cat is None:
        return seq

    return seq, cat
