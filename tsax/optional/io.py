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
from typing import Optional, Union, Tuple
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

    def sizes(self) -> Tuple[int,...]:
        """
        Get Sizes of used Time Stamp Features

        Returns
        -------
        Vs : tuple of ints
            Sizes of used Time Stamp Features
        """
        Vs = []

        if self.month:
            Vs.append(12)

        if self.day:
            Vs.append(31)

        if self.week:
            Vs.append(7)

        if self.hour:
            Vs.append(24)

        if self.minute:
            Vs.append(60)

        if self.second:
            Vs.append(60)

        logger.debug("Feature Sizes: %s", Vs)
        return tuple(Vs)


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

    return jnp.stack(cat, axis=1)


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
    DT = t[-1] - t[0]
    dt = t[1]  - t[0]

    logger.debug("%s <= time delta <= %s", DT, dt)

    return TimeStampFeaturesOption(
        month = (dt < pd.to_timedelta("31 day") < DT),
        day = (dt < pd.to_timedelta("1 day") < DT),
        week = (dt < pd.to_timedelta("7 day") < DT),
        hour = (dt < pd.to_timedelta("1 hour") < DT),
        minute = (dt < pd.to_timedelta("1 minute") < DT),
        second = (dt < pd.to_timedelta("1 second") < DT),
    )


def read_csv(
        filepath_or_buffer: Union[str, os.PathLike, io.RawIOBase],
        timestamp: Optional[int] = None,
        opt: Optional[TimeStampFeaturesOption] = None,
        **kwargs
) -> Tuple[Union[Array, Tuple[Array, Array]], Tuple[int,...]]:
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
    data: Array or tuple of Array and Array
        Sequence Data
    Vs : tuple of ints
        Sizes of Categorical Features
    """
    if timestamp is not None:
        kwargs = {**kwargs, "index_col": timestamp, "parse_dates": True}

    kwargs.pop("iterator", None)
    kwargs.pop("chunksize", None)

    d: pd.DataFrame = pd.read_csv(filepath_or_buffer, # type: ignore[call-overload]
                                  **kwargs,
                                  iterator=False,
                                  chunksize=None)

    seq = jnp.asarray(d, dtype=float)
    logger.debug("Sequence Shape: %s", seq.shape)

    if timestamp is None:
        return seq, tuple()

    if not isinstance(d.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex")
        return seq, tuple()

    if opt is None:
        opt = inferTimeStampFeaturesOption(d.index)

    cat = extractTimeStampFeatures(d.index, opt)

    if cat is None:
        return seq, tuple()

    logger.debug("Categorical Shape: %s", cat.shape)
    return (seq, cat), opt.sizes()
