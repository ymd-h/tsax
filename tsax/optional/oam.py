"""
Object Argument Mapper (:mod:`tsax.optional.oam`)
=================================================

Notes
-----
This module requires optional dependancies.
``pip install tsax["oam"]``
"""
from __future__ import annotations
from dataclasses import field, fields, MISSING
from typing import (
    get_args,
    get_origin,
    get_type_hints,
    Literal,
)

import argparse_dataclass
import wblog

from tsax.version import get_version_tuple


__all__ = [
    "ArgumentParser",
    "arg",
]

logger = wblog.getLogger()


def _patch_fields(cls, *args, **kwargs):
    t = get_type_hints(cls)

    def _ensure_type(_f):
        # Ensure type under `from __future__ import annotations`
        # https://github.com/mivade/argparse_dataclass/issues/47
        _f.type = t[_f.name]
        return _f

    return tuple(_ensure_type(f) for f in fields(cls, *args, **kwargs))

argparse_dataclass.fields = _patch_fields


def arg(**kwargs):
    f = {}
    known_list = [
        "default",
        "default_factory",
        "init",
        "repr",
        "hash",
        "compare",
        "kw_only"
    ]
    for kl in known_list:
        if kl in kwargs:
            f[kl] = kwargs.pop(kl)
    f = {**f, "metadata": kwargs}

    return field(**f)


class ArgumentParser(argparse_dataclass.ArgumentParser):
    def add_argument(self, *args, **kwargs):
        if kwargs.get("default") == MISSING:
            # https://github.com/mivade/argparse_dataclass/issues/32
            name = args[0].replace("--", "").replace("-", "_")
            f = next(filter(lambda f: f.name == name, fields(self._options_type)))
            kwargs["default"] = f.default
            logger.debug("Patch default of %s: MISSING -> %s", name, f.default)

        # Literal Support has been implemented at main branch,
        # however, it hasn't been released yet.
        v = get_version_tuple("argparse_dataclass")
        if (v <= (1, 0, 0)) and (get_origin(kwargs.get("type")) is Literal):
            t = kwargs["type"]
            types = [type(a) for a in get_args(t)]

            kwargs["choices"] =  get_args(t)
            kwargs["type"] = types[0]

        return super().add_argument(*args, **kwargs)
