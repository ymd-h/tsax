"""
Board (:mod:`tsax.optional.board`)
==================================

This module requires optional dependancies.
``pip install tsax[board]``
"""
from __future__ import annotations
from dataclasses import dataclass
import pathlib
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
from orbax.checkpoint import (
    Checkpointer,
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointHandler,
)
import pandas as pd
import streamlit as st
import wblog

from tsax.typing import Array, DataT
from tsax.typed_jax import vmap
from tsax.optional.oam import arg, ArgumentParser


logger = wblog.getLogger()

@dataclass
class CLIArgs:
    """
    CLI Arguments
    """
    directory: str = arg()


def best_fn(m: Dict[str, float]) -> float:
    return m.get("train_loss", jnp.inf)


def draw_train_metrics(path: List[pathlib.Path], check: List[bool]) -> None:
    data: List[pd.DataFrame] = []
    for p, c in zip(path, check):
        logger.debug("dir: %s, check: %s", p.name, c)
        if not c:
            continue

        logger.debug("Load %s Metrics", p.name)
        ckpt = CheckpointManager(p,
                                 Checkpointer(PyTreeCheckpointHandler()),
                                 CheckpointManagerOptions(best_fn=best_fn))

        data.append(pd.DataFrame([{"step": ck.step,
                                   p.name: ck.metrics["train_loss"]}
                                  for ck in ckpt._checkpoints])
                    .set_index("step"))
        logger.debug(data[-1])

    st.line_chart(pd.concat(data, axis=1))


def train_metrics(args: CLIArgs) -> None:
    st.title("Training Metrics")

    d = pathlib.Path(args.directory)
    chkpt = sorted([g for g in d.glob("*") if g.is_dir()], key=lambda g: g.name)
    logger.debug("Load Dir: %s", chkpt)

    checkbox = [st.checkbox(c.name) for c in chkpt]

    st.button("Draw", on_click=draw_train_metrics, args=(chkpt, checkbox))


def flatten(data: List[DataT]) -> Dict[str, Array]:
    leaves = []
    keys = []
    for d in data:
        KL, _ = jax.tree_util.tree_flatten_with_path(data)
        keys.append([kl[0] for kl in KL])
        leaves.append([kl[1] for kl in KL])

    leaves_zip = zip(*leaves)
    ret = {}
    for k, l in zip(keys[0], leaves_zip):
        ret[''.join(str(_k) for _k in k[1:])] = jnp.stack(l)

    return ret


@st.cache_data
def load_weight(path: pathlib.Path) -> Tuple[List[int], Dict[str, Array]]:
    ckpt = CheckpointManager(path,
                             Checkpointer(PyTreeCheckpointHandler()),
                             CheckpointManagerOptions(best_fn=best_fn))
    steps: List[int] = ckpt.all_steps()

    params: Dict[str, Array] = flatten([ckpt.restore(s)["params"] for s in steps])

    return steps, params

def visualize_weight(args: CLIArgs) -> None:
    st.title("Visualize Weights")

    d = pathlib.Path(args.directory)
    chkpt = [g for g in d.glob("*") if g.is_dir()]
    logger.debug("Load Dir: %s", chkpt)

    radio = st.radio("checkpoint", options=sorted([c.name for c in chkpt]))

    steps, w = load_weight(next(c for c in chkpt if c.name == radio))

    radio_w = st.radio("weight", options=w.keys())

    if radio_w is not None:
        data = pd.DataFrame(jnp.reshape(w[radio_w], (len(steps), -1)), index=steps)
        st.line_chart(data)

features: Dict[str, Callable[[CLIArgs], None]] = {
    "Training Metrics": train_metrics,
    "Visualize Weight": visualize_weight,
}

args = ArgumentParser(
    CLIArgs,
    "python -m tsax --action board",
).parse_args()

selected = st.sidebar.selectbox("Features", features.keys())
if selected is not None:
    features[selected](args)
