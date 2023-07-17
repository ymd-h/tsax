"""
Experiment (:mod:`tsax.optional.experiment`)
============================================

Notes
-----
This module requires additional dependencies,
which can be installed by ``pip install tsax[experiment]``
"""
from __future__ import annotations
import dataclasses
from datetime import datetime
import functools
from logging import getLogger, FileHandler, Formatter
import os
from typing import (
    cast,
    overload,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)
import time

from typing_extensions import TypeAlias
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map
import flax
import flax.linen as nn
from flax.training import train_state, orbax_utils
from flax.struct import PyTreeNode, field
import optax
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    Checkpointer,
    PyTreeCheckpointHandler,
)
from tqdm import tqdm
import wblog

from tsax.typing import (
    Array,
    KeyArray,
    DataT,
    ModelCall,
    SplitFn,
    ModelParam,
    ModelVars,
)
from tsax.typed_jax import jit, value_and_grad
from tsax.core import Model
from tsax.data import SeqData, ensure_BatchSeqShape, data_shape

__all__ = [
    "TrainState",
    "PredictState",
    "train",
    "load",
    "predict",
]


logger = wblog.getLogger()


class TrainState(train_state.TrainState):
    """
    TSax Experiment Training State
    """
    apply_fn: ModelCall = field(pytree_node=False)
    split_fn: SplitFn = field(pytree_node=False)
    sigma_reparam: Optional[ModelParam] = field(pytree_node=True)

    @staticmethod
    def create_for(key: KeyArray,
                   model: Model,
                   data: Union[SeqData[DataT], DataT],
                   tx: optax.GradientTransformation) -> TrainState:
        """
        Create TrainState for Model & Data

        Parameters
        ----------
        key : KeyArray
            PRNG Key
        model : Model
            TSax model
        data : SeqData or DataT
            Input Data
        tx : optax.GradientTransformation
            Optax Optimizer

        Returns
        -------
        state : TrainState
            Training State
        """
        model.log_model()

        if isinstance(data, SeqData):
            x, _ = data.ibatch(0)
        else:
            x = data

        logger.info("Create TrainState for Shape: %s", data_shape(x))

        key_p, key = model.split_key(key, train=False)
        key["params"] = key_p
        if not isinstance(x, Array):
            params = model.init(key, *x)
            def apply_fn(variables, _x, *args, **kwargs):
                return model.apply(variables, *_x, *args, **kwargs)
        else:
            params = model.init(key, x)
            def apply_fn(variables, _x, *args, **kwargs):
                return model.apply(variables, _x, *args, **kwargs)

        return cast(TrainState, TrainState.create(
            apply_fn=cast(ModelCall, apply_fn),
            params=params["params"],
            tx=tx,
            split_fn=model.split_key,
            sigma_reparam=params.get("sigma_reparam", None)
        ))

    def vars(self) -> ModelVars:
        """
        Get Model Variables for ModelCall

        Returns
        -------
        ModelVars
        """
        v: ModelVars = {
            "params": self.params,
        }
        if self.sigma_reparam is not None:
            v["sigma_reparam"] = self.sigma_reparam

        return v


class PredictState(PyTreeNode):
    """
    TSax Experiment Predict State
    """
    apply_fn: ModelCall = field(pytree_node=False)
    params: ModelParam = field(pytree_node=True)
    split_fn: SplitFn = field(pytree_node=False)
    sigma_reparam: Optional[ModelParam] = field(pytree_node=True)

    @staticmethod
    def create_for(key: KeyArray,
                   model: Model,
                   data: Union[SeqData[DataT], DataT]) -> PredictState:
        """
        Create TrainState for Model & Data

        Parameters
        ----------
        key : KeyArray
            PRNG Key
        model : Model
            TSax model
        data : SeqData or DataT
            Input Data

        Returns
        -------
        state : PredictState
            Predict State
        """
        model.log_model()

        if isinstance(data, SeqData):
            x, _ = data.ibatch(0)
        else:
            x = data

        logger.info("Create TrainState for Shape: %s", data_shape(x))

        key_p, key = model.split_key(key, train=False)
        key["params"] = key_p
        if not isinstance(x, Array):
            params = model.init(key, *x)
            def apply_fn(variables, _x, *args, **kwargs):
                return model.apply(variables, *_x, *args, **kwargs)
        else:
            params = model.init(key, x)
            def apply_fn(variables, _x, *args, **kwargs):
                return model.apply(variables, _x, *args, **kwargs)

        return PredictState(
            apply_fn=cast(ModelCall, apply_fn),
            params=cast(ModelParam, params["params"]),
            split_fn=model.split_key,
            sigma_reparam=cast(Optional[ModelParam],
                               params.get("sigma_reparam", None))
        )

    def vars(self) -> ModelVars:
        """
        Get Model Variables for ModelCall

        Returns
        -------
        ModelVars
        """
        v: ModelVars = {
            "params": self.params,
        }
        if self.sigma_reparam is not None:
            v["sigma_reparam"] = self.sigma_reparam

        return v


State: TypeAlias = Union[TrainState, PredictState]


def train(
        key: KeyArray,
        state: TrainState,
        train_data: SeqData[DataT],
        epoch: int,
        loss_fn: Callable[[Array, DataT], Array],
        valid_data: Optional[SeqData[DataT]] = None,
        valid_freq: int = 10,
        checkpoint_directory: str = "./tsax-ckpt",
        checkpoint_options: Optional[CheckpointManagerOptions] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[TrainState, str]:
    """
    Train Model

    Parameters
    ----------
    key : KeyArray
        PRNG Key
    state : TrainState
        Training State
    train_data : SeqData
        Training Data
    epoch : int
        Training Epoch
    loss_fn : callable
        Loss Function
    valid_data : SeqData, optional
        Validation Data
    valid_freq : int, optional
        Validation Frequency
    checkpoint_directory : str, optional
        Directory for CheckpointManager
    checkpoint_options : CheckpointManagerOptions, optional
        Options for CheckpointManager
    checkpoint_metadata : dict, optional
        Metadata for CheckpointManager

    Returns
    -------
    state : TrainState
        Trained State
    directory : str
        Checkpoint Directory
    """
    t0 = time.perf_counter()

    directory = os.path.join(checkpoint_directory,
                             datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(directory, exist_ok=True)

    h = FileHandler(os.path.join(directory, "train.log"))
    Formatter.default_msec_format = '%s.%03d'
    h.setFormatter(Formatter("%(asctime)s: %(name)s: %(levelname)s: %(message)s"))
    getLogger("tsax").addHandler(h)
    logger.info("Checkpoint Directory: %s", directory)


    if checkpoint_options is None:
        # To save metrics, add (pseudo) best_fn.
        checkpoint_options = CheckpointManagerOptions(
            best_fn=lambda metrics: metrics.get("train_loss", 1e+10),
            best_mode="max",
        )

    ckpt = CheckpointManager(
        directory,
        Checkpointer(PyTreeCheckpointHandler()),
        options=checkpoint_options,
        metadata=checkpoint_metadata
    )

    logger.info("Epoch: %d", epoch)
    if valid_data is not None:
        logger.info("Valid Freq: %d", valid_freq)

    logger.info("Train Data: Batch Size: %d, # of Batch: %d",
                train_data.batch_size, train_data.nbatch)

    if valid_data is not None:
        logger.info("Valid Data: Batch Size: %d, # of Batch: %d",
                    valid_data.batch_size, valid_data.nbatch)

    def _train_fn(p: ModelParam, s: TrainState, k: KeyArray,
                 x: DataT, y: DataT) -> Tuple[Array, ModelParam]:
        pred, update = s.apply_fn(s.vars(), x, train=True, rngs=k,
                                  mutable=["sigma_reparam"])
        return loss_fn(pred, y), update

    train_fn = value_and_grad(_train_fn, has_aux=True)

    @jit
    def valid_fn(s: TrainState, k: KeyArray, x: DataT, y: DataT) -> Array:
        return loss_fn(s.apply_fn(s.vars(), x, train=False, rngs=k), y)


    @jit
    def train_step_fn(
            s: TrainState, k: KeyArray,
            l: Array, g: Array,
            x: DataT, y: DataT
    ) -> Tuple[TrainState, KeyArray, Array, Array]:
        k, k_use = s.split_fn(k, train=True)
        loss: Array
        update: ModelParam
        (loss, update), grad = train_fn(s.params, s, k_use, x, y)

        s = s.apply_gradients(grads=grad)

        _grad, _ = tree_flatten(tree_map(lambda _g: jnp.sum(jnp.square(_g)), grad))

        r: Optional[ModelParam] = update.get("sigma_reparam", None)
        if r is not None:
            s = s.replace(sigma_reparam=r)

        return s, k, l+loss, g + jnp.sum(jnp.asarray(_grad))

    @jit
    def valid_step_fn(s: TrainState, k: KeyArray, l: Array,
                      x: DataT, y: DataT) -> Tuple[KeyArray, Array]:
        k, k_use = s.split_fn(k, train=False)
        loss = valid_fn(s, k_use, x, y)

        return k, l+loss

    train_size: int = train_data.batch_size * train_data.nbatch

    if valid_data is not None:
        valid_size: int = valid_data.batch_size * valid_data.nbatch

    logger.info("Start Training")
    for ep in range(epoch):
        t = time.perf_counter()

        key, key_use = jax.random.split(key, 2)
        train_data.shuffle(key_use)

        epoch_loss = jnp.zeros((1,))
        epoch_grad2 = jnp.zeros((1,))
        for i in tqdm(range(train_data.nbatch),
                      ascii=True, leave=False, desc=f"Train: Epoch: {ep}"):
            x, y = train_data.ibatch(i)
            state, key, epoch_loss, epoch_grad2 = train_step_fn(state, key,
                                                                epoch_loss,
                                                                epoch_grad2,
                                                                x, y)

        dt = (time.perf_counter() - t)
        logger.info("Train: Epoch: %d, Loss: %.6f, Loss/Length: %.6f, "
                    "|Grad|^2: %.6e, "
                    "Elapsed: %.3f sec",
                    ep, float(epoch_loss) / train_size,
                    float(epoch_loss)/(train_size * train_data.yLen),
                    float(epoch_grad2) / train_size,
                    dt)

        save_args = orbax_utils.save_args_from_target(state.params)
        ckpt.save(ep,
                  state.params,
                  save_kwargs={"save_args": save_args},
                  metrics={"train_loss": float(epoch_loss) / train_size},
                  force=(ep == epoch -1))

        if (valid_data is not None) and (ep % valid_freq == 0):
            valid_loss = jnp.zeros((1,))
            for i in tqdm(range(valid_data.nbatch),
                          ascii=True, leave=False, desc=f"Valid: Epoch: {ep}"):
                x, y = valid_data.ibatch(i)
                key, valid_loss = valid_step_fn(state, key, valid_loss, x, y)

            logger.info("Valid: Epoch: %d, Loss: %.6f, Loss/Length: %.6f",
                        ep, float(valid_loss) / valid_size,
                        float(valid_loss)/(valid_size * valid_data.yLen))


    if valid_data is not None:
        final_loss = jnp.zeros((1,))
        for i in tqdm(range(valid_data.nbatch),
                      ascii=True, leave=False,
                      desc=f"Final Valid: Total Epoch: {epoch}"):
            x, y = valid_data.ibatch(i)
            key, final_loss = valid_step_fn(state, key, final_loss, x, y)

        logger.info("Final Valid: Total Epoch %d, Loss: %.6f, Loss/Length: %.6f, "
                    "Elapsed: %.3f sec",
                    epoch, float(final_loss) / valid_size,
                    float(final_loss)/(valid_size * valid_data.yLen),
                    time.perf_counter() - t0)

    getLogger("tsax").removeHandler(h)
    return state, directory


@overload
def load(state: TrainState,
         checkpoint_directory: str,
         which: Union[int,
                      Literal["latest"],
                      Literal["best"]] = "latest",
         best_fn: Optional[Callable[[Any], float]] = None) -> TrainState: ...

@overload
def load(state: PredictState,
         checkpoint_directory: str,
         which: Union[int,
                      Literal["latest"],
                      Literal["best"]] = "latest",
         best_fn: Optional[Callable[[Any], float]] = None) -> PredictState: ...

def load(state: State,
         checkpoint_directory: str,
         which: Union[int,
                      Literal["latest"],
                      Literal["best"]] = "latest",
         best_fn: Optional[Callable[[Any], float]] = None) -> State:
    """
    Load Checkpoint

    Parameters
    ----------
    state : TrainState or PredictState
        State to be set
    checkpoint_directory : str
        Checkpoint Directory
    which : int, "latest", "best", optional
        Checkpoiint Step.
    best_fn : callable, optional
        Function Determine Best Step

    Returns
    -------
    state : TrainState
        State with loaded Checkpoint
    """
    if best_fn is None:
        best_fn = lambda metrics: cast(float, metrics.get("train_loss", 1e+10))

    ckpt = CheckpointManager(checkpoint_directory,
                             Checkpointer(PyTreeCheckpointHandler()),
                             CheckpointManagerOptions(best_fn=best_fn,
                                                      best_mode="min",
                                                      create=False))
    logger.debug("Stored Checkpoints: %s", ckpt.all_steps())
    logger.debug("Checkpoints Metrics: %s", [c.metrics for c in ckpt._checkpoints])

    if which == "best":
        logger.info("Use best step")
        which = ckpt.best_step()
        if which is None:
            logger.warning("No metrics are saved. Use latest step")
            which = "latest"

    if which == "latest":
        logger.info("Use latest step")
        which = ckpt.latest_step()

    if which is None:
        logger.warning("No Checkpoints are saved. Return without load.")
        return state

    logger.info("Load %d step", which)
    restore_args = orbax_utils.restore_args_from_target(state.params)
    state = dataclasses.replace(state,
                                params=ckpt.restore(which,
                                                    items=state.params,
                                                    restore_kwargs={
                                                        "restore_args": restore_args
                                                    }))

    return state

def predict(key: KeyArray,
            state: Union[TrainState, PredictState],
            data: Union[DataT, SeqData[DataT]]) -> Array:
    """
    Predict with Model

    Parameters
    ----------
    key : KeyArray
        PRNG Key
    state : TrainState or PredictState
        Trained State
    data : DataT
        Predict Data. [L, d]

    Returns
    -------
    pred : Array
        Predicted
    """
    @jit
    def pred_fn(k: KeyArray, x: DataT) -> Array:
        _, k = state.split_fn(k, train=False)
        return cast(Array, state.apply_fn(state.vars(), x, train=False, rngs=k))


    if isinstance(data, SeqData):
        idx = jnp.arange(data.nbatch)
        key = jax.random.split(key, data.nbatch)

        return jax.vmap(
            lambda i, k: pred_fn(k, cast(SeqData, data)._vxget(i))
        )(idx, key)


    data = ensure_BatchSeqShape(data)
    pred = pred_fn(key, data)

    return pred
