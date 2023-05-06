"""
Example 00: Copy Transformer
============================

Additional Requirements
-----------------------
* optax: https://optax.readthedocs.io/
* tqdm: https://tqdm.github.io/
"""
import argparse
import functools
from logging import getLogger, INFO, DEBUG, StreamHandler
import os
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from tqdm import tqdm

from tsax.transformer import Transformer

logger = getLogger("tsax.example.example00")
logger.addHandler(StreamHandler())


def example00(V, L, N, dm, nH, dff, Pdrop,
              seed, data_size, batch, epoch, lr,
              log_level):
    logger.setLevel(log_level)
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    logger.info("Random Seed: %x", seed)
    key = jax.random.PRNGKey(seed)

    logger.info("Data: size=%d, max-length=%d", data_size, L)
    key, key_use = jax.random.split(key, 2)
    data = jax.random.randint(key_use, (data_size, L), 0, V)

    # Inputs must include 2 steps
    key, key_use = jax.random.split(key, 2)
    inputs_mask = jax.random.bernoulli(key_use,
                                       p=0.8, shape=(data_size, L)).at[:,:2].set(1)
    inputs_mask = jnp.cumprod(inputs_mask, axis=1, dtype=int)
    inputs = data * inputs_mask

    # Outputs mus include 1 step and less than Inputs steps
    key, key_use = jax.random.split(key, 2)
    outputs_mask = jax.random.bernoulli(key_use,
                                        p=0.7, shape=(data_size, L)).at[:,0].set(1)
    outputs_mask = outputs_mask.at[:].set(
        jax.vmap(lambda o, idx: o.at[idx].set(0))(
            outputs_mask,
            jnp.where(jnp.all(inputs_mask == 1, axis=1),
                      L -1,
                      jnp.argmin(inputs_mask, axis=1) - 1)
        )
    )
    outputs_mask = jnp.cumprod(outputs_mask, axis=1, dtype=int)
    outputs = data * outputs_mask

    valid_size = int(data_size * 0.1)
    train_size = data_size - valid_size
    logger.info("Train: %d, Valid: %d", train_size, valid_size)

    train_inputs = inputs.at[:train_size].get()
    train_inputs_mask = inputs_mask.at[:train_size].get()
    train_outputs = outputs.at[:train_size].get()
    train_outputs_mask = outputs_mask.at[:train_size].get()

    valid_inputs = inputs.at[train_size:].get()
    valid_inputs_mask = inputs_mask.at[train_size:].get()
    valid_outputs = outputs.at[train_size:].get()
    valid_outputs_mask = outputs_mask.at[train_size:].get()


    logger.info("Transformer: V=%d, L=%d, N=%d, dm=%d, nH=%d, dff=%d, Pdrop=%f",
                V, L, N, dm, nH, dff, Pdrop)
    T = Transformer(V=V,
                    L=L,
                    N=N,
                    dm=dm,
                    nH=nH,
                    dff=dff,
                    Pdrop=Pdrop)

    key, key_use = jax.random.split(key, 2)
    params = T.init(key_use,
                    train_inputs.at[:batch].get(),
                    train_inputs_mask.at[:batch].get(),
                    train_outputs.at[:batch].get(),
                    train_outputs_mask.at[:batch].get())

    logger.info("Adam: Learning Rate: %f", lr)
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)

    @functools.partial(jax.jit, static_argnames="with_dropout")
    def loss_fn(p, k, i, imask, o, omask, *, with_dropout):
        prob = T.apply(p, i, imask, o, omask,
                       with_dropout=with_dropout, only_next=True,
                       rngs={"dropout": key})
        true = jax.vmap(lambda _i, _idx: _i.at[_idx].get())(i,
                                                            jnp.argmin(omask, axis=1))

        cross = jax.vmap(lambda _p, _t: -jnp.log(_p.at[_t].get()))(prob, true)
        cross = cross.at[:].set(jnp.where(jnp.isfinite(cross), cross, 1000))
        return jnp.sum(cross), (prob, true)

    train_fn = jax.value_and_grad(functools.partial(loss_fn, with_dropout=True),
                                  has_aux=True)

    idx = jnp.arange(train_size)
    nbatch = train_size // batch
    logger.info("Start Training")
    for ep in range(epoch):
        t = time.perf_counter()
        key, key_use = jax.random.split(key, 2)
        idx = jax.random.permutation(key_use, idx)

        epoch_loss = 0
        for i in tqdm(range(nbatch), ascii=True, leave=False):
            bidx = idx.at[i:i+batch].get()
            key, key_use = jax.random.split(key, 2)
            (loss, _), grad = train_fn(params,
                                  key_use,
                                  train_inputs.at[bidx].get(),
                                  train_inputs_mask.at[bidx].get(),
                                  train_outputs.at[bidx].get(),
                                  train_outputs_mask.at[bidx].get())
            epoch_loss += loss
            updates, opt_state = tx.update(grad, opt_state)
            params = optax.apply_updates(params, updates)

        dt = (time.perf_counter() - t)
        logger.info("Train: Epoch: %d, Loss: %.6f, Elapsed: %.3f sec",
                    ep, epoch_loss / (batch * nbatch), dt)

        if ep % 10 == 0:
            valid_loss, (valid_prob, valid_true) = loss_fn(params, None,
                                                           valid_inputs,
                                                           valid_inputs_mask,
                                                           valid_outputs,
                                                           valid_outputs_mask,
                                                           with_dropout=False)
            logger.info("Valid: Epoch: %d, Loss: %.6f, Acc: %.6f",
                        ep, valid_loss / valid_size,
                        jnp.sum(jnp.argmax(valid_prob, axis=1) == valid_true) /
                        valid_size)

    final_loss = loss_fn(params, None,
                         valid_inputs, valid_inputs_mask,
                         valid_outputs, valid_outputs_mask,
                         with_dropout=False)
    logger.info("Final Valid: Total Epoch %d, Loss: %.6f", epoch, final_loss)


if __name__ == "__main__":
    p = argparse.ArgumentParser("example-00")
    p.add_argument("--max-value", help="Max Value (Exclusive)", type=int, default=256)
    p.add_argument("--max-length", help="Max Length", type=int, default=32)
    p.add_argument("--N", type=int, default=3,
                    help="Number of Layers at Encoder/Decoder Stack")
    p.add_argument("--dm", type=int, default=32,
                    help="Number of Model Dimension")
    p.add_argument("--nH", type=int, default=4,
                    help="Number of Multi Head")
    p.add_argument("--dff", type=int, default=64,
                    help="Number of Units at Hidden Layer in Feed Forward")
    p.add_argument("--drop-rate", type=float, default=0.1,
                    help="Rate of Dropout for Training")
    p.add_argument("--seed", type=int, default=None,
                    help="Random Seed")
    p.add_argument("--data-size", type=int, default=10000,
                    help="Number of Data")
    p.add_argument("--batch", type=int, default=32,
                    help="Batch Size")
    p.add_argument("--epoch", type=int, default=300,
                    help="Number of Epoch")
    p.add_argument("--lr", type=float, default=1e-4,
                    help="Learning Rate")
    p.add_argument("--debug", action="store_true", help="Enable Debug")

    args = p.parse_args()
    example00(V=args.max_value,
              L=args.max_length,
              N=args.N,
              dm=args.dm,
              nH=args.nH,
              dff=args.dff,
              Pdrop=args.drop_rate,
              seed=args.seed,
              data_size=args.data_size,
              batch=args.batch,
              epoch=args.epoch,
              lr=args.lr,
              log_level=DEBUG if args.debug else INFO)
