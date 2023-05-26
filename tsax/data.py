from __future__ import annotations
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from tsax.typing import ArrayLike, Array, KeyArray, CarryT


class SeqData:
    """
    Sequence Data
    """

    def __init__(self,
                 data: ArrayLike,
                 I: int,
                 O: int,
                 batch_size: int,
                 stride: int = 1):
        """
        Initialize Sequence Data

        Parameters
        ----------
        data : ArrayLike
            Sequence Data
        I : int
            Input Length
        O : int
            Output Length
        batch_size : int
            Batch Size
        stride : int, optional
            Stride Size
        """
        self.data: Array = data # [L, d]
        self.I: int = I
        self.O: int = O
        self.batch_size: int = batch_size
        self.stride: int = stride
        self.idx: Array = jnp.arange(0, self.data.shape[0], self.stride)

        self.Ishape: Tuple[int, ...] = (I, *self.data.shape[1:])
        self.Oshape: Tuple[int, ...] = (O, *self.data.shape[1:])

        self._iget = lambda i: jax.lax.dynamic_slice(self.data, (i, 0), self.Ishape)
        self._oget = lambda i: jax.lax.dynamic_slice(self.data, (i, 0), self.Oshape)

        self._viget = jax.vmap(self._iget)
        self._voget = jax.vmap(self._oget)

    def shuffle(self, key: KeyArray) -> None:
        """
        Shuffle

        Parameters
        ----------
        key : KeyArray
            PRNG Key
        """
        self.idx = jax.random.shuffle(key, self.idx)

    def _bidx(self) -> Array:
        n = self.idx.shape[0] - (self.idx.shape[0] % self.batch_size)
        return jnp.reshape(self.idx.at[:n], (-1, self.batch_size))

    def scan(self,
             fn: Callable[[CarryT, ArrayLike, ArrayLike], Tuple[CarryT, Array]],
             init: CarryT) -> Tuple[CarryT, Array]:
        """
        Call Function on Data with scan

        Parameters
        ----------
        fn : Callable
            Scan function
        init : TypeVar(Carry)
            Initial Carry

        Returns
        -------
        carry : TypeVar(Carry)
            Carry
        values : Array
            Return Values
        """
        def f(carry, i):
            carry, v = fn(carry, self._viget(i), self._voget(i))
            return carry, v

        return jax.lax.scan(f, init, self._bidx())

    def vmap(self, fn: Callable[[ArrayLike, ArrayLike], Array]) -> Array:
        """
        Call Function on Data with vmap

        Parameters
        ----------
        fn : Callable
            Vmap function

        Returns
        -------
        values : Array
            Return Values
        """
        @jax.vmap
        def f(i):
            return fn(self._viget(i), self._voget(i))

        return f(self._bidx())
