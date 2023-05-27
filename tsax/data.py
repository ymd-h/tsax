from __future__ import annotations
from typing import Callable, Generic, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map

from tsax.typing import ArrayLike, Array, KeyArray, CarryT, DataT


class SeqData(Generic[DataT]):
    """
    Sequence Data
    """

    def __init__(self,
                 data: DataT,
                 xLen: int,
                 yLen: int,
                 batch_size: int,
                 stride: int = 1):
        """
        Initialize Sequence Data

        Parameters
        ----------
        data : ArrayLike, list of ArrayLike, tuple of ArrayLike
            Sequence Data. [L, d]
        xLen : int
            X Length
        yLen : int
            Y Length
        batch_size : int
            Batch Size
        stride : int, optional
            Stride Size

        Notes
        -----
        x[0] = [data[0], ..., data[xLen-1]]
        y[0] = [data[xLen], ..., data[xLen+yLen-1]]

        x[1] = [data[stride], ..., data[stride+xLen-1]]
        ...


        Examples
        --------
        >>> seq = SeqData(jnp.arange(10, dtype=float), xLen=2, yLen=2, batch_size=2)
        >>> seq.ibatch(0)
        (Array([[0., 1.],
                [1., 2.]], dtype=float32),
         Array([[2., 3.],
                [3., 4.]], dtype=float32))

        Togetter with categorical sequence

        >>> seqcat = SeqData((jnp.arange(10, dtype=float), jnp.arange(10)),
        ...                  xLen=1, yLen=1, batch_size=2)
        >>> seqcat.ibatch(0)
        """
        self.data: DataT = data # [L, d]
        self.xLen: int = int(xLen)
        self.yLen: int = int(yLen)
        self.batch_size: int = int(batch_size)
        self.stride: int = int(stride)

        Len = tree_map(lambda d: d.shape[0], self.data)
        if not isinstance(Len, int):
            # DataT: tuple or list
            Len, _ = tree_flatten(Len) # Support nested type
            Len = Len[0]
        self.idx: Array = jnp.arange(0, Len - self.xLen - self.yLen, self.stride)

        self.nbatch: int = self.idx.shape[0] // self.batch_size

        _get = lambda L, b: lambda i: tree_map(
            lambda d: jax.lax.dynamic_slice(d,
                                            (i+b, *((0,) * (d.ndim-1))),
                                            (L, *d.shape[1:])),
            self.data
        )

        self._xget = _get(self.xLen, 0)
        self._yget = _get(self.yLen, self.xLen)

        self._vxget = jax.vmap(self._xget)
        self._vyget = jax.vmap(self._yget)

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
        n = self.nbatch * self.batch_size
        return jnp.reshape(self.idx.at[:n].get(), (-1, self.batch_size))

    def ibatch(self, i: int) -> Tuple[DataT, DataT]:
        """
        Get i-th Batch

        Parameters
        ----------
        i : int
            i-th

        Returns
        -------
        x : DataT
        y : DataT
        """
        bstride = self.batch_size * self.stride
        idx = self.idx.at[i*bstride:(i+1)*bstride].get()
        return self._vxget(idx), self._vyget(idx)

    def scan(self,
             fn: Callable[[CarryT, DataT, DataT], Tuple[CarryT, Array]],
             init: CarryT) -> Tuple[CarryT, Array]:
        """
        Call Function over all batches with scan

        Parameters
        ----------
        fn : Callable
            Scan function
        init : CarryT
            Initial Carry

        Returns
        -------
        carry : CarryT
            Carry
        values : Array
            Return Values
        """
        def f(carry, i):
            carry, v = fn(carry, self._vxget(i), self._vyget(i))
            return carry, v

        return jax.lax.scan(f, init, self._bidx())

    def vmap(self, fn: Callable[[DataT, DataT], Array]) -> DataT:
        """
        Call Function over all batches with vmap

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
            return fn(self._vxget(i), self._vyget(i))

        return f(self._bidx())
