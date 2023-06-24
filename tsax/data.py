from __future__ import annotations
from typing import cast, Callable, Generic, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map
import wblog

from tsax.typing import ArrayLike, Array, KeyArray, CarryT, DataT
from tsax.typed_jax import vmap

__all__ = [
    "SeqData",
    "ensure_SeqShape",
    "ensure_BatchSeqShape",
    "data_shape",
]


logger = wblog.getLogger()


def ensure_SeqShape(data: DataT) -> DataT:
    """
    Ensure Sequence Shape [L, d]

    Parameters
    ----------
    data : DataT

    Returns
    -------
    data : DataT
        Shape ensured Data. [L, d]

    Notes
    -----
    [L] -> [L, 1]
    """
    def f(d):
        if d.ndim == 1:
            d = jnp.reshape(d, (d.shape[0], 1))
        logger.debug("Ensured Shape: %s", d.shape)
        return d

    return cast(DataT, tree_map(f, data))


def ensure_BatchSeqShape(data: DataT) -> DataT:
    """
    Ensure Batch Sequence Shape [B, L, d]

    Parameters
    ----------
    data : DataT

    Returns
    -------
    data : DataT
        Shape ensured Data. [B, L, d]

    Notes
    -----
    [L] -> [1, L, 1]
    [L, d] -> [1, L, d]
    """
    def f(d):
        if d.ndim == 1:
            d = jnp.reshape(d, (1, d.shape[0], 1))
        elif d.ndim == 2:
            d = jnp.reshape(d, (1, *d.shape))

        return d

    return cast(DataT, tree_map(f, data))


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
        >>> d = jnp.arange(10, dtype=float)
        >>> seq = SeqData(d, xLen=2, yLen=2, batch_size=2)
        >>> seq.ibatch(0)
        (Array([[0., 1.],
                [1., 2.]], dtype=float32),
         Array([[2., 3.],
                [3., 4.]], dtype=float32))

        Togetter with categorical sequence

        >>> c = jnp.arange(10)
        >>> seqcat = SeqData((d, c), xLen=1, yLen=1, batch_size=2)
        >>> seqcat.ibatch(0)
        ((Array([[0.],
                 [1.]], dtype=float32),
          Array([[0],
                 [1]], dtype=int32)),
         (Array([[1.],
                 [2.]], dtype=float32),
          Array([[1],
                 [2]], dtype=int32)))
        """
        self.data: DataT = ensure_SeqShape(data) # [L, d]
        self.xLen: int = int(xLen)
        self.yLen: int = int(yLen)
        self.batch_size: int = int(batch_size)
        self.stride: int = int(stride)
        logger.info("SeqData(xLen=%d, yLen=%d, batch_size=%d, stride=%d)",
                    self.xLen, self.yLen, self.batch_size, self.stride)

        Len = tree_map(lambda d: d.shape[0], self.data)
        if not isinstance(Len, int):
            # DataT: tuple or list
            Len = _extract_1st(Len) # Support nested type

        self.idx: Array = jnp.arange(0, Len - self.xLen - self.yLen, self.stride)

        self.nbatch: int = self.idx.shape[0] // self.batch_size
        logger.debug("nbatch: %d", self.nbatch)

        _get = lambda L, b: lambda i: tree_map(
            lambda d: jax.lax.dynamic_slice(d,
                                            (i+b, *((0,) * (d.ndim-1))),
                                            (L, *d.shape[1:])),
            self.data
        )

        self._xget = _get(self.xLen, 0)
        self._yget = _get(self.yLen, self.xLen)

        self._vxget = vmap(self._xget)
        self._vyget = vmap(self._yget)

    def shuffle(self, key: KeyArray) -> None:
        """
        Shuffle

        Parameters
        ----------
        key : KeyArray
            PRNG Key
        """
        self.idx = jax.random.permutation(key, self.idx, independent=True)

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
        idx = self.idx.at[i*self.batch_size:(i+1)*self.batch_size].get()
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

    def vmap(self, fn: Callable[[DataT, DataT], Array]) -> Array:
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
        def f(i) -> Array:
            return fn(self._vxget(i), self._vyget(i))

        return f(self._bidx())

    def train_test_split(self,
                         test_size: Union[int, float]) -> Tuple[SeqData[DataT],
                                                                SeqData[DataT]]:
        """
        Split for Train / Test

        Parameters
        ----------
        test_size : int or float
            If it is ``int``, considered as the number of records.
            If it is ``float``, considered as ratio.

        Returns
        -------
        train : SeqData
        test : SeqData
        """
        if 0 < test_size < 1:
            Len = tree_map(lambda d: d.shape[0], self.data)
            if not isinstance(Len, int):
                # DataT: tuple or list
                Len, _ = tree_flatten(Len) # Support nested type
                Len = Len[0]
            test_size = int(Len * test_size)

        size: int = int(test_size)

        def f(d: DataT) -> SeqData[DataT]:
            return SeqData(d, xLen=self.xLen, yLen=self.yLen,
                           batch_size=self.batch_size, stride=self.stride)

        return (
            f(tree_map(lambda d: d.at[:-size].get(), self.data)),
            f(tree_map(lambda d: d.at[-size:].get(), self.data)),
        )

    def dimension(self) -> int:
        """
        Get Dimension of Sequence

        Returns
        -------
        d : int
            Data Dimension
        """
        return _extract_1st(self.data).shape[-1]


def _extract_1st(data: DataT) -> Array:
    d, _ = tree_flatten(data)
    return cast(Array, d[0])


def data_shape(data: Union[SeqData[DataT], DataT]) -> Tuple[int, ...]:
    """
    Get Shape

    Parameters
    ----------
    data : SeqData or ArrayLike
        Data

    Returns
    -------
    shape : tuple of ints
        Shape
    """
    if isinstance(data, SeqData):
        data, _ = data.ibatch(0)

    return _extract_1st(data).shape


def data_to_list(data: DataT) -> List[Array]:
    """
    Convert DataT to List of Array

    Parameters
    ----------
    data : DataT
        Maybe Structured Data

    Returns
    -------
    flattened : list of Array
        Flattened Data
    """
    d, _ = tree_flatten(data)
    return cast(List[Array], d)
