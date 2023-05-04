# TSax

> **Warning**  
> This module is still under development, and doesn't work yet.

TSax is a Python package providing Time Series Forecasting Models.

The name comes from Time Series on
[JAX](https://jax.readthedocs.io/en/latest/index.html) /
[Flax](https://flax.readthedocs.io/en/latest/index.html)


## Development


### docstring
We obey [Numpy's Style guide](https://numpydoc.readthedocs.io/en/latest/format.html),
and the docstring is treated by
[Sphinx Napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).


### Annotation

Follow [JAX Typing Best Practices](https://jax.readthedocs.io/en/latest/jax.typing.html).
Use `jax.typing.ArrayLike` for inputs, `jax.Array` for outputs.
