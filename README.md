# TSax

> **Warning**  
> This module is still under development, and doesn't work yet.

TSax is a Python package providing Time Series Forecasting Models.

The name comes from Time Series on
[JAX](https://jax.readthedocs.io/en/latest/index.html) /
[Flax](https://flax.readthedocs.io/en/latest/index.html)


## Algorithms

- [x] Transformer
  - [A. Vaswani et al., "Attention is All you Need", NIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html),
    [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
  - Notes: This is not intended for Time Series Forecasting, but for understanding of Transformer basics.
- [ ] Reformer
  - [N. Kitaev et al., "Reformer: The Efficient Transformer", ICLR 2020](https://openreview.net/forum?id=rkgNKkHtvB),
    [arXiv:2001.04451](https://arxiv.org/abs/2001.04451)
- [ ] (WIP) Informer
  - [H. Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021, Vol. 35, No. 12](https://ojs.aaai.org/index.php/AAAI/article/view/17325),
    [arXiv:2012.07436](https://arxiv.org/abs/2012.07436)
- [ ] (WIP) Autoformer
  - [H. Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting", NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html),
    [arXiv:2106.13008](https://arxiv.org/abs/2106.13008)
- [ ] FEDformer
  - [T. Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting", ICML 2022](https://proceedings.mlr.press/v162/zhou22g.html)
    [aeXiv:2201.12740](https://arxiv.org/abs/2201.12740)
- ...


## Avoid to Vanishing Gradient
- [ ] ResiDual
  - [S. Xie et al., "ResiDual: Transformer with Dual Residual Connections", arXiv:2304.14802](https://arxiv.org/abs/2304.14802)
- [ ] Admin
  - [L. Liu et al., "Understanding the Difficulty of Training Transformers", Empirical Methods in Natural Language Processing (EMNLP), (2020)](https://aclanthology.org/2020.emnlp-main.463/)
    [arXiv:2004.08249](https://arxiv.org/abs/2004.08249)
- [x] Sin2-max-shifted, Sin-Softmax
    - [S. Wang et al., "Escaping the Gradient Vanishing: Periodic Alternatives of Softmax in Attention Mechanism", IEEE Access Vol.9, 168749--168759 (2021), doi: 10.1109/ACCESS.2021.3138201](https://ieeexplore.ieee.org/document/9662308)
      [arXiv:2108.07153](https://arxiv.org/abs/2108.07153)
- ...

## Development


### docstring
We obey [Numpy's Style guide](https://numpydoc.readthedocs.io/en/latest/format.html),
and the docstring is treated by
[Sphinx Napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).


### Annotation

Follow [JAX Typing Best Practices](https://jax.readthedocs.io/en/latest/jax.typing.html).
Use `jax.typing.ArrayLike` for inputs, `jax.Array` for outputs.
