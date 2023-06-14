from setuptools import setup, find_packages

extras_require = {
    "experiment": ["optax", "orbax", "tqdm"],
    "io": ["pandas", "pyarrow"],
    "oam": ["argparse_dataclass"],
    "dev": ["coverage", "mypy", "pandas-stubs", "types-tqdm", "unittest-xml-reporting"],
}
extras_require["cli"] = list(
    set(extras_require["experiment"] +
        extras_require["io"] +
        extras_require["oam"])
)
extras_require["all"] = list(
    set(sum(extras_require.values(), []))
)

setup(name="tsax",
      description="TSax: Time Series Forecasting",
      version="0.0.0",
      install_requires = ["jax", "flax", "typing_extensions", "well-behaved-logging"],
      extras_require=extras_require,
      packages=find_packages(),
      package_data={"tsax": ["py.typed"]})
