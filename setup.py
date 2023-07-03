from setuptools import setup, find_packages

extras_require = {
    "experiment": ["optax", "orbax", "tqdm"],
    "io": ["pandas", "pyarrow"],
    "oam": ["argparse_dataclass"],
    "dev": ["coverage", "mypy", "pandas-stubs", "types-tqdm", "unittest-xml-reporting"],
    "doc": ["sphinx", "sphinx-rtd-theme", "myst-parser"],
    "board": ["streamlit"],
}
extras_require["cli"] = list(
    set(extras_require["experiment"] +
        extras_require["io"] +
        extras_require["oam"])
)
extras_require["all"] = list(
    set(sum(extras_require.values(), []))
)

with open("README.md") as R:
    long_description = R.read()


setup(name="tsax",
      description="TSax: Time Series Forecasting",
      version="0.0.0",
      long_description=long_description,
      long_description_content_type="text/markdown",
      install_requires = [
          "jax <= 0.4.10", # Optax silently Requires because of jax.experimental
          "flax",
          "typing_extensions",
          "well-behaved-logging"
      ],
      extras_require=extras_require,
      packages=find_packages(),
      package_data={"tsax": ["py.typed"]})
