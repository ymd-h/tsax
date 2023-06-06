from setuptools import setup, find_packages

extras_require = {
          "experiment": ["optax", "orbax", "tqdm"],
          "io": ["pandas", "pyarrow"],
}
extras_require["cli"] = list(set(extras_require["experiment"] +
                                 extras_require["io"]))

setup(name="tsax",
      description="TSax: Time Series Forecasting",
      version="0.0.0",
      install_requires = ["jax", "flax", "well-behaved-logging"],
      extras_require=extras_require,
      packages=find_packages())
