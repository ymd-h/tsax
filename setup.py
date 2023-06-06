from setuptools import setup, find_packages

setup(name="tsax",
      description="TSax: Time Series Forecasting",
      version="0.0.0",
      install_requires = ["jax", "flax", "well-behaved-logging"],
      extras_require={
          "experiment": ["optax", "orbax", "tqdm"],
          "io": ["pandas", "pyarrow"],
      },
      packages=find_packages())
