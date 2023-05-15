from setuptools import setup, find_packages

setup(name="tsax",
      version="0.0.0",
      install_requires = ["jax", "flax", "well-behaved-logging"],
      packages=find_packages())
